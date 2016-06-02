//
// Created by jan on 01.06.16.
//

#include "MoGDistribution.h"
#include "StringHelper.h"
#include <random>
#include <fstream>

template<unsigned int K, unsigned int N, typename T>
MoGDistribution<K, N, T>::MoGDistribution()
{
    // Initialize means with zero
    m_mu.fill(Data::Zero());
    // Initialize covariances with non-singular matrices
    m_cov.fill(Cov::Identity() * 100);
    // Initialize coefficients with 1
    m_pi.fill(1);
}

template<unsigned int K, unsigned int N, typename T>
template<typename Iterator>
void MoGDistribution<K, N, T>::train(Iterator begin, Iterator end, T eps)
{
    size_t const M = std::distance(begin, end);
    reinitialize(begin, end);

    T prevLogLikelihood = 0;
    unsigned int iter = 0;
    while (true)
    {
        std::cout << "Iteration " << iter + 1 << std::endl;

        decltype(m_mu) mu_new;
        decltype(m_cov) cov_new;
        decltype(m_pi) pi_new;

        // Sum of responsibilities per k
        std::array<T, K> sumResp;
        for (unsigned int k = 0; k < K; k++)
        {
            sumResp[k] = responsibility(k, *begin);
            for (unsigned int m = 1; m < M; m++)
                sumResp[k] += responsibility(k, *(begin + m));
        }

        // Re-estimate parameters
        for (unsigned int k = 0; k < K; k++)
        {
            // New means
            mu_new[k] = *begin * responsibility(k, *begin);
            for (unsigned int m = 1; m < M; m++)
                mu_new[k] += *(begin + m) * responsibility(k, *(begin + m));
            mu_new[k] /= sumResp[k];

            // New covariance
            cov_new[k] = responsibility(k, *begin) * (*begin - mu_new[k]) * (*begin - mu_new[k]).transpose();
            for (unsigned int m = 1; m < M; m++)
                cov_new[k] += responsibility(k, *(begin + m)) * (*(begin + m) - mu_new[k]) *
                              (*(begin + m) - mu_new[k]).transpose();
            cov_new[k] /= sumResp[k];

            // New coefficient
            pi_new[k] = sumResp[k] / M;

            m_mu[k] = mu_new[k];
            m_cov[k] = cov_new[k];
            m_pi[k] = pi_new[k];
        }

        // Check if any covariance became non positive-definite, or nans
        bool reinit = false;
        for (unsigned int k = 0; k < K; k++)
        {
            if (std::isnan(m_pi[k]) || m_cov[k].determinant() <= 0)
            {
                std::cout << "-------------- Reinit! --------------" << std::endl;
                if (std::isnan(m_pi[k]))
                    std::cout << " pi_" << k << " was NaN." << std::endl;
                if (m_cov[k].determinant() <= 0)
                    std::cout << " cov_" << k << " was not positive definite." << std::endl;
                std::cout << "-------------------------------------" << std::endl;

                reinitialize(begin, end);
                prevLogLikelihood = 0;
                reinit = true;
                break;
            }
        }

        T logLikeli = logLikelihood(begin, end);
        std::cout << "log-likelihood change: " << std::abs(prevLogLikelihood - logLikeli) << std::endl;
        if (std::abs(logLikeli - prevLogLikelihood) < eps && !reinit)
            break;
        prevLogLikelihood = logLikeli;

        iter++;
    }

    std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
}

template<unsigned int K, unsigned int N, typename T>
T MoGDistribution<K, N, T>::confidence(Data const& x) const
{
    T confidence = 0;
    for (unsigned int k = 0; k < K; k++)
    {
        T gaussian = evalGaussian(x, m_mu[k], m_cov[k]);
        confidence += m_pi[k] * gaussian;
    }
    return confidence;
}

template<unsigned int K, unsigned int N, typename T>
bool MoGDistribution<K, N, T>::save(std::string const& filename) const
{
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if (out.is_open())
    {
        out << "GMM";
        unsigned int K_ = K;
        out.write(reinterpret_cast<char const*>(&K_), sizeof(K_));
        unsigned int N_ = N;
        out.write(reinterpret_cast<char const*>(&N_), sizeof(N_));
        char const typeSize = sizeof(T);
        out.write(&typeSize, sizeof(typeSize));
        for (auto const& pi : m_pi)
            out.write(reinterpret_cast<char const*>(&pi), sizeof(pi));
        for (auto const& mu : m_mu)
        {
            for (unsigned int i = 0; i < N; i++)
                out.write(reinterpret_cast<char const*>(&mu(i)), sizeof(mu(i)));
        }
        for (auto const& cov : m_cov)
        {
            for (unsigned int i = 0; i < N; i++)
            {
                for (unsigned int j = 0; j < N; j++)
                    out.write(reinterpret_cast<char const*>(&cov(j, i)), sizeof(cov(j, i)));
            }
        }
        out.close();
        return true;
    }
    return false;
}

template<unsigned int K, unsigned int N, typename T>
bool MoGDistribution<K, N, T>::load(std::string const& filename)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (in.is_open())
    {
        char identifier[4];
        in.readsome(identifier, 3);
        identifier[3] = '\0';
        unsigned int K_, N_;
        char typeSize;
        in.read(reinterpret_cast<char*>(&K_), sizeof(K_));
        in.read(reinterpret_cast<char*>(&N_), sizeof(N_));
        in.read(&typeSize, sizeof(typeSize));
        if (std::string(identifier) != "GMM" || K != K_ || N != N_ || typeSize != sizeof(T))
        {
            in.close();
            std::cout << "The loaded file is corrupt. Identifier: \"" << identifier << "\". K: " << K_ << ". N: " << N_
                      << ". Size: " << (int) typeSize << "." << std::endl;
            return false;
        }
        for (auto& pi : m_pi)
            in.read(reinterpret_cast<char*>(&pi), sizeof(pi));
        for (auto& mu : m_mu)
        {
            for (unsigned int i = 0; i < N; i++)
            {
                T e;
                in.read(reinterpret_cast<char*>(&e), sizeof(mu(i)));
                mu(i) = e;
            }
        }
        for (auto& cov : m_cov)
        {
            for (unsigned int i = 0; i < N; i++)
            {
                for (unsigned int j = 0; j < N; j++)
                {
                    T e;
                    in.read(reinterpret_cast<char*>(&e), sizeof(e));
                    cov(j, i) = e;
                }
            }
        }

        in.close();
        return true;
    }
    return false;
}

template<unsigned int K, unsigned int N, typename T>
void MoGDistribution<K, N, T>::print() const
{
    // Print pi
    std::cout << " | ";
    for (unsigned int k = 0; k < K; k++)
        std::cout << "pi_" << k << " = " << StringHelper::format("%*f", 12, m_pi[k]) << " | ";
    std::cout << std::endl << std::endl;

    // Print mu
    for (unsigned int n = 0; n < N; n++)
    {
        std::cout << " | ";
        for (unsigned int k = 0; k < K; k++)
        {
            if (n == N / 2)
                std::cout << "mu_" << k << " = ";
            else
                std::cout << "       ";
            std::cout << "(" << StringHelper::format("%*f", 12, m_mu[k](n)) << ") | ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Print cov
    for (unsigned int n = 0; n < N; n++)
    {
        std::cout << " | ";
        for (unsigned int k = 0; k < K; k++)
        {
            if (n == N / 2)
                std::cout << "cov_" << k << " = ";
            else
                std::cout << "        ";
            std::cout << "( ";
            for (unsigned int n2 = 0; n2 < N; n2++)
                std::cout << StringHelper::format("%*f", 12, m_cov[k](n, n2)) << " ";
            std::cout << ") | ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

template<unsigned int K, unsigned int N, typename T>
template<typename Iterator>
void MoGDistribution<K, N, T>::reinitialize(Iterator begin, Iterator end)
{
    std::uniform_int_distribution<unsigned int> distribution(0, std::distance(begin, end));

    for (auto& mu : m_mu)
        mu = *(begin + distribution(m_rand));
    // Initialize covariances with non-singular matrices
    m_cov.fill(Cov::Identity() * 100);
    // Initialize coefficients with 1
    m_pi.fill(1. / K);
};

template<unsigned int K, unsigned int N, typename T>
T MoGDistribution<K, N, T>::evalGaussian(Data const& x, Data const& mu, Cov const& cov) const
{
    Data diff = x - mu;
    T factor = -0.5;
    T inner = factor * diff.transpose() * cov.inverse() * diff;
    assert(inner <= 0);
    T det = cov.determinant();
    T denom = std::sqrt(std::pow(2 * pi, N) * det);
    //assert(denom != 0);
    T exponential = std::exp(inner);
    T result = exponential / denom;
    return result;
}

template<unsigned int K, unsigned int N, typename T>
T MoGDistribution<K, N, T>::responsibility(unsigned int k, Data const& x) const
{
    T enumerator = m_pi[k] * evalGaussian(x, m_mu[k], m_cov[k]);
    T denominator = 0;
    for (unsigned int l = 0; l < K; l++)
        denominator += m_pi[l] * evalGaussian(x, m_mu[l], m_cov[l]);
    //assert(denominator != 0);
    return enumerator / denominator;
}

template<unsigned int K, unsigned int N, typename T>
template<typename Iterator>
T MoGDistribution<K, N, T>::logLikelihood(Iterator begin, Iterator end)
{
    T result = 0;

    for (size_t n = 0; n < std::distance(begin, end); n++)
    {
        T inner = 0;
        for (unsigned int k = 0; k < K; k++)
            inner += m_pi[k] * evalGaussian(*(begin + n), m_mu[k], m_cov[k]);
        result += std::log(inner);
    }

    return result;
}
