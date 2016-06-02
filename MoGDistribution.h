//
// Created by jan on 01.06.16.
//

#ifndef SHEET_4_MOGDISTRIBUTION_H
#define SHEET_4_MOGDISTRIBUTION_H

#include <array>
#include <Eigen/Dense>

/**
 * This is a mixture-of-gaussians distribution. It can be trained from data.
 * @tparam K Amount of gaussians
 * @tparam N Dimensionality of the data
 * @tparam T Type of the data, defaults to double
 */
template<unsigned int K, unsigned int N, typename T = double>
class MoGDistribution
{
public:
    using Data = Eigen::Matrix<T, N, 1>;
    using Cov = Eigen::Matrix<T, N, N>;

    /**
     * Initializes this distribution with default values
     */
    MoGDistribution();

    /**
     * Trains the distribution based on some data
     * @param begin Iterator to the beginning of a collection of type Data
     * @param end Iterator the end of a collection of type Data
     * @param eps Termination criterion. Log-likelihood change must be less than this number
     */
    template<typename Iterator>
    void train(Iterator begin, Iterator end, T eps);

    /**
     * Provides the confidence of a data point
     * @param x Data point
     * @return The confidence of the passed data point
     */
    T confidence(Data const& x) const;

    /**
     * Stores the current configuration to a file
     * @param filename Name of the file to write to
     * @returns True in case saving was successful, otherwise false
     */
    bool save(std::string const& filename) const;

    /**
     * Loads a configuration from a file
     * @param filename Name of the file to load from
     * @return True in case loading was successful, otherwise false
     */
    bool load(std::string const& filename);

    /**
     * Prints the current configuration to stdout
     */
    void print() const;

private:
    std::array<T, K> m_pi;    // Mixing coefficients
    std::array<Data, K> m_mu; // Means
    std::array<Cov, K> m_cov; // Covariances
    std::default_random_engine m_rand;

    T const pi = 3.1415926535897932385;

    template<typename Iterator>
    void reinitialize(Iterator begin, Iterator end);

    T evalGaussian(Data const& x, Data const& mu, Cov const& cov) const;

    T responsibility(unsigned int k, Data const& x) const;

    template<typename Iterator>
    T logLikelihood(Iterator begin, Iterator end);
};

#include "MoGDistribution.hh"

#endif //SHEET_4_MOGDISTRIBUTION_H
