//
// Created by jan on 01.06.16.
//

#ifndef SHEET_4_STRINGHELPER_H
#define SHEET_4_STRINGHELPER_H

#include <cstdio>
#include <memory>

namespace StringHelper
{
    template<typename ... Args>
    static std::string format(std::string const& format, Args ... args)
    {
        size_t size = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
        std::unique_ptr<char[]> buf(new char[size]);
        std::snprintf(buf.get(), size, format.c_str(), args ...);
        return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
    }
}

#endif //SHEET_4_STRINGHELPER_H
