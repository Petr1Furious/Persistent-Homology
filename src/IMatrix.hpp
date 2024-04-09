#pragma once

#include <vector>

class IMatrix
{
public:
    virtual ~IMatrix() = default;

    virtual std::vector<uint32_t> reduce(bool run_twist = true) = 0;

    virtual size_t size() const = 0;
};