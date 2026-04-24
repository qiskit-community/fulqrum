/**
 * This code is part of Fulqrum.
 *
 * (C) Copyright IBM 2024.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */
#pragma once
#include <complex>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

typedef unsigned int width_t;

const double ATOL = 1e-14;
const std::size_t MAX_SIZE_T = (std::size_t)-1;
const unsigned int MAX_UINT = (unsigned int)-1;
const unsigned int BITS_PER_BLOCK = 8 * sizeof(std::size_t);
const unsigned int DEFAULT_LADDER_WIDTH = 4;
const unsigned int BLOCK_EXPONENT = __builtin_ctz(BITS_PER_BLOCK);
const unsigned int BLOCK_SHIFT = BITS_PER_BLOCK - 1;

typedef std::tuple<std::string, std::vector<width_t>, std::complex<double>> TermData;
typedef std::tuple<std::string, std::vector<width_t>> OpData;

// Map converting standard char values into continuous values
inline std::unordered_map<unsigned char, unsigned char> oper_map = {
    {90, 0}, {48, 1}, {49, 2}, {88, 3}, {89, 4}, {45, 5}, {43, 6}};

// Reverse map back to standard char values
inline std::unordered_map<unsigned char, unsigned char> rev_oper_map = {
    {0, 90}, {1, 48}, {2, 49}, {3, 88}, {4, 89}, {5, 45}, {6, 43}};

/**
 * Validate that term indices are less than operator width
 *
 * @param[in] indices Indices for the given term
 * @param[in] width The operator width
 */
inline void _validate_indices(std::vector<width_t>& inds, width_t width)
{
    std::size_t size = inds.size();
    for(std::size_t kk = 0; kk < size; kk++)
    {
        if(inds[kk] >= width)
        {
            throw std::runtime_error("Index is larger than the operator width.");
        }
    }
}
