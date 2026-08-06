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
#include <array>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

typedef uint16_t width_t;

const double ATOL = 1e-14;
const std::size_t MAX_SIZE_T = (std::size_t)-1;
const unsigned int MAX_UINT = (unsigned int)-1;
const width_t MAX_WIDTH = (width_t)-1;
const unsigned int BITS_PER_BLOCK = 8 * sizeof(std::size_t);
const unsigned int DEFAULT_LADDER_WIDTH = 2;
const unsigned int BLOCK_EXPONENT = __builtin_ctz(BITS_PER_BLOCK);
const unsigned int BLOCK_SHIFT = BITS_PER_BLOCK - 1;

typedef std::tuple<std::string, std::vector<width_t>, std::complex<double>> TermData;
typedef std::tuple<std::string, std::vector<width_t>> OpData;

// Maps operator standard char values into continuous values used internally.
// Unused values are set to 0xFF for clarity that they really do nothing.
// Mapping: 'Z'=90->0, '0'=48->1, '1'=49->2, 'X'=88->3, 'Y'=89->4, '-'=45->5, '+'=43->6
inline constexpr std::array<unsigned char, 91> oper_map = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, // 0-9
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, // 10-19
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, // 20-29
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, // 30-39
    0xFF,0xFF,0xFF,   6,0xFF,   5,0xFF,0xFF,   1,   2, // 40-49 [+=43,  -=45,  0=48, 1=49]
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, // 50-59
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, // 60-69
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF, // 70-79
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,   3,   4, // 80-89 [X=88, Y=89]
    0};                                                // 90    [Z=90]


// Reverse operator map to get back to standard char values for things like printing out
// operators
inline constexpr std::array<unsigned char, 7> rev_oper_map = {
    90, 48, 49, 88, 89, 45, 43};

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
