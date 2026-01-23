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
#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif
#pragma once
#include <cstdlib>
#include <complex>
#include "base.hpp"
#include <boost/dynamic_bitset.hpp>

const std::complex<double> OPER_ELEMENTS[28] = {
    {1, 0}, {0, 0}, {0, 0}, {-1, 0}, // Z
    {1, 0},
    {0, 0},
    {0, 0},
    {0, 0}, // 0
    {0, 0},
    {0, 0},
    {0, 0},
    {1, 0}, // 1
    {0, 0},
    {1, 0},
    {1, 0},
    {0, 0}, // X
    {0, 0},
    {0, -1},
    {0, 1},
    {0, 0}, // Y
    {0, 0},
    {1, 0},
    {0, 0},
    {0, 0}, // -
    {0, 0},
    {0, 0},
    {1, 0},
    {0, 0} // +
};

const double REAL_OPER_ELEMENTS[28] = {
    1, 0, 0, -1, // Z
    1, 0, 0, 0,  // 0
    0, 0, 0, 1,  // 1
    0, 1, 1, 0,  // X
    0, -1, 1, 0, // Y
    0, 1, 0, 0,  // -
    0, 0, 1, 0   // +
};

/**
 * Accumulate the matrix element value for given row and column bit-strings
 *
 *
 * @param row The row bit-string
 * @param col The col bit-string
 * @param pos The positions (qubits) of the non-idenity operators in ther term
 * @param val The char value for each operator
 * @param coeff The complex coefficient of the term in question
 * @param N The length of the pos and val vector, i.e. number of non-ID operators in term
 * @param out The complex number to accumulate to
 */
template <typename T>
void accum_element(const boost::dynamic_bitset<std::size_t> &row,
                   const boost::dynamic_bitset<std::size_t> &col,
                   const unsigned int *__restrict inds,
                   const unsigned char *__restrict val,
                   const std::complex<double> &coeff,
                   const int real_phase,
                   const unsigned int N,
                   T &out)
{
    T temp = 1.0;
    unsigned int kk, pos, block_num, block_idx;
    std::size_t offset, row_int, col_int;
    for (kk = 0; kk < N; kk++)
    {
        pos = inds[kk];
        block_num = pos / BITS_PER_BLOCK;
        block_idx = pos % BITS_PER_BLOCK;
        row_int = (row.m_bits[block_num] >> block_idx) & 1;
        col_int = (col.m_bits[block_num] >> block_idx) & 1;
        offset = 2 * row_int + col_int;
        if constexpr (std::is_same_v<T, double>)
        {
            temp *= REAL_OPER_ELEMENTS[4 * val[kk] + offset];
        }
        else
        {
            temp *= OPER_ELEMENTS[4 * val[kk] + offset];
        }
    } // end for-loop
    // accumulate to output value
    if constexpr (std::is_same_v<T, double>)
    {
        out += real_phase * coeff.real() * temp;
    }
    else
    {
        out += coeff * temp;
    }
}
