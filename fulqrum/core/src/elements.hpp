/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif
#pragma once
#include <cstdlib>
#include <complex>
#include "base.hpp"
#include <boost/dynamic_bitset.hpp>



const std::complex<double> OPER_ELEMENTS[28] = {{1,0}, {0,0}, {0,0}, {-1,0},   // Z
                                                {1,0}, {0,0}, {0,0}, {0,0},    // 0
                                                {0,0}, {0,0}, {0,0}, {1,0},    // 1
                                                {0,0}, {1,0}, {1,0}, {0,0},    // X
                                                {0,0}, {0,-1}, {0,1}, {0,0},   // Y
                                                {0,0}, {1,0}, {0,0}, {0,0},    // -
                                                {0,0}, {0,0}, {1,0}, {0,0}     // +
                                               };

const double REAL_OPER_ELEMENTS[28] = {1, 0, 0, -1,   // Z
                                       1, 0, 0, 0,    // 0
                                       0, 0, 0, 1,    // 1
                                       0, 1, 1, 0,    // X
                                       0, 0, 0, 0,   // Y (placeholder)
                                       0, 1, 0, 0,    // -
                                       0, 0, 1, 0     // +
                                       };


/**
 * Accumulate the COMPLEX matrix element value for given row and column bit-strings
 *
 *
 * @param row The row bit-string
 * @param col The col bit-string
 * @param pos The positions (qubits) of the non-idenity operators in ther term
 * @param val The char value for each operator
 * @param coeff The complex coefficient of the term in question
 * @param N The length of the pos and val vector, i.e. number of non-ID operators in term
 * @param out The complex number to accumulate to
 * @return Column string
 */
template <typename T> void accum_element(const boost::dynamic_bitset<std::size_t>& row,
                                        const boost::dynamic_bitset<std::size_t>& col,
                                        const unsigned int *__restrict inds,
                                        const unsigned char *__restrict val,
                                        const std::complex<double>& coeff,
                                        const unsigned int N,
                                        T & out)
{
        T temp = 1.0;
        unsigned int kk, pos, block_num, block_idx;
        std::size_t offset, row_int, col_int;
        for(kk=0; kk<N; kk++)
        {
            pos = inds[kk];
            block_num = pos / BITS_PER_BLOCK;
            block_idx = pos % BITS_PER_BLOCK;
            row_int = (row.m_bits[block_num] >> block_idx) & 1;
            col_int = (col.m_bits[block_num] >> block_idx) & 1;
            offset = 2*row_int + col_int;
            if constexpr(std::is_same_v<T, double>)
            {
                temp *= REAL_OPER_ELEMENTS[4*val[kk] + offset];
            }
            else
            {
                temp *= OPER_ELEMENTS[4*val[kk] + offset];
            }
        } // end for-loop
        // accumulate to output value
        if constexpr(std::is_same_v<T, double>)
        {
            out += coeff.real()*temp;
        }
        else
        {
            out += coeff*temp;
        }
}
