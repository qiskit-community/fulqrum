/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <complex>


const std::complex<double> OPER_ELEMENTS[28] = {{1,0}, {0,0}, {0,0}, {-1,0},   // Z
                                                {1,0}, {0,0}, {0,0}, {0,0},    // 0
                                                {0,0}, {0,0}, {0,0}, {1,0},    // 1
                                                {0,0}, {1,0}, {1,0}, {0,0},    // X
                                                {0,0}, {0,-1}, {0,1}, {0,0},   // Y
                                                {0,0}, {1,0}, {0,0}, {0,0},    // -
                                                {0,0}, {0,0}, {1,0}, {0,0}     // +
                                               };


/**
 * Return the matrix element value for given row and column bit-strings
 *
 *
 * @param row The row bit-string
 * @param col The col bit-string
 * @param bit_len The length of the bit-strings
 * @param pos The positions (qubits) of the non-idenity operators in ther term
 * @param val The char value for each operator
 * @param coeff The complex coefficient of the term in question
 * @param N The length of the pos and val vector, i.e. number of non-ID operators in term
 * @return Column string
 */
inline std::complex<double> compute_element_vec(const unsigned char * __restrict row,
                                            const unsigned char * __restrict col,
                                            std::size_t bit_len,
                                            const std::size_t * __restrict pos,
                                            const char * __restrict val,
                                            std::complex<double> coeff,
                                            std::size_t N)
    {
        std::complex<double> out = 1.0;
        std::size_t kk;
        int offset;
        for(kk=0; kk<N; kk++){
            offset = 2*(row[bit_len - pos[kk] - 1]) + (col[bit_len - pos[kk] - 1]);
            out *= OPER_ELEMENTS[4*val[kk] + offset];
        }
        return coeff*out;
    }
