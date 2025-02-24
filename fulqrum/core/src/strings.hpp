/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <string>


/**
 * Return the column bit-string for a given operator term and row.
 *
 * Computes column bit-string via branchless XOR 
 *
 * @param row The row bit-string
 * @param bit_len The length of the bit-strings
 * @param pos The positions (qubits) of the non-idenity operators in ther term
 * @param val The char value for each operator
 * @param N The length of the pos and val vector, i.e. number of non-ID operators in term
 * @return Column string
 */
inline std::string get_column_str(const char * __restrict row,
                                  std::size_t bit_len,
                                  const std::size_t * __restrict pos,
                                  const unsigned char * __restrict val,
                                  std::size_t N)
        {
            std::string column(row, bit_len);
            int sign;
            std::size_t idx;
            for (std::size_t kk = 0; kk < N; kk++)
                {
                    idx = bit_len - pos[kk] - 1; // Need to flip index for LSB ordering
                    sign = ~((2 - val[kk]) >> 8 * sizeof(char)) + 1;
                    column[idx] = row[idx] ^ sign;
                }
            return column;
        }

