/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <cstring>

const std::size_t MAX_SIZE_T = (std::size_t)-1;


/**
 * Convert a standard char string to binary char string
 *
 * Used for converting Python dict elements to internal vector
 *
 * @param in_string Input bit-string
 * @param out_string Output string
 * @param num_qubits Width of the strings
 */
inline void string_to_vec(const char * in_string, 
                          unsigned char * out_string, 
                          std::size_t num_qubits)
    {
    for(std::size_t kk=0; kk < num_qubits; kk++)
        {
        out_string[kk] = in_string[kk] - 48;
        }
    }

/**
 * Convert the bin of a bit-string to an integer
 *
 * @param vec bit-string vector
 * @param num_qubits Width of the string
 * @param bin_width Number of bits to evaulate for bin
 *
 * @return Integer value of bin bits
 */
inline int bin_width_to_int(const unsigned char *__restrict vec,
                            std::size_t num_qubits,
                            std::size_t bin_width)
	{
    int val = 0;
    #pragma omp simd reduction(+:val)
    for(std::size_t kk=0; kk < bin_width; kk++)
    	{
        val += vec[num_qubits-kk-1]*(1 << kk);
    	}
    return val;
	}

/**
 * Determin the column index from start and stop
 * values for the located bin
 *
 * @param start Start index of the bin
 * @param stop Stop index of the bin
 * @param col Column bit-string
 * @param subspace The whole subspace vector
 * @param num_qubits Width of the strings
 *
 * @return Integer value of bin bits
 */
inline std::size_t col_index(std::size_t start, std::size_t stop,
                             const unsigned char * col, 
                             const unsigned char * subspace,
                             std::size_t num_qubits)
    {
    int val;
    for(std::size_t kk=start; kk < stop; kk++)
        {
        val = memcmp(col, &subspace[kk*num_qubits], num_qubits);
        if(val == 0)
            {
            return kk;
            }
        }
  	return MAX_SIZE_T;
    }


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
inline void get_column_vec(const unsigned char * row,
                           unsigned char * col,
                           std::size_t bit_len,
                           const std::size_t * pos,
                           const unsigned char * val,
                           std::size_t N)
        {
            std::size_t idx;
            for (std::size_t kk = 0; kk < N; kk++)
                {
                    idx = bit_len - pos[kk] - 1; // Need to flip index for LSB ordering
                    col[idx] = row[idx] ^ (val[kk] > 2);
                }
        }