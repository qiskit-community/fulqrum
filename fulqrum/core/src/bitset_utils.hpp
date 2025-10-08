#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif
#pragma once
#include <vector>
#include <algorithm>
#include "base.hpp"
#include "constants.hpp"
#include <boost/dynamic_bitset.hpp>



/**
 * Flip a series of bits in a bitset
 *
 * @param bitset The target bitset
 * @param arr Pointer to array of indices which to flip
 * @param size The size of the array
 */
inline void flip_bits(boost::dynamic_bitset<std::size_t> &bitset,
                      const unsigned int *__restrict arr, const unsigned int size)
{
    unsigned int kk;
    unsigned int pos;
    for (kk = 0; kk < size; kk++)
    {
        pos = arr[kk];
        bitset.m_bits[pos >> BLOCK_EXPONENT] ^= ((size_t(1) << (pos & BLOCK_SHIFT)));
    }
}

/**
 * Gets the column bitset from an input row bitset and operator term
 *
 * @param col The input row bitset that will be converted to column
 * @param pos Array of indices on which operators act
 * @param val The value representing each operator
 * @param N Number of non-ID operators in the term
 */
inline void get_column_bitset(boost::dynamic_bitset<std::size_t> &col,
                              const unsigned int *__restrict pos,
                              const unsigned char *__restrict val,
                              const unsigned int N)
{
    unsigned int block_num, block_idx;
    unsigned int ind;
    unsigned int kk;
    for (kk = 0; kk < N; kk++)
    {
        ind = pos[kk];
        block_num = ind / BITS_PER_BLOCK;
        block_idx = ind % BITS_PER_BLOCK;
        col.m_bits[block_num] = col.m_bits[block_num] ^ (std::size_t)((val[kk] > 2) << block_idx);
    }
}

inline void bitset_column_index(const std::size_t start, const std::size_t stop,
                                const boost::dynamic_bitset<std::size_t> &col,
                                const std::vector<boost::dynamic_bitset<std::size_t>> &subspace,
                                std::size_t &col_idx)
{
    std::size_t kk;
    col_idx = MAX_SIZE_T;
    for (kk = start; kk < stop; kk++)
    {
        if (col == subspace[kk])
        {
            col_idx = kk;
            break;
        }
    }
}


/**
 * Convert bits at given indices into an unsigned integer
 *
 * @param row A pointer to a vector that is an alternate representation of row bitset
 * @param inds Pointer to array of indices as unsigned ints
 * @param num_bits The number of bits to consider
 */
inline unsigned int bitset_ladder_int(const uint8_t *row,
                                      const unsigned int *__restrict inds,
                                      const unsigned int num_bits)
{
    std::size_t row_int, out_int = 0;
    unsigned int kk, pos;

    for (kk = 0; kk < num_bits; kk++)
    {
        pos = inds[kk];
        row_int = row[pos];
        out_int |= (row_int << kk);
    }

    return out_int;
}


/**
 * verifies that bitstring passes constraints of the term projection operators
 *
 * @param bitset The target row bitset
 * @param values The values representing the type of operator
 * @param proj_indices Pointer to array of indices on which projectors act
 * @param size The size of the proj array
 */
inline unsigned int passes_proj_validation(const OperatorTerm_t *__restrict term,
                                           const boost::dynamic_bitset<std::size_t> &bitset)
{
    unsigned int kk;
    unsigned int block_num, block_idx;
    unsigned int pos;
    unsigned int bit;
    for (kk = 0; kk < term->proj_indices.size(); kk++)
    {
        pos = term->proj_indices[kk];
        block_num = pos >> BLOCK_EXPONENT;
        block_idx = pos & BLOCK_SHIFT;
        bit = ((bitset.m_bits[block_num] >> block_idx) & std::size_t(1));
        if (bit != term->proj_bits[kk])
        {
            return 0;
        }
    }
    return 1;
}
