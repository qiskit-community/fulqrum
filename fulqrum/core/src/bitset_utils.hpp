#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif
#pragma once
#include <boost/dynamic_bitset.hpp>

const std::size_t BITS_PER_BLOCK = 8 * sizeof(std::size_t);

/**
 * Compute the integer corresponding to the bin-width of a bitset
 *
 * @param bitset The target bitset
 * @param bin_width The bin-width
 * @param res The resulting integer
 */
inline void bin_int(const boost::dynamic_bitset<std::size_t>& bitset, 
                    unsigned int bin_width, std::size_t& res)
    {
        res = bitset.m_bits[0] & (( 1ULL << bin_width) - 1);
    }


/**
 * Flip a series of bits in a bitset
 *
 * @param bitset The target bitset
 * @param arr Pointer to array of indices which to flip
 * @param size The size of the array
 */
inline void flip_bits(boost::dynamic_bitset<std::size_t>& bitset,
                      std::size_t * arr, std::size_t size)
    {
        std::size_t kk;
        std::size_t block_num, block_idx;
        std::size_t pos;
        for(kk=0; kk < size; kk++)
        {
            pos = arr[kk];
            block_num = pos / BITS_PER_BLOCK;
            block_idx = pos % BITS_PER_BLOCK;
            bitset.m_bits[block_num] = bitset.m_bits[block_num] ^ (1ULL << block_idx);
        }
    }



inline void get_column_bitset(const boost::dynamic_bitset<std::size_t>& row,
                              boost::dynamic_bitset<std::size_t>& col,
                              const std::size_t bit_len,
                              const std::size_t *__restrict pos,
                              const unsigned char *__restrict val,
                              const std::size_t N)
{
    std::size_t block_num, block_idx;
    std::size_t ind;
    for (std::size_t kk = 0; kk < N; kk++)
        {
            ind = pos[kk];
            block_num = ind / BITS_PER_BLOCK;
            block_idx = ind % BITS_PER_BLOCK;
            col.m_bits[block_num] = row.m_bits[block_num] ^ ((std::size_t)(val[kk] > 2) << block_idx);
        }
}