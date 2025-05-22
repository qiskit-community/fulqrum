#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif
#pragma once
#include <boost/dynamic_bitset.hpp>

const std::size_t BITS_PER_BLOCK = 8 * sizeof(std::size_t);


inline void bin_int(const boost::dynamic_bitset<std::size_t>& b, unsigned int len, std::size_t& res)
    {
        res = b.m_bits[0] & (( 1ULL << len) - 1);
    }


inline void flip_bits(boost::dynamic_bitset<std::size_t>& b, std::size_t * arr, std::size_t size)
    {
        std::size_t kk;
        std::size_t block_num, block_idx;
        std::size_t pos;
        for(kk=0; kk < size; kk++)
        {
            pos = arr[kk];
            block_num = pos / BITS_PER_BLOCK;
            block_idx = pos % BITS_PER_BLOCK;
            b.m_bits[block_num] = b.m_bits[block_num] ^ (1ULL << block_idx);
        }
    }