#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif
#pragma once
#include <boost/dynamic_bitset.hpp>


inline void bin_int(const boost::dynamic_bitset<std::size_t>& b, unsigned int len, std::size_t& res)
    {
        res = b.m_bits[0] & (( 1ULL << len) - 1);
    }
