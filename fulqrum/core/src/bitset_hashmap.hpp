#pragma once

#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif

#include <cassert>
#include <boost/dynamic_bitset.hpp>
#include "constants.hpp"
#include "./external/hash_table8.hpp"
#include "./external/rapidhash.h"


struct BitsetHasherRapid
{
    std::size_t operator()(const boost::dynamic_bitset<std::size_t> &bs) const
    {
        const std::size_t num_bytes = bs.num_blocks() * sizeof(std::size_t);
        
        return rapidhashMicro(bs.m_bits.data(), num_bytes);
    }
};

struct BitsetHasherRapidFirstBlock
{
    std::size_t operator()(const boost::dynamic_bitset<std::size_t> &bs) const
    {
        return rapidhashNano(bs.m_bits.data(), 8);
    }
};

namespace bitset_map_namespace
{
    using Bitset = boost::dynamic_bitset<std::size_t>;

    using BitsetMap = emhash8::HashMap<Bitset, std::size_t, BitsetHasherRapid>;
    using BitsetMap2 = emhash8::HashMap<Bitset, std::size_t, BitsetHasherRapidFirstBlock>;

    class BitsetHashMapWrapper
    {
    public:
        BitsetMap map;
        BitsetMap2 map2;
        bool use_all_blocks;

        BitsetHashMapWrapper(bool use_all_bitset_blocks = true)
        {
            use_all_blocks = use_all_bitset_blocks;
        }

        const auto *get_bitsets() const
        {
            if (use_all_blocks)
            {
                return map.values();
            }
            return map2.values();
        }

        void reserve(const uint64_t num_items)
        {
            if (use_all_blocks)
            {
                map.reserve(num_items);
            }
            else
            {
                map2.reserve(num_items);
            }
        }

        void insert_unique(const Bitset &bs, std::size_t value)
        {
            if (use_all_blocks)
            {
                map.emplace_unique(bs, value);
            }
            else
            {
                map2.emplace_unique(bs, value);
            }
        }

        void emplace(const Bitset &bs, std::size_t value)
        {
            if (use_all_blocks)
            {
                map.emplace(std::make_pair(bs, value));
            }
            else
            {
                map2.emplace(std::make_pair(bs, value));
            }
        }

        std::size_t *get_ptr(const Bitset &bs) const
        {
            if (use_all_blocks)
            {
                return map.try_get_using_bucket_occ(bs);
            }
            return map2.try_get_using_bucket_occ(bs);
        }

        std::size_t *get_ptr2(const Bitset &bs) const
        {
            if (use_all_blocks)
            {
                return map.try_get(bs);
            }
            return map2.try_get(bs);
        }

        std::size_t get(const Bitset &bs) const
        {
            if (use_all_blocks)
            {
                auto it = map.find(bs);
                if (it == map.end())
                    return MAX_SIZE_T;
                return it->second;
            }
            auto it = map2.find(bs);
            if (it == map2.end())
                return MAX_SIZE_T;
            return it->second;
        }

        Bitset get_n_th_bitset(std::size_t n) const
        {
            if (use_all_blocks)
            {
                assert(n < map.size());
                const auto *keys = map.values();
                return keys[n].first;
            }
            assert(n < map2.size());
            const auto *keys = map2.values();
            return keys[n].first;
        }

        void set_bucket_occupancy()
        {
            if (use_all_blocks)
            {
                map.set_bucket_occupancy();
            }
            else
            {
                map2.set_bucket_occupancy();
            }
        }

        std::size_t size() const
        {
            if (use_all_blocks)
            {
                return map.size();
            }
            return map2.size();
        }
    };
}