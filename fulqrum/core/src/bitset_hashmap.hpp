#pragma once

#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif

#include <cassert>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>
#include "constants.hpp"
#include "./external/hash_table8.hpp"
#include "./external/rapidhash.h"


struct BitsetHasher {
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const {
        return boost::hash_value(bs);
    }
};

struct BitsetHasherRapid {
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const {
        const std::size_t num_bytes = bs.num_blocks() * sizeof(std::size_t);

        return rapidhash(bs.m_bits.data(), num_bytes);
    }
};

namespace bitset_map_namespace {

    using Bitset = boost::dynamic_bitset<std::size_t>;
    using BitsetMap = emhash8::HashMap<Bitset, std::size_t, BitsetHasherRapid>;

    class BitsetHashMapWrapper {
    public:
        BitsetMap map;

        const BitsetMap& get_map() const {
            return map;
        }

        void insert_unique(const Bitset& bs, std::size_t value) {
            map.emplace_unique(bs, value);
        }

        std::size_t* get_ptr(const Bitset& bs) const {
            return map.try_get(bs);
        }

        std::size_t get(const Bitset& bs) const {
            auto it = map.find(bs);
            if (it == map.end()) return MAX_SIZE_T;
            return it->second;
        }

        Bitset get_n_th_bitset(std::size_t n) const {
            assert(n < map.size());
            const auto* keys = map.values();
            return keys[n].first;
        }

        std::size_t size() const {
            return map.size();
        }
    };
}