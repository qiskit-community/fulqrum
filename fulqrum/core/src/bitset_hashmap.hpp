#pragma once

#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif

#include <cassert>
#include <variant>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>
#include "constants.hpp"
#include "./external/hash_table8.hpp"
// #define RAPIDHASH_PROTECTED
#include "./external/rapidhash.h"
#include "./external/a5hash.h"
// #define XXH_INLINE_ALL
#include "./external/xxhash.h"


struct BitsetHasher {
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const {
        return boost::hash_value(bs);
    }
};

struct BitsetHasherBinWidth {
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const {
        return bs.m_bits[0];
        // return bs.m_bits[0] & ((std::size_t(1) << 26) - 1);
    }
};

struct BitsetHasherXXHash {
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const {
        const std::size_t num_bytes = bs.num_blocks() * sizeof(std::size_t);

        return XXH3_64bits(bs.m_bits.data(), num_bytes);
    }
};

struct BitsetHasherRapid {
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const {
        const std::size_t num_bytes = bs.num_blocks() * sizeof(std::size_t);

        return rapidhashMicro(bs.m_bits.data(), num_bytes);
        
        // murmur hash mixing step
        // std::size_t key = rapidhashMicro(bs.m_bits.data(), num_bytes);
        // key ^= (key >> 33);
        // key *= 0xff51afd7ed558ccd;
        // key ^= (key >> 33);
        // key *= 0xc4ceb9fe1a85ec53;
        // key ^= (key >> 33);
        // return  key;
    }
};

struct BitsetHasherSimple {
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const {

        __uint128_t prod = bs.m_bits[0] * bs.m_bits[0];
        uint64_t low = (uint64_t)prod;
        uint64_t high = (uint64_t)(prod >> 64);

        return low ^ high;
    }
};

struct BitsetHasherRapidOneBlock {
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const {
        return rapidhashNano(bs.m_bits.data(), 8);

        // murmurhash mixing step
        // std::size_t key = rapidhashNano(bs.m_bits.data(), 8);
        // key ^= (key >> 33);
        // key *= 0xff51afd7ed558ccd;
        // key ^= (key >> 33);
        // key *= 0xc4ceb9fe1a85ec53;
        // key ^= (key >> 33);
        // return  key;
    }
};

struct BitsetHasherA5 {
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& bs) const {
        const std::size_t num_bytes = bs.num_blocks() * sizeof(std::size_t);

        return a5hash(bs.m_bits.data(), num_bytes, 0);
    }
};

namespace bitset_map_namespace {

    using Bitset = boost::dynamic_bitset<std::size_t>;
    // using BitsetMap = emhash8::HashMap<Bitset, std::size_t, BitsetHasher>;
    
    using BitsetMap = emhash8::HashMap<Bitset, std::size_t, BitsetHasherRapid>;
    using BitsetMap2 = emhash8::HashMap<Bitset, std::size_t, BitsetHasherRapidOneBlock>;
    
    // using BitsetMap = emhash8::HashMap<Bitset, std::size_t, BitsetHasherA5>;
    // using BitsetMap2 = emhash8::HashMap<Bitset, std::size_t, BitsetHasherBinWidth>;
    // using BitsetMap = emhash8::HashMap<Bitset, std::size_t, BitsetHasherXXHash>;

    class BitsetHashMapWrapper {
    public:
        BitsetMap map;
        BitsetMap2 map2;
        bool use_full;

        BitsetHashMapWrapper(bool full_block=true) {
            use_full = full_block;
        }

        const auto* get_bitsets() const {
            if (use_full) {return map.values();}
            return map2.values();
        }

        void reserve(const uint32_t num_items) {
            if (use_full) {
                map.reserve(num_items);
            }
            else
            {
                map2.reserve(num_items);
            }
        }

        void insert_unique(const Bitset& bs, std::size_t value) {
            if (use_full)
            {
                map.emplace_unique(bs, value);
            }
            else
            {
                map2.emplace_unique(bs, value);
            }
        }

        std::size_t* get_ptr(const Bitset& bs) const {
            if (use_full)
            {
                return map.try_get(bs);
            }
            return map2.try_get(bs);
        }

        std::size_t get(const Bitset& bs) const {
            if (use_full)
            {
                auto it = map.find(bs);
                if (it == map.end()) return MAX_SIZE_T;
                return it->second;
            }
            auto it = map2.find(bs);
            if (it == map2.end()) return MAX_SIZE_T;
            return it->second;
        }

        Bitset get_n_th_bitset(std::size_t n) const {
            if (use_full)
            {
                assert(n < map.size());
                const auto* keys = map.values();
                return keys[n].first;
            }
            assert(n < map2.size());
            const auto* keys = map2.values();
            return keys[n].first;
        }

        std::size_t size() const {
            if(use_full) {return map.size();}
            return map2.size();
        }

        void dump_statistics() const {
            if(use_full) {map.dump_statics();}
            else{map2.dump_statics();}
        }
    };
}