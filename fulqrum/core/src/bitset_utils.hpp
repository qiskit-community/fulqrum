#ifndef BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#define BOOST_DYNAMIC_BITSET_DONT_USE_FRIENDS
#endif
#pragma once
#include <vector>
#include <algorithm>
#include "base.hpp"
#include "constants.hpp"
#include "bitset_hashmap.hpp"
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
    unsigned int row_int, out_int = 0;
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
inline bool passes_proj_validation(const OperatorTerm_t *__restrict term,
                                   const boost::dynamic_bitset<std::size_t> &bitset)
{
    std::size_t kk;
    unsigned int block_num, block_idx;
    unsigned int pos;
    unsigned int bit;
    bool out = 1;
    for (kk = 0; kk < term->proj_indices.size(); kk++)
    {
        pos = term->proj_indices[kk];
        block_num = pos >> BLOCK_EXPONENT;
        block_idx = pos & BLOCK_SHIFT;
        bit = ((bitset.m_bits[block_num] >> block_idx) & std::size_t(1));
        if (bit != term->proj_bits[kk])
        {
            out = 0;
            break;
        }
    }
    return out;
}

/**
 * Computes orbital occupancy information of spin orbitals.
 *
 * @param subspace Subsapce as the HashMap.
 * @param subspace_dim The number of bitsets in the subspace.
 * @param probabilities Absolute squared eigenvector representing
 * probability of each basis vector (subspace bitset).
 * @param out Orbital occupancies of spin orbitals.
 */
void compute_orbital_occupancies(
    const bitset_map_namespace::BitsetHashMapWrapper &subspace,
    const std::size_t subspace_dim,
    const double *__restrict probabilities,
    double *out)
{
    const auto *bitsets = subspace.get_bitsets();
    std::size_t kk;
    for (kk = 0; kk < subspace_dim; kk++)
    {
        const boost::dynamic_bitset<size_t> &row = bitsets[kk].first;
        std::vector<std::size_t> set_indices;
        for (size_t block = 0; block < row.num_blocks(); block++)
        {
            auto bitset = row.m_bits[block];
            while (bitset != 0)
            {
                uint64_t t = bitset & -bitset;
                int r = __builtin_ctzll(bitset);
                set_indices.push_back(block * BITS_PER_BLOCK + r);
                bitset ^= t;
            }
        }
        for (std::size_t &idx : set_indices)
        {
            out[idx] += probabilities[kk];
        }
    }
}

/**
 * Constructs a vector of max offdiagonal group indices. The vector allows
 * us to check only onw bit position in the row bitset to determine whether
 * the candidate column bitset will be smaller or greater than the row bitset.
 *
 * @param grp_max_inds Vector to hold max index of each group. As the index
 * is represented by ``uint16_t``, we can solve max 2^16 = 65536 qubit problem
 * using Fulqrum.
 * @param group_offdiag_inds Offdiagonal indices of a group. This list determines
 * which bit positions in a row bitset will be flipped to construct a candidate
 * column bitset.
 * @param num_groups The number of groups.
 */
void get_group_max_inds(std::vector<uint16_t> &grp_max_inds,
                        const std::vector<std::vector<unsigned int>> &group_offdiag_inds,
                        const std::size_t &num_groups)
{
#pragma omp parallel for schedule(dynamic) if (num_groups > 4096)
    for (size_t group = 0; group < num_groups; group++)
    {
        auto group_inds = &group_offdiag_inds[group];

        // It is likely that the ``group_inds`` is already sorted, and
        // the last element is the maximum. However, as size of ``group_inds``
        // is moderate (2 or 4) and computing max element is cheap, we make use
        // of ``std::max_element()`` to be safe.
        auto max_element = std::max_element((*group_inds).begin(), (*group_inds).end());
        grp_max_inds[group] = *max_element;
    }
}

/**
 * creates a vector representation of the row bitset
 * with 1 at set-bit positions. This vector is easier to
 * look-up by index as looking up a bit in a bitset required
 * division followed modulo operations.
 * code from:
 * https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
 *
 * @param row Bitstring in boost::dynamic_bitset<> format that will be converted
 * into a vector of only 1s and 0s.
 * @param row_set_bits Vector to hold bits of ``row`` bitset.
 */
void bitset_to_bitvec(const boost::dynamic_bitset<size_t> &row,
                      std::vector<uint8_t> &row_set_bits)
{
    for (size_t block = 0; block < row.num_blocks(); block++)
    {
        auto bitset = row.m_bits[block];
        while (bitset != 0)
        {
            uint64_t t = bitset & -bitset;
            int r = __builtin_ctzll(bitset);
            row_set_bits[block * BITS_PER_BLOCK + r] = 1;
            bitset ^= t;
        }
    }
}
