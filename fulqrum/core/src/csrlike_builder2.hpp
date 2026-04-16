/**
 * This code is part of Fulqrum.
 *
 * (C) Copyright IBM 2024.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */
#pragma once
#include "external/hash_table8.hpp"
#include <array>
#include <complex>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "csr_utils.hpp"
#include "csrlike.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include "operators.hpp"
#include <boost/dynamic_bitset.hpp>

// ---------------------------------------------------------------------------
// Helpers for the XOR-based connection scan
// ---------------------------------------------------------------------------

// XOR two half-width bitsets and emit differing bit positions
// Adds an offset=half-width to beta to convert it to full bitset indices.
// Returns the popcount of differing bits; returns 5 early if count > 4.
// Positions are emitted in ascending order (natural bit-scan order).
inline int half_xor_positions(const boost::dynamic_bitset<std::size_t>& a,
                              const boost::dynamic_bitset<std::size_t>& b,
                              const uint16_t offset,
                              uint16_t* positions)
{
    int count = 0;
    const std::size_t n_blocks = a.num_blocks();
    for(std::size_t blk = 0; blk < n_blocks; blk++)
    {
        std::size_t xr = a.m_bits[blk] ^ b.m_bits[blk];
        while(xr)
        {
            if(count >= 4)
                return 5;
            positions[count++] = offset + static_cast<uint16_t>(blk * 64 + __builtin_ctzll(xr));
            xr &= xr - 1;
        }
    }
    return count;
}

// Pack two sorted uint16_t positions into a uint32_t key.
inline uint32_t pack2(const uint16_t* p)
{
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 16);
}

// Pack four sorted uint16_t positions into a uint64_t key.
inline uint64_t pack4(const uint16_t* p)
{
    return static_cast<uint64_t>(p[0]) | (static_cast<uint64_t>(p[1]) << 16) |
           (static_cast<uint64_t>(p[2]) << 32) | (static_cast<uint64_t>(p[3]) << 48);
}

inline int jw_parity(const boost::dynamic_bitset<std::size_t>& bs, uint16_t lo, uint16_t hi)
{
    if(hi <= lo + 1)
        return 1;
    const uint16_t lo1 = lo + 1;
    const uint16_t hi1 = hi - 1;
    const std::size_t blk_lo = lo1 / 64;
    const std::size_t blk_hi = hi1 / 64;
    int cnt = 0;
    for(std::size_t b = blk_lo; b <= blk_hi; b++)
    {
        std::size_t word = bs.m_bits[b];
        if(b == blk_lo)
            word &= ~((std::size_t(1) << (lo1 % 64)) - 1);
        if(b == blk_hi && (hi1 % 64) < 63)
            word &= (std::size_t(2) << (hi1 % 64)) - 1;
        cnt += __builtin_popcountll(word);
    }
    return (cnt & 1) ? -1 : 1;
}

// ---------------------------------------------------------------------------
// Original: without half-det parameters.
// Works for full-str mode of Fulqrum.
// ---------------------------------------------------------------------------
template <typename T, typename U>
void csrlike_builder2(const std::vector<OperatorTerm_t>& terms,
                      const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                      const T* __restrict diag_vec,
                      const unsigned int width,
                      const std::size_t subspace_dim,
                      const int has_nonzero_diag,
                      const std::size_t* __restrict group_ptrs,
                      const std::size_t* __restrict group_ladder_ptrs,
                      const unsigned int* __restrict group_rowint_length,
                      const std::vector<std::vector<unsigned int>>& group_offdiag_inds,
                      const std::size_t num_groups,
                      const unsigned int ladder_offset,
                      std::vector<std::vector<U>>& cols,
                      std::vector<std::vector<T>>& data)
{
    std::size_t kk;
    const unsigned int half_width = width >> 1;
    const auto* bitsets = subspace.get_bitsets();
    cols.resize(subspace_dim);
    data.resize(subspace_dim);

    std::vector<std::mutex> mutex1(subspace_dim);

    std::vector<uint16_t> group_rowint_length_u16(num_groups);
    for(std::size_t i = 0; i < num_groups; i++)
        group_rowint_length_u16[i] = static_cast<uint16_t>(group_rowint_length[i]);

    std::vector<std::array<uint16_t, 5>> group_offdiag_inds_array(num_groups);
    for(std::size_t i = 0; i < num_groups; i++)
    {
        group_offdiag_inds_array[i][4] = static_cast<uint16_t>(group_offdiag_inds[i].size());
        for(std::size_t j = 0; j < group_offdiag_inds[i].size(); j++)
            group_offdiag_inds_array[i][j] = static_cast<uint16_t>(group_offdiag_inds[i][j]);
        for(std::size_t j = group_offdiag_inds[i].size(); j < 4; j++)
            group_offdiag_inds_array[i][j] = 0;
    }

    std::vector<uint16_t> grp_max_inds(num_groups, width);
    get_group_max_inds(grp_max_inds, group_offdiag_inds, num_groups);

    std::vector<std::vector<uint32_t>> groups_by_maxbit(width);
    for(std::size_t g = 0; g < num_groups; g++)
        groups_by_maxbit[grp_max_inds[g]].push_back(static_cast<uint32_t>(g));
    for(std::size_t b = 0; b < width; b++)
        std::sort(groups_by_maxbit[b].begin(), groups_by_maxbit[b].end());

    if(has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
        for(kk = 0; kk < subspace_dim; kk++)
        {
            if(diag_vec[kk] != 0.0)
            {
                cols[kk].push_back(kk);
                data[kk].push_back(diag_vec[kk]);
            }
        }
    }

#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    {
        const boost::dynamic_bitset<std::size_t>& row = bitsets[kk].first;

        std::size_t idx;
        std::size_t group_int_start, group_int_stop;
        const OperatorTerm_t* term;
        boost::dynamic_bitset<std::size_t> col_vec;
        std::size_t* col_ptr;
        std::size_t col_idx;
        T val;
        unsigned int row_int = 0;

        std::vector<uint8_t> row_set_bits(row.size(), 0);
        bitset_to_bitvec(row, row_set_bits);

        for(std::size_t b = 0; b < static_cast<std::size_t>(width); b++)
        {
            if(!row_set_bits[b])
                continue;
            for(const uint32_t group : groups_by_maxbit[b])
            {
                const auto& group_inds = group_offdiag_inds_array[group];
                const uint8_t group_size = static_cast<uint8_t>(group_inds[4]);

                bool passes_filter = true;
                if(group_size == 4)
                {
                    const uint16_t pos0 = group_inds[0];
                    const uint16_t pos1 = group_inds[1];
                    const uint16_t pos2 = group_inds[2];
                    const uint8_t _p = row_set_bits[pos0];
                    const uint8_t _q = row_set_bits[pos1];
                    const uint8_t _r = row_set_bits[pos2];
                    const bool aabb_group = (pos1 < half_width) && (pos2 >= half_width);
                    passes_filter = !(aabb_group && (_r || (_p == _q))) && ((_p + _q + _r) == 1);
                }
                else
                {
                    passes_filter = !row_set_bits[group_inds[0]];
                }
                if(!passes_filter)
                    continue;

                col_vec = row;
                flip_bits_u16_2(col_vec, group_inds, group_size);

                col_ptr = subspace.get_ptr(col_vec);
                if(col_ptr == nullptr)
                    continue;
                col_idx = *col_ptr;

                row_int = bitset_ladder_int_u16_2(
                    row_set_bits.data(), group_inds, group_rowint_length_u16[group]);
                group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
                group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];

                val = 0;
                for(idx = group_int_start; idx < group_int_stop; idx++)
                {
                    term = &terms[idx];
                    if(passes_proj_validation(term, row))
                        accum_element(row,
                                      col_vec,
                                      term->indices,
                                      term->values,
                                      term->coeff,
                                      term->real_phase,
                                      term->indices.size(),
                                      val);
                }

                if(std::abs(val) > ATOL)
                {
                    {
                        std::lock_guard<std::mutex> lock_kk(mutex1[kk]);
                        cols[kk].push_back(col_idx);
                        data[kk].push_back(val);
                    }
                    {
                        std::lock_guard<std::mutex> lock_col_idx(mutex1[col_idx]);
                        cols[col_idx].push_back(kk);
                        if constexpr(std::is_same_v<T, double>)
                            data[col_idx].push_back(val);
                        else
                            data[col_idx].push_back(std::conj(val));
                    }
                }
            }
        }
    }

    sort_paired(cols, data);
}

// ----------------------------------------------------------------------------
// Works for half-strs mode of Fulqrum
//
// In Fulqrum's generic case, we loop over num_groups for each row.
// But, chemistry problems can have large num_groups yet small valid groups.
// Groups that preserve electron numbers are only valid.
// Also, when the Subspace is a tensor/Cartesian product of alpha and beta
// half strs, we can further exploit symmetry.
// This overload of csrlike_builder2 follows the logic below:
// - For the alpha half of a row bitset (row_alpha), it compares each alpha
//      det to find differing bit positions using XOR op.
//      Only length-2 and length-4 differing bit positions are valid for
//      chemistry problems (Slater-Condone rule). We denote length-2 differing
//      bit positions as aa and length-4 as aaaa (each a = alpha orbital index)
// - We repeat the above for beta half of a row bitset and all_beta_dets.
//      It gives us bb and bbbb.
// - We use the differing bit positions to construct following
//      aa = single excitation
//      bb = single excitation
//      aaaa = double excitation (within only alpha orbitals)
//      bbbb = double excitation (within only beta orbitals)
//      aa x bb = double excitation (cross spin; cross product of aa and bb)
// - Now, above differing bit positions are actually our grp offdiag inds from
//      generic (original) case. The difference is instead of num_groups
//      for the whole moleculer operator, we only have valid flips for a row.
//      Num valids < num groups typically; significantly cutting down time.
// - However, to get start and end of contributing terms, we need the
//      group number of a corresponding grp offdiag inds array.
// - Thus, we need to construct a look-up table (hashmap) to convert
//      inds to group number (see inds_to_group2 and inds_to_group4).
// ----------------------------------------------------------------------------
// T is the data type, U is the index type, e.g (complex, int)
template <typename T, typename U>
void csrlike_builder2(const std::vector<OperatorTerm_t>& terms,
                      const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                      const T* __restrict diag_vec,
                      const unsigned int width,
                      const std::size_t subspace_dim,
                      const int has_nonzero_diag,
                      const std::size_t* __restrict group_ptrs,
                      const std::size_t* __restrict group_ladder_ptrs,
                      const unsigned int* __restrict group_rowint_length,
                      const std::vector<std::vector<unsigned int>>& group_offdiag_inds,
                      const std::size_t num_groups,
                      const unsigned int ladder_offset,
                      const std::vector<boost::dynamic_bitset<std::size_t>>& all_alpha_dets,
                      const std::vector<boost::dynamic_bitset<std::size_t>>& all_beta_dets,
                      std::vector<std::vector<U>>& cols,
                      std::vector<std::vector<T>>& data)
{
    std::size_t kk;
    const unsigned int half_width = width >> 1;
    const auto* bitsets = subspace.get_bitsets();
    cols.resize(subspace_dim);
    data.resize(subspace_dim);

    // -----------------------------------------------------------------------
    // Convert group_rowint_length to uint16
    // -----------------------------------------------------------------------
    std::vector<uint16_t> group_rowint_length_u16(num_groups);
    for(std::size_t i = 0; i < num_groups; i++)
        group_rowint_length_u16[i] = static_cast<uint16_t>(group_rowint_length[i]);

    // -----------------------------------------------------------------------
    // Convert group_offdiag_inds to array-based structure; trim to uint16.
    // Index [4] holds the group size; unused position slots are 0.
    // -----------------------------------------------------------------------
    std::vector<std::array<uint16_t, 5>> group_offdiag_inds_array(num_groups);
    for(std::size_t i = 0; i < num_groups; i++)
    {
        group_offdiag_inds_array[i][4] = static_cast<uint16_t>(group_offdiag_inds[i].size());
        for(std::size_t j = 0; j < group_offdiag_inds[i].size(); j++)
            group_offdiag_inds_array[i][j] = static_cast<uint16_t>(group_offdiag_inds[i][j]);
        for(std::size_t j = group_offdiag_inds[i].size(); j < 4; j++)
            group_offdiag_inds_array[i][j] = 0;
    }

    // -----------------------------------------------------------------------
    // Build packed flip-positions -> group index lookup maps (built once).
    //
    // group_offdiag_inds positions are sorted ascending (guaranteed by the
    // grouping code). XOR bit-scan also produces positions in ascending order.
    // Therefore no sorting is needed when forming lookup keys.
    //
    //   size-2 group key: pack2 -> uint32  (p0 | p1<<16)
    //   size-4 group key: pack4 -> uint64  (p0 | p1<<16 | p2<<32 | p3<<48)
    // -----------------------------------------------------------------------
    struct HashU32
    {
        std::size_t operator()(uint32_t k) const noexcept
        {
            return rapidhash(&k, sizeof(k));
        }
    };
    struct HashU64
    {
        std::size_t operator()(uint64_t k) const noexcept
        {
            return rapidhash(&k, sizeof(k));
        }
    };
    emhash8::HashMap<uint32_t, uint32_t, HashU32> inds_to_group2;
    emhash8::HashMap<uint64_t, uint32_t, HashU64> inds_to_group4;
    inds_to_group2.reserve(num_groups);
    inds_to_group4.reserve(num_groups);

    // Keys must be packed in ascending position order to match the XOR scan output.
    for(std::size_t g = 0; g < num_groups; g++)
    {
        auto& inds = group_offdiag_inds_array[g];
        const uint8_t sz = static_cast<uint8_t>(inds[4]);

        if(sz == 2)
            inds_to_group2[pack2(inds.data())] = static_cast<uint32_t>(g);
        else if(sz == 4)
            inds_to_group4[pack4(inds.data())] = static_cast<uint32_t>(g);
    }

    // -----------------------------------------------------------------------
    // Precompute aabb coeff table.
    // -----------------------------------------------------------------------
    std::vector<bool> is_aabb_group(num_groups, false);
    std::vector<T> aabb_coeff(num_groups, T(0));
    for(std::size_t g = 0; g < num_groups; g++)
    {
        const auto& ginds = group_offdiag_inds_array[g];
        if(ginds[4] != 4)
            continue;

        if(ginds[1] >= half_width || ginds[2] < half_width)
            continue;
        is_aabb_group[g] = true;

        // extract the group coefficient
        for(unsigned ri = 0; ri + 1 < ladder_offset; ri++)
        {
            const std::size_t t0 = group_ladder_ptrs[g * ladder_offset + ri];
            const std::size_t t1 = group_ladder_ptrs[g * ladder_offset + ri + 1];
            if(t0 < t1)
            {
                const double c = terms[t0].coeff.real() * static_cast<double>(terms[t0].real_phase);
                if constexpr(std::is_same_v<T, double>)
                    aabb_coeff[g] = c;
                else
                    aabb_coeff[g] = T(c, 0.0);
                break;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Diagonal elements
    // -----------------------------------------------------------------------
    if(has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
        for(kk = 0; kk < subspace_dim; kk++)
        {
            if(diag_vec[kk] != 0.0)
            {
                cols[kk].push_back(kk);
                data[kk].push_back(diag_vec[kk]);
            }
        }
    }

    const std::size_t N_alpha = all_alpha_dets.size();
    const std::size_t N_beta = all_beta_dets.size();

    // -----------------------------------------------------------------------
    // Crude estimate of nnz per row
    // -----------------------------------------------------------------------
    std::vector<std::size_t> n_aa(N_alpha, 0), n_aaaa(N_alpha, 0);
    std::vector<std::size_t> n_bb(N_beta,  0), n_bbbb(N_beta,  0);

#pragma omp parallel for schedule(static)
    for(std::size_t ia = 0; ia < 1; ia++)
    {
        uint16_t pos[4];
        std::size_t ns = 0, nd = 0;
        for(std::size_t ja = 0; ja < N_alpha; ja++)
        {
            const int d = half_xor_positions(all_alpha_dets[ia], all_alpha_dets[ja], 0u, pos);
            if(d == 2)      ns++;
            else if(d == 4) nd++;
        }
        n_aa[ia]   = ns;
        n_aaaa[ia] = nd;
    }

#pragma omp parallel for schedule(static)
    for(std::size_t ib = 0; ib < 1; ib++)
    {
        uint16_t pos[4];
        std::size_t ns = 0, nd = 0;
        for(std::size_t jb = 0; jb < N_beta; jb++)
        {
            const int d = half_xor_positions(
                all_beta_dets[ib], all_beta_dets[jb], static_cast<uint16_t>(half_width), pos);
            if(d == 2)      ns++;
            else if(d == 4) nd++;
        }
        n_bb[ib]   = ns;
        n_bbbb[ib] = nd;
    }

    const std::size_t est = n_aa[0] + n_aaaa[0]
                              + n_bb[0] + n_bbbb[0]
                              + n_aa[0] * n_bb[0];
#pragma omp parallel for schedule(static)
    for(std::size_t kk = 0; kk < subspace_dim; kk++)
    {
        // const std::size_t ia = kk % N_alpha;
        // const std::size_t ib = kk / N_alpha;
        // const std::size_t est = n_aa[0] + n_aaaa[0]
        //                       + n_bb[0] + n_bbbb[0]
        //                       + n_aa[0] * n_bb[0];
        
        // n_aa[ia] + n_aaaa[ia]
        //                       + n_bb[ib] + n_bbbb[ib]
        //                       + n_aa[ia] * n_bb[ib];
        cols[kk].reserve(est);
        data[kk].reserve(est);
    }

    // -----------------------------------------------------------------------
    // Main loop:
    //
    // Insertion order in subspace is beta-outer x alpha-inner (ascending full
    // bitset value), so kk = ib*N_alpha + ia.
    //
    // Five excitation types per row (ia, ib):
    //   aa   : alpha single, col_idx = ib*N_a + ja
    //   aaaa : alpha double, col_idx = ib*N_a + ja
    //   bb   : beta  single, col_idx = jb*N_a + ia
    //   bbbb : beta  double, col_idx = jb*N_a + ia
    //   aabb : cross double, col_idx = jb*N_a + ja
    // -----------------------------------------------------------------------
#pragma omp parallel for schedule(static) if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    {
        const boost::dynamic_bitset<std::size_t>& row = bitsets[kk].first;
        thread_local boost::dynamic_bitset<std::size_t> col_vec;
        col_vec = row;

        auto process_group =
            [&](uint32_t g, std::size_t col_idx, const uint16_t* inds, uint8_t gsz) {
                flip_bits_u16_3(col_vec, inds, gsz);

                const uint8_t rowint_len = group_rowint_length_u16[g];
                const unsigned int row_int = bitset_ladder_int_direct(row, inds, rowint_len);
                const std::size_t g_start = group_ladder_ptrs[g * ladder_offset + row_int];
                const std::size_t g_stop = group_ladder_ptrs[g * ladder_offset + row_int + 1];

                T val = 0;
                for(std::size_t idx = g_start; idx < g_stop; idx++)
                {
                    const auto* term = &terms[idx];
                    if(passes_proj_validation(term, row))
                        accum_element(row,
                                      col_vec,
                                      term->indices,
                                      term->values,
                                      term->coeff,
                                      term->real_phase,
                                      term->indices.size(),
                                      val);
                }

                flip_bits_u16_3(col_vec, inds, gsz);

                if(std::abs(val) > ATOL)
                {
                    cols[kk].push_back(col_idx);
                    data[kk].push_back(val);
                }
            };

        uint16_t pos[4];
        const std::size_t ib = kk / N_alpha;
        const std::size_t ia = kk % N_alpha;
        const boost::dynamic_bitset<std::size_t>& row_alpha = all_alpha_dets[ia];
        const boost::dynamic_bitset<std::size_t>& row_beta = all_beta_dets[ib];
        thread_local std::vector<std::pair<std::size_t, std::array<uint16_t, 2>>> a_singles, b_singles;
        a_singles.clear();
        b_singles.clear();

        // Alpha XOR: aa and aaaa (all ja; each thread owns cols[kk])
        for(std::size_t ja = 0; ja < N_alpha; ja++)
        {
            const int d = half_xor_positions(row_alpha, all_alpha_dets[ja], 0u, pos);
            if(d == 2)
            {
                const auto it = inds_to_group2.find(pack2(pos));
                if(it != inds_to_group2.end())
                    process_group(it->second, ib * N_alpha + ja, pos, 2u);
                a_singles.push_back({ja, {pos[0], pos[1]}});
            }
            else if(d == 4)
            {
                const auto it = inds_to_group4.find(pack4(pos));
                if(it != inds_to_group4.end())
                    process_group(it->second, ib * N_alpha + ja, pos, 4u);
            }
        }

        // Beta XOR: bb and bbbb (all jb; each thread owns cols[kk])
        for(std::size_t jb = 0; jb < N_beta; jb++)
        {
            const int d = half_xor_positions(
                row_beta, all_beta_dets[jb], static_cast<uint16_t>(half_width), pos);
            if(d == 2)
            {
                const auto it = inds_to_group2.find(pack2(pos));
                if(it != inds_to_group2.end())
                    process_group(it->second, jb * N_alpha + ia, pos, 2u);
                b_singles.push_back({jb, {pos[0], pos[1]}});
            }
            else if(d == 4)
            {
                const auto it = inds_to_group4.find(pack4(pos));
                if(it != inds_to_group4.end())
                {
                    process_group(it->second, jb * N_alpha + ia, pos, 4u);
                }
            }
        }

        // aabb: all jb != ib, all ja; each thread owns cols[kk]
        for(const auto& [jb_idx, bp] : b_singles)
        {
            if(jb_idx == ib)
            {
                continue;
            }

            const int bsign = jw_parity(row_beta,
                                        static_cast<uint16_t>(bp[0] - half_width),
                                        static_cast<uint16_t>(bp[1] - half_width));
            for(const auto& [ja_idx, ap] : a_singles)
            {
                const uint16_t cross[4] = {ap[0], ap[1], bp[0], bp[1]};
                const auto it = inds_to_group4.find(pack4(cross));
                if(it == inds_to_group4.end())
                {
                    continue;
                }

                const uint32_t g = it->second;
                // if(is_aabb_group[g])
                {
                    // Direct formula: val = coeff(g) * asign * bsign
                    const int asign = jw_parity(row_alpha, ap[0], ap[1]);
                    const T val = aabb_coeff[g] * T(asign * bsign);
                    if(std::abs(val) > ATOL)
                    {
                        const std::size_t col_idx = jb_idx * N_alpha + ja_idx;
                        cols[kk].push_back(col_idx);
                        data[kk].push_back(val);
                    }
                }
                // else
                // {
                //     process_group(g, jb_idx * N_alpha + ja_idx, cross, 4u);
                // }
            }
        }

    } // end parallel for over rows

    sort_paired(cols, data);
} // end function
