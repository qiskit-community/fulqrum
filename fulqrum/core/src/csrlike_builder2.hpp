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
#include <array>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

// #include <gperftools/profiler.h>

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

// XOR the lower half_width bits of full_row against alpha_det (size == half_width).
// Fills positions[] with the bit indices (in full-bitset indices) of differing bits.
// Returns the popcount; stops early and returns 5 if count > 4.
// Positions are always emitted in ascending order (natural bit-scan order).
inline int alpha_xor_positions(const boost::dynamic_bitset<std::size_t>& full_row,
                               const boost::dynamic_bitset<std::size_t>& alpha_det,
                               const unsigned int half_width,
                               uint16_t* positions)
{
    int count = 0;
    const std::size_t n_blocks = alpha_det.num_blocks();
    const std::size_t rem = half_width % 64;
    for(std::size_t b = 0; b < n_blocks; b++)
    {
        std::size_t xr = full_row.m_bits[b] ^ alpha_det.m_bits[b];
        // mask the last block to only the valid half_width bits
        if(b == n_blocks - 1 && rem != 0)
            xr &= (std::size_t(1) << rem) - 1;
        while(xr)
        {
            if(count >= 4)
                return 5;
            positions[count++] = static_cast<uint16_t>(b * 64 + __builtin_ctzll(xr));
            xr &= xr - 1; // clear lowest set bit
        }
    }
    return count;
}

// XOR the upper half_width bits of full_row (starting at bit half_width) against
// beta_det (size == half_width).
// Fills positions[] with indices in FULL-bitset indices (i.e. half_width + bit_in_half).
// Returns the popcount; stops early and returns 5 if count > 4.
// Positions are always emitted in ascending order.
inline int beta_xor_positions(const boost::dynamic_bitset<std::size_t>& full_row,
                              const boost::dynamic_bitset<std::size_t>& beta_det,
                              const unsigned int half_width,
                              uint16_t* positions)
{
    int count = 0;
    const std::size_t n_blocks = beta_det.num_blocks();
    const std::size_t shift = half_width % 64; // bit offset within the starting block
    const std::size_t start_blk = half_width / 64; // first full-bitset block containing beta bits
    const std::size_t rem = half_width % 64;

    for(std::size_t b = 0; b < n_blocks; b++)
    {
        std::size_t full_block;
        if(shift == 0)
        {
            full_block = full_row.m_bits[start_blk + b];
        }
        else
        {
            full_block = full_row.m_bits[start_blk + b] >> shift;
            if(start_blk + b + 1 < full_row.num_blocks())
                full_block |= full_row.m_bits[start_blk + b + 1] << (64 - shift);
        }

        std::size_t xr = full_block ^ beta_det.m_bits[b];
        // mask the last block to only the valid half_width bits
        if(b == n_blocks - 1 && rem != 0)
            xr &= (std::size_t(1) << rem) - 1;

        while(xr)
        {
            if(count >= 4)
                return 5;
            // position in full bitset = half_width + (b * 64 + bit)
            positions[count++] = static_cast<uint16_t>(half_width + b * 64 + __builtin_ctzll(xr));
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

// ---------------------------------------------------------------------------
// Original: without half-det parameters.
// Works for full-str mode of Fulqrum.
// ---------------------------------------------------------------------------
template <typename T, typename U>
void csrlike_builder2(const OperatorTerm_t* terms,
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
                                      &term->indices[0],
                                      &term->values[0],
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

    std::cout << "Num groups: " << num_groups << std::endl;
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
// bit positions as aa and length-4 as aaaa (each a = alpha orbital index)
// - We repeat the above for beta half of a row bitset and all_beta_dets.
//      It gives us bb and bbbb.
// - We use the differing bit positions to construct following
//      aa = single excitation
//      bb = single excitation
//      aaaa = double excitation (within only alpha orbitals)
//      bbbb = double excitation (within only beta orbitals)
//      aa x bb = double excitation (cross spin; cross product aa and bb)
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
void csrlike_builder2(const OperatorTerm_t* terms,
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

    std::vector<std::mutex> mutex1(subspace_dim);

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
    std::unordered_map<uint32_t, uint32_t> inds_to_group2;
    std::unordered_map<uint64_t, uint32_t> inds_to_group4;
    inds_to_group2.reserve(num_groups);
    inds_to_group4.reserve(num_groups);

    for(std::size_t g = 0; g < num_groups; g++)
    {
        const auto& inds = group_offdiag_inds_array[g];
        const uint8_t sz = static_cast<uint8_t>(inds[4]);
        if(sz == 2)
            inds_to_group2[pack2(inds.data())] = static_cast<uint32_t>(g);
        else if(sz == 4)
            inds_to_group4[pack4(inds.data())] = static_cast<uint32_t>(g);
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

    // ------------------------------------------------------------------------------
    // Main loop: XOR-based valid group filter
    //
    // For each row (ia, ib) we find connected columns via five excitation types:
    //   aa   : alpha single -> size-2 group, both p < half_width
    //   aaaa : alpha double -> size-4 group, all  p < half_width
    //   bb   : beta  single -> size-2 group, both p >= half_width
    //   bbbb : beta  double -> size-4 group, all  p >= half_width
    //   aabb : cross double  (alpha single x beta single)  -> size-4 group, mixed p
    //
    // Cross-double positions must be sorted.
    //
    // For a strict Cartesian-product subspace subspace hashmap subsapce look up
    // can be replaced by direct index arithmetic (col_idx = ja * N_beta + jb)
    // -----------------------------------------------------------------------------
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    {
        const boost::dynamic_bitset<std::size_t>& row = bitsets[kk].first;

        // consider getting rid of row_set_bits
        std::vector<uint8_t> row_set_bits(width, 0);
        bitset_to_bitvec(row, row_set_bits);

        // col_vec is allocated once per row and reused across all process_group calls.
        // Flipped twice to restore row for a different group
        boost::dynamic_bitset<std::size_t> col_vec = row;

        // inds (uint16_t*) and gsz are passed from the call site (already known from the
        // XOR scan), avoiding the group_offdiag_inds_array[g] lookup
        // _3 variants of flip and bitset_ladder_int accept uint16_t*.
        auto process_group =
            [&](uint32_t g, std::size_t col_idx, const uint16_t* inds, uint8_t gsz) {
                flip_bits_u16_3(col_vec, inds, gsz); // flip row to get col

                const unsigned int row_int =
                    bitset_ladder_int_u16_3(row_set_bits.data(), inds, group_rowint_length_u16[g]);
                const std::size_t g_start = group_ladder_ptrs[g * ladder_offset + row_int];
                const std::size_t g_stop = group_ladder_ptrs[g * ladder_offset + row_int + 1];

                T val = 0;
                for(std::size_t idx = g_start; idx < g_stop; idx++)
                {
                    const auto* term = &terms[idx];
                    if(passes_proj_validation(term, row))
                        accum_element(row,
                                      col_vec,
                                      &term->indices[0],
                                      &term->values[0],
                                      term->coeff,
                                      term->real_phase,
                                      term->indices.size(),
                                      val);
                }

                flip_bits_u16_3(col_vec, inds, gsz); // flip col to restore row

                if(std::abs(val) > ATOL)
                {
                    {
                        std::lock_guard<std::mutex> lock(mutex1[kk]);
                        cols[kk].push_back(col_idx);
                        data[kk].push_back(val);
                    }

                    {
                        std::lock_guard<std::mutex> lock(mutex1[col_idx]);
                        cols[col_idx].push_back(kk);
                        if constexpr(std::is_same_v<T, double>)
                            data[col_idx].push_back(val);
                        else
                            data[col_idx].push_back(std::conj(val));
                    }
                }
            };

        // -------------------------------------------------------------------------
        // ia/ib are the Cartesian-product indices of this row: kk = ia*N_beta + ib.
        // They let us check the lower triangle (col_idx <= kk).
        // -------------------------------------------------------------------------
        uint16_t pos[4];
        const std::size_t ia = kk / N_beta;
        const std::size_t ib = kk % N_beta;
        // a_singles/b_singles store (det_index, flip_positions) for cross doubles.
        std::vector<std::pair<std::size_t, std::array<uint16_t, 2>>> a_singles, b_singles;

        // ------------------------------------------------------------------
        // Alpha XOR: aa (single) and aaaa (double)
        // col_idx = ja * N_beta + ib  =>  col_idx > kk  only if  ja > ia
        // ------------------------------------------------------------------
        for(std::size_t ja = 0; ja < N_alpha; ja++)
        {
            const int d = alpha_xor_positions(row, all_alpha_dets[ja], half_width, pos);
            if(d == 2)
            {
                if(ja > ia)
                {
                    const auto it = inds_to_group2.find(pack2(pos));
                    if(it != inds_to_group2.end())
                        process_group(it->second, ja * N_beta + ib, pos, 2u);
                }
                a_singles.push_back({ja, {pos[0], pos[1]}});
            }
            else if(d == 4)
            {
                if(ja > ia)
                {
                    const auto it = inds_to_group4.find(pack4(pos));
                    if(it != inds_to_group4.end())
                        process_group(it->second, ja * N_beta + ib, pos, 4u);
                }
            }
        }

        // ------------------------------------------------------------------
        // Beta XOR: bb (single) and bbbb (double)
        // col_idx = ia * N_beta + jb  =>  col_idx > kk  only if  jb > ib
        // ------------------------------------------------------------------
        for(std::size_t jb = 0; jb < N_beta; jb++)
        {
            const int d = beta_xor_positions(row, all_beta_dets[jb], half_width, pos);
            if(d == 2)
            {
                if(jb > ib)
                {
                    const auto it = inds_to_group2.find(pack2(pos));
                    if(it != inds_to_group2.end())
                        process_group(it->second, ia * N_beta + jb, pos, 2u);
                }
                b_singles.push_back({jb, {pos[0], pos[1]}});
            }
            else if(d == 4)
            {
                if(jb > ib)
                {
                    const auto it = inds_to_group4.find(pack4(pos));
                    if(it != inds_to_group4.end())
                        process_group(it->second, ia * N_beta + jb, pos, 4u);
                }
            }
        }

        // ------------------------------------------------------------------
        // aa x bb cross doubles: Cartesian product of alpha and beta singles.
        // {a[0], a[1], b[0], b[1]} is always sorted ascending because
        // a < half_width <= b, so no extra sorting is needed.
        // col_idx = ja * N_beta + jb  =>  col_idx > kk  checked directly.
        // ------------------------------------------------------------------
        for(const auto& [ja_idx, ap] : a_singles)
        {
            for(const auto& [jb_idx, bp] : b_singles)
            {
                const std::size_t col_idx = ja_idx * N_beta + jb_idx;
                if(col_idx <= kk)
                    continue;
                const uint16_t cross[4] = {ap[0], ap[1], bp[0], bp[1]};
                const auto it = inds_to_group4.find(pack4(cross));
                if(it != inds_to_group4.end())
                    process_group(it->second, col_idx, cross, 4u);
            }
        }

    } // end parallel for over rows

    std::cout << "Num groups: " << num_groups << std::endl;
    sort_paired(cols, data);
} // end function
