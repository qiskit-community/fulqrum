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
#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "external/hash_set8.hpp"

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "csr_utils.hpp"
#include "csrlike.hpp"
#include "elements.hpp"
#include "matvec2.hpp"
#include "offdiag_grouping.hpp"
#include <boost/dynamic_bitset.hpp>

// T is the data type, U is in the index type, e.g (complex, int)
template <typename T, typename U>
void csrlike_builder2(const std::vector<OperatorTerm_t>& terms,
                      const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                      const T* __restrict diag_vec,
                      const width_t width,
                      const std::size_t subspace_dim,
                      const int has_nonzero_diag,
                      const std::size_t* __restrict group_ptrs,
                      const std::size_t* __restrict group_ladder_ptrs,
                      const width_t* __restrict group_rowint_length,
                      const std::vector<std::vector<width_t>>& group_offdiag_inds,
                      const std::size_t num_groups,
                      const unsigned int ladder_offset,
                      std::vector<std::vector<U>>& cols,
                      std::vector<std::vector<T>>& data)
{
    std::size_t kk;
    const auto* bitsets = subspace.get_bitsets();

    // Flatten group_offdiag_inds (vector<vector> -> contiguous CSR) once.
    // Inner vec in a 2D vector can be scattered across the heap. Contiguos CSR
    // has more favorable memory access pattern. Shows decent speed-up in tests.
    // The one-time flatten is negligible. Output is unchanged.
    std::vector<width_t> _flat_inds;
    std::vector<std::size_t> _inds_offsets;
    flatten_offdiag_inds(group_offdiag_inds, _flat_inds, _inds_offsets);
    const width_t* __restrict flat_inds = _flat_inds.data();
    const std::size_t* __restrict inds_offsets = _inds_offsets.data();
    auto gview = [&](std::size_t g) -> GroupIndsView {
        const std::size_t off = inds_offsets[g];
        return GroupIndsView{flat_inds + off, inds_offsets[g + 1] - off};
    };

    // Make ladder table leaner: size_t -> uint32 once.
    // A lean table is more cache-friendly.
    // Any reasonable operator will have <= UINT32_MAX terms.
    // For the (very unlikely) > 2^32-term operator we skip the lean table
    // and read the original size_t table via ladder(), so we never silently
    // truncate an offset.
    const std::size_t _ladder_len = num_groups * ladder_offset + 1;
    const bool _ladder_fits = terms.size() <= UINT32_MAX;
    std::vector<std::uint32_t> _ladder32;
    if(_ladder_fits)
    {
        _ladder32.resize(_ladder_len);
        for(std::size_t i = 0; i < _ladder_len; ++i)
            _ladder32[i] = static_cast<std::uint32_t>(group_ladder_ptrs[i]);
    }
    const std::uint32_t* __restrict ladder32 = _ladder32.data();

    auto ladder = [&](std::size_t idx) -> std::size_t {
        return _ladder_fits ? static_cast<std::size_t>(ladder32[idx]) : group_ladder_ptrs[idx];
    };

    // BLK defines number of rows that we process per group in one iter.
    // Group middle x block of rows inner iteration order cut down cache misses
    // significantly. BLK=128 is a good middle-ground.
    // A too large of BLK may overflow cache.
    // User can override it through env var FQ_BLK.
    std::size_t BLK = 128;
    if(const char* _blk_env = std::getenv("FQ_BLK"))
    {
        long _blk = std::atol(_blk_env);
        if(_blk > 0)
            BLK = static_cast<std::size_t>(_blk);
    }

    cols.resize(subspace_dim);
    data.resize(subspace_dim);

    // Categorize groups into 5 buckets based on offdiag indices
    std::vector<std::size_t> aa_groups;
    std::vector<std::size_t> aaaa_groups;
    std::vector<std::size_t> bb_groups;
    std::vector<std::size_t> bbbb_groups;
    std::vector<std::size_t> aabb_groups;
    std::vector<std::size_t> other_groups; // Groups that don't fit the 5 categories

    const width_t half_width = width / 2;
    for(std::size_t g = 0; g < num_groups; g++)
    {
        const GroupIndsView inds = gview(g);
        const std::size_t ind_size = inds.size();

        if(ind_size == 2)
        {
            // Check if both indices are in alpha half (aa) or beta half (bb)
            if(inds[1] < half_width)
            {
                aa_groups.push_back(g);
            }
            else if(inds[0] >= half_width)
            {
                bb_groups.push_back(g);
            }
            else
            {
                // Mixed case: one index < half_width, one >= half_width
                other_groups.push_back(g);
            }
        }
        else if(ind_size == 4)
        {
            // Check if all 4 indices are in alpha half (aaaa), beta half (bbbb), or split (aabb)
            if(inds[3] < half_width)
            {
                aaaa_groups.push_back(g);
            }
            else if(inds[0] >= half_width)
            {
                bbbb_groups.push_back(g);
            }
            else if(inds[1] < half_width && inds[2] >= half_width)
            {
                aabb_groups.push_back(g);
            }
            else
            {
                // Other 4-index patterns
                other_groups.push_back(g);
            }
        }
        else
        {
            // Groups with sizes other than 2 or 4
            other_groups.push_back(g);
        }
    }

    // Partition aabb_groups into a fast path (single coeff + real_phase shared
    // by all terms in the group -> matrix element evaluates as
    // aabb_direct[g] * asign * bsign) and a slow fallback path that goes
    // through accum_element per term.  aabb_direct is only meaningful for
    // groups in aabb_fast_groups; aabb_slow_groups read it as zero.
    std::vector<T> aabb_direct(num_groups, 0);
    std::vector<std::size_t> aabb_fast_groups;
    std::vector<std::size_t> aabb_slow_groups;
    aabb_fast_groups.reserve(aabb_groups.size());

    for(const auto& g : aabb_groups)
    {
        std::size_t group_start = group_ptrs[g];
        std::size_t group_stop = group_ptrs[g + 1];

        if(group_start >= group_stop)
        {
            continue; // Empty group, nothing to emit on either path
        }

        const OperatorTerm_t& first_term = terms[group_start];
        std::complex<double> group_coeff = first_term.coeff;
        int group_real_phase = first_term.real_phase;

        bool fast_eligible = true;
        for(std::size_t idx = group_start; idx < group_stop; idx++)
        {
            const OperatorTerm_t& term = terms[idx];
            if(term.real_phase != group_real_phase || std::abs(term.coeff - group_coeff) > 1e-14)
            {
                fast_eligible = false;
                break;
            }
        }

        if(fast_eligible)
        {
            if constexpr(std::is_same_v<T, double>)
            {
                aabb_direct[g] = group_real_phase * group_coeff.real();
            }
            else
            {
                aabb_direct[g] = static_cast<T>(group_real_phase) * group_coeff;
            }
            aabb_fast_groups.push_back(g);
        }
        else
        {
            aabb_slow_groups.push_back(g);
        }
    }

    // aabb fast-path prefilter setup
    // The aabb hot loop is O(subspace_dim * num_aabb_groups) with mostly misses
    // (most columns are not in the subspace).
    //
    // An aabb group flips two alpha bits and two beta bits, so the flipped column's
    // alpha half must equal some subspace element's alpha half AND its beta
    // half must equal some subspace element's beta half.
    //
    // We hash each half and, per row, evaluate the condition once
    // per unique alpha pair (alpha group inds, aa) and beta pair (beta group inds, bb)
    // instead of once per group. aabb groups are then visited grouped by alpha pair,
    // so a failing alpha pair skips all its groups without touching the subspace hash map.
    auto region_hash = [&](const boost::dynamic_bitset<std::size_t>& bs,
                           width_t lo,
                           width_t hi,
                           width_t fa,
                           width_t fb) -> std::uint64_t {
        const std::size_t lo_blk = lo >> BLOCK_EXPONENT;
        const std::size_t hi_blk = static_cast<std::size_t>(hi - 1) >> BLOCK_EXPONENT;
        const std::size_t n_blk = hi_blk - lo_blk + 1;
        static thread_local std::vector<std::size_t> buf;
        if(buf.size() < n_blk)
            buf.resize(n_blk);
        for(std::size_t b = lo_blk; b <= hi_blk; ++b)
        {
            std::size_t w = bs.m_bits[b];
            if(fa != MAX_WIDTH && (fa >> BLOCK_EXPONENT) == b)
                w ^= (std::size_t(1) << (fa & BLOCK_SHIFT));
            if(fb != MAX_WIDTH && (fb >> BLOCK_EXPONENT) == b)
                w ^= (std::size_t(1) << (fb & BLOCK_SHIFT));
            if(b == lo_blk)
                w &= (~std::size_t(0) << (lo & BLOCK_SHIFT));
            if(b == hi_blk)
            {
                const std::size_t off = static_cast<std::size_t>(hi - 1) & BLOCK_SHIFT;
                w &= (off == BLOCK_SHIFT) ? ~std::size_t(0)
                                          : (~std::size_t(0) >> (BLOCK_SHIFT - off));
            }
            buf[b - lo_blk] = w;
        }
        return rapidhashMicro(buf.data(), n_blk * sizeof(std::size_t));
    };

    emhash8::HashSet<std::uint64_t> alpha_half_hashes;
    emhash8::HashSet<std::uint64_t> beta_half_hashes;
    std::vector<std::array<width_t, 2>> a_pairs;
    std::vector<std::array<width_t, 2>> b_pairs;
    struct AabbEntry
    {
        std::uint32_t b_id;
        std::size_t g;
    };
    std::vector<std::vector<AabbEntry>> aabb_by_alpha;

    if(!aabb_fast_groups.empty())
    {
        for(std::size_t s = 0; s < subspace_dim; ++s)
        {
            const auto& bs = bitsets[s].first;
            alpha_half_hashes.insert(region_hash(bs, 0, half_width, MAX_WIDTH, MAX_WIDTH));
            beta_half_hashes.insert(region_hash(bs, half_width, width, MAX_WIDTH, MAX_WIDTH));
        }

        std::unordered_map<std::uint32_t, std::uint32_t> a_pair_id;
        std::unordered_map<std::uint32_t, std::uint32_t> b_pair_id;
        for(const auto& g : aabb_fast_groups)
        {
            const GroupIndsView inds = gview(g);
            const width_t ap0 = inds[0], ap1 = inds[1], bp0 = inds[2], bp1 = inds[3];
            const std::uint32_t ak = (std::uint32_t(ap0) << 16) | ap1;
            const std::uint32_t bk = (std::uint32_t(bp0) << 16) | bp1;

            std::uint32_t a_id;
            auto ai = a_pair_id.find(ak);
            if(ai == a_pair_id.end())
            {
                a_id = static_cast<std::uint32_t>(a_pairs.size());
                a_pair_id.emplace(ak, a_id);
                a_pairs.push_back({ap0, ap1});
                aabb_by_alpha.emplace_back();
            }
            else
            {
                a_id = ai->second;
            }

            std::uint32_t b_id;
            auto bi = b_pair_id.find(bk);
            if(bi == b_pair_id.end())
            {
                b_id = static_cast<std::uint32_t>(b_pairs.size());
                b_pair_id.emplace(bk, b_id);
                b_pairs.push_back({bp0, bp1});
            }
            else
            {
                b_id = bi->second;
            }
            aabb_by_alpha[a_id].push_back({b_id, g});
        }
    }
    const std::size_t num_a_pairs = a_pairs.size();
    const std::size_t num_b_pairs = b_pairs.size();

    // Populate diagonal elements first separately
    if(has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
        for(kk = 0; kk < subspace_dim; kk++)
        { // begin loop over all rows

            if(diag_vec[kk] != 0.0)
            {
                cols[kk].push_back(kk);
                data[kk].push_back(diag_vec[kk]);
            }
        }
    }

    const std::size_t rsb_w = width; // one uint8 per qubit
    const std::size_t num_blocks = (subspace_dim + BLK - 1) / BLK;

    // New loop order: block-of-rows outer, group middle, row inner
    // for block_of_rows in rows:
    //   for g in groups:
    //     for row in block_of_rows:
    //       ...
    // In group_ladder_ptrs[g * ladder_offset + row_int] lookup, row_int varies only over [0, ladder_offset - 1]
    // (= [0,3] if ladder_width=2, [0,15] if ladder_width=4). For a fixed g, the first
    // row in the block of rows fetches a cache-line of group_ladder_ptrs[] that can be resued by remaining rows
    // while being cache resident.
#pragma omp parallel if(subspace_dim > 4096)
    {
        // Per-thread scratch, reused across blocks (no per-block realloc).
        std::vector<uint8_t> rsb_buf;
        std::vector<char> alpha_ok(num_a_pairs);
        std::vector<char> beta_ok(num_b_pairs);
        boost::dynamic_bitset<std::size_t> col_vec;

#pragma omp for schedule(dynamic)
        for(std::size_t blk = 0; blk < num_blocks; ++blk)
        {
            const std::size_t r0 = blk * BLK;
            const std::size_t r1 = std::min(r0 + BLK, subspace_dim);
            const std::size_t bn = r1 - r0;

            // row_set_bits for every row in the block, 1D contiguous vec.
            rsb_buf.assign(bn * rsb_w, 0);
            for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
            {
                const boost::dynamic_bitset<std::size_t>& row = bitsets[r0 + row_in_block].first;
                uint8_t* dst = rsb_buf.data() + row_in_block * rsb_w;
                for(std::size_t b = 0; b < row.num_blocks(); ++b)
                {
                    std::size_t bits = row.m_bits[b];
                    while(bits != 0)
                    {
                        int r = __builtin_ctzll(bits);
                        dst[b * BITS_PER_BLOCK + r] = 1;
                        bits &= bits - 1;
                    }
                }
            }

            // Standard per-group path for one (group g, row_in_block) pair.
            auto process_standard_group = [&](std::size_t g, std::size_t row_in_block) {
                const GroupIndsView group_inds = gview(g);
                const uint8_t* row_set_bits = rsb_buf.data() + row_in_block * rsb_w;

                // Hamming weight check.
                const width_t _p = group_inds[0];
                const width_t _q = group_inds[1];
                if(group_inds.size() == 2)
                {
                    if(row_set_bits[_p] == row_set_bits[_q])
                        return;
                }
                else if(group_inds.size() == 4)
                {
                    const width_t _r = group_inds[2];
                    const width_t _s = group_inds[3];
                    if(row_set_bits[_p] + row_set_bits[_q] + row_set_bits[_r] + row_set_bits[_s] !=
                       2)
                        return;
                }

                const unsigned int row_int =
                    bitset_ladder_int(row_set_bits, group_inds.data(), group_rowint_length[g]);
                const std::size_t group_int_start = ladder(g * ladder_offset + row_int);
                const std::size_t group_int_stop = ladder(g * ladder_offset + row_int + 1);
                if(group_int_start >= group_int_stop)
                    return;

                const boost::dynamic_bitset<std::size_t>& row = bitsets[r0 + row_in_block].first;
                col_vec = row;
                flip_bits(col_vec, group_inds.data(), group_inds.size());

                std::size_t* col_ptr = subspace.get_ptr(col_vec);
                if(col_ptr == nullptr)
                    return;
                const std::size_t col_idx = *col_ptr;

                T val = 0;
                for(std::size_t idx = group_int_start; idx < group_int_stop; idx++)
                {
                    const OperatorTerm_t* term = &terms[idx];
                    if(passes_proj_validation(term, row))
                    {
                        accum_element(row,
                                      col_vec,
                                      term->indices,
                                      term->values,
                                      term->coeff,
                                      term->real_phase,
                                      term->indices.size(),
                                      val);
                    }
                }

                if(std::abs(val) > ATOL)
                {
                    cols[r0 + row_in_block].push_back(col_idx);
                    data[r0 + row_in_block].push_back(val);
                }
            };

            // aa / aaaa / bb (group-outer, row-inner).
            for(const auto& g : aa_groups)
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                    process_standard_group(g, row_in_block);
            for(const auto& g : aaaa_groups)
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                    process_standard_group(g, row_in_block);
            for(const auto& g : bb_groups)
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                    process_standard_group(g, row_in_block);

            // aabb fast path, ROW-major within the block to preserve the
            // per-row alpha/beta prefilter skip (a failing alpha pair drops all
            // its groups at once).
            if(!aabb_fast_groups.empty())
            {
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                {
                    const boost::dynamic_bitset<std::size_t>& row =
                        bitsets[r0 + row_in_block].first;
                    const uint8_t* row_set_bits = rsb_buf.data() + row_in_block * rsb_w;

                    auto range_parity = [&](width_t lo, width_t hi) -> std::size_t {
                        if(lo >= hi)
                            return 0;
                        const width_t last = hi - 1; // inclusive last bit
                        const std::size_t lo_blk = lo >> BLOCK_EXPONENT;
                        const std::size_t hi_blk = static_cast<std::size_t>(last) >> BLOCK_EXPONENT;
                        const std::size_t lo_mask = ~std::size_t(0) << (lo & BLOCK_SHIFT);
                        const std::size_t hi_mask =
                            ~std::size_t(0) >> (BLOCK_SHIFT - (last & BLOCK_SHIFT));
                        std::size_t acc;
                        if(lo_blk == hi_blk)
                        {
                            acc = row.m_bits[lo_blk] & lo_mask & hi_mask;
                        }
                        else
                        {
                            acc = row.m_bits[lo_blk] & lo_mask;
                            for(std::size_t b = lo_blk + 1; b < hi_blk; b++)
                                acc ^= row.m_bits[b];
                            acc ^= row.m_bits[hi_blk] & hi_mask;
                        }
                        return static_cast<std::size_t>(
                            __builtin_parityll(static_cast<unsigned long long>(acc)));
                    };

                    for(std::size_t i = 0; i < num_a_pairs; ++i)
                    {
                        const width_t p0 = a_pairs[i][0];
                        const width_t p1 = a_pairs[i][1];
                        if(row_set_bits[p0] == row_set_bits[p1])
                        {
                            alpha_ok[i] = 0;
                            continue;
                        }
                        alpha_ok[i] =
                            alpha_half_hashes.contains(region_hash(row, 0, half_width, p0, p1));
                    }
                    for(std::size_t j = 0; j < num_b_pairs; ++j)
                    {
                        const width_t p0 = b_pairs[j][0];
                        const width_t p1 = b_pairs[j][1];
                        if(row_set_bits[p0] == row_set_bits[p1])
                        {
                            beta_ok[j] = 0;
                            continue;
                        }
                        beta_ok[j] =
                            beta_half_hashes.contains(region_hash(row, half_width, width, p0, p1));
                    }

                    for(std::size_t i = 0; i < num_a_pairs; ++i)
                    {
                        if(!alpha_ok[i])
                            continue;
                        const width_t ap0 = a_pairs[i][0];
                        const width_t ap1 = a_pairs[i][1];
                        for(const auto& e : aabb_by_alpha[i])
                        {
                            if(!beta_ok[e.b_id])
                                continue;
                            const GroupIndsView group_inds = gview(e.g);
                            const width_t bp0 = group_inds[2];
                            const width_t bp1 = group_inds[3];

                            const unsigned int row_int = bitset_ladder_int(
                                row_set_bits, group_inds.data(), group_rowint_length[e.g]);
                            const std::size_t group_int_start =
                                ladder(e.g * ladder_offset + row_int);
                            const std::size_t group_int_stop =
                                ladder(e.g * ladder_offset + row_int + 1);
                            if(group_int_start >= group_int_stop)
                                continue;

                            col_vec = row;
                            flip_bits(col_vec, group_inds.data(), group_inds.size());
                            std::size_t* col_ptr = subspace.get_ptr(col_vec);
                            if(col_ptr == nullptr)
                                continue;
                            const std::size_t col_idx = *col_ptr;

                            const std::size_t aabb_parity =
                                range_parity(ap0 + 1, ap1) ^ range_parity(bp0 + 1, bp1);
                            double sign = aabb_parity ? -1.0 : 1.0;
                            T val = aabb_direct[e.g] * static_cast<T>(sign);

                            if(std::abs(val) > ATOL)
                            {
                                cols[r0 + row_in_block].push_back(col_idx);
                                data[r0 + row_in_block].push_back(val);
                            }
                        }
                    }
                }
            }

            // aabb_slow / bbbb / other (group-outer, row-inner).
            for(const auto& g : aabb_slow_groups)
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                    process_standard_group(g, row_in_block);
            for(const auto& g : bbbb_groups)
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                    process_standard_group(g, row_in_block);
            for(const auto& g : other_groups)
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                    process_standard_group(g, row_in_block);
        }
    }

    sort_paired(cols, data);
} // end function

template <typename T, typename U>
void csrlike_builder2_halfstr(const HalfStrContext<T>& context,
                              const std::vector<OperatorTerm_t>& terms,
                              const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                              const T* __restrict diag_vec,
                              const std::size_t subspace_dim,
                              const int has_nonzero_diag,
                              const width_t* __restrict group_rowint_length,
                              const unsigned int ladder_offset,
                              std::vector<std::vector<U>>& cols,
                              std::vector<std::vector<T>>& data)
{
    if(has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
        for(std::size_t kk = 0; kk < subspace_dim; kk++)
            if(diag_vec[kk] != T(0))
            {
                cols[kk].push_back(static_cast<U>(kk));
                data[kk].push_back(diag_vec[kk]);
            }
    }
    halfstr_walk<T>(context,
                    terms,
                    subspace,
                    subspace_dim,
                    group_rowint_length,
                    ladder_offset,
                    [&](std::size_t out_row, std::size_t col_idx, T val) {
                        cols[out_row].push_back(static_cast<U>(col_idx));
                        data[out_row].push_back(val);
                    });
    sort_paired(cols, data);
}
