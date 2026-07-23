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
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "external/hash_set8.hpp"

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include "type2_common.hpp"
#include <boost/dynamic_bitset.hpp>
#ifdef _OPENMP
#    include <omp.h>
#endif

// ============================================================================
// The GENERIC type-2 element visitor -- the fallback that works on any subspace.
//
// Streams every off-diagonal nonzero (row, col, H[row][col]) to the sink. Groups are
// bucketed (aa / aaaa / bb / bbbb / aabb / other); aabb groups whose terms share a
// single coeff + real_phase use the direct asign*bsign formula instead of the per-term
// accum_element loop. other and aabb_slow exist for safety and are supposed to be empty.
//
// Partners are found by flipping the group's bits and probing the subspace hash map, so
// this needs no precomputation -- but it costs a hash probe per (group, row) candidate,
// most of which miss. When HalfStrTables can be built, type2_visit_cartesian /
// type2_visit_non_cartesian replace those probes with direct partner discovery and are orders of
// magnitude faster (measured 634 s -> 5.6 s per matvec on 1M fe4s4 determinants).
//
// sink(row, col, val) is called concurrently, but rows are partitioned across threads by
// the block loop, so each row is touched by exactly one thread and no sink needs locking.
// The diagonal is NOT emitted here; callers own it (they each store it differently).
// ============================================================================
template <typename T, typename Sink>
void type2_visit_groups(const std::vector<OperatorTerm_t>& terms,
                        const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                        const width_t width,
                        const std::size_t subspace_dim,
                        const std::size_t* __restrict group_ptrs,
                        const std::size_t* __restrict group_ladder_ptrs,
                        const width_t* __restrict group_rowint_length,
                        const std::vector<std::vector<width_t>>& group_offdiag_inds,
                        const std::size_t num_groups,
                        const unsigned int ladder_offset,
                        Sink sink)
{
    const auto* bitsets = subspace.get_bitsets();

    // The flatten is O(num_groups) and negligible vs the matvec.
    std::vector<width_t> _flat_inds;
    std::vector<std::size_t> _inds_offsets;
    flatten_offdiag_inds(group_offdiag_inds, _flat_inds, _inds_offsets);
    const width_t* __restrict flat_inds = _flat_inds.data();
    const std::size_t* __restrict inds_offsets = _inds_offsets.data();
    auto gview = [&](std::size_t g) -> GroupIndsView {
        const std::size_t off = inds_offsets[g];
        return GroupIndsView{flat_inds + off, inds_offsets[g + 1] - off};
    };

    const std::size_t _ladder_len = static_cast<std::size_t>(num_groups) * ladder_offset + 1;
    const bool _ladder_fits = terms.size() <= UINT32_MAX;
    std::vector<std::uint32_t> _ladder32;
    if(_ladder_fits)
    {
        _ladder32.resize(_ladder_len);
        for(std::size_t i = 0; i < _ladder_len; i++)
            _ladder32[i] = static_cast<std::uint32_t>(group_ladder_ptrs[i]);
    }
    const std::uint32_t* __restrict ladder32 = _ladder32.data();
    auto ladder = [&](std::size_t idx) -> std::size_t {
        return _ladder_fits ? static_cast<std::size_t>(ladder32[idx]) : group_ladder_ptrs[idx];
    };

    std::size_t BLK = 128;
    if(const char* _blk_env = std::getenv("FQ_BLK"))
    {
        long _blk = std::atol(_blk_env);
        if(_blk > 0)
            BLK = static_cast<std::size_t>(_blk);
    }
    const std::size_t rsb_w = width; // one uint8 per qubit
    const std::size_t num_blocks = (subspace_dim + BLK - 1) / BLK;

    // Categorize groups into buckets based on offdiag indices:
    // aa / aaaa / bb / bbbb / aabb / other.  half_width
    // splits the bitset into the alpha (lower) and beta (upper) sectors.
    std::vector<std::size_t> aa_groups;
    std::vector<std::size_t> aaaa_groups;
    std::vector<std::size_t> bb_groups;
    std::vector<std::size_t> bbbb_groups;
    std::vector<std::size_t> aabb_groups;
    std::vector<std::size_t> other_groups;

    const width_t half_width = width / 2;
    for(std::size_t g = 0; g < num_groups; g++)
    {
        const GroupIndsView inds = gview(g);
        const std::size_t ind_size = inds.size();

        if(ind_size == 2)
        {
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
                other_groups.push_back(g);
            }
        }
        else if(ind_size == 4)
        {
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
                other_groups.push_back(g);
            }
        }
        else
        {
            other_groups.push_back(g);
        }
    }

    // Partition aabb_groups into a fast path (single coeff + real_phase shared
    // by all terms in the group -> matrix element evaluates as
    // aabb_direct[g] * asign * bsign) and a slow fallback path that goes
    // through accum_element per term.
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
        for(std::size_t i = group_start; i < group_stop; i++)
        {
            const OperatorTerm_t& term = terms[i];
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
        for(std::size_t b = lo_blk; b <= hi_blk; b++)
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
        for(std::size_t s = 0; s < subspace_dim; s++)
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
            const std::uint32_t ak = pack2(ap0, ap1);
            const std::uint32_t bk = pack2(bp0, bp1);

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

#pragma omp parallel if(subspace_dim > 4096)
    {
        std::size_t num_terms = terms.size();
        // Take care of off-diagonal terms
        if(num_terms)
        {
            // Per-thread scratch, reused across blocks (no per-block realloc).
            std::vector<uint8_t> rsb_buf;
            std::vector<char> alpha_ok(num_a_pairs);
            std::vector<char> beta_ok(num_b_pairs);
            boost::dynamic_bitset<std::size_t> col_vec;

#pragma omp for schedule(dynamic)
            for(std::size_t blk = 0; blk < num_blocks; blk++)
            {
                const std::size_t r0 = blk * BLK;
                const std::size_t r1 = std::min(r0 + BLK, subspace_dim);
                const std::size_t bn = r1 - r0;

                // row_set_bits for every row in the block, contiguous.
                rsb_buf.assign(bn * rsb_w, 0);
                for(std::size_t row_in_block = 0; row_in_block < bn; row_in_block++)
                {
                    const boost::dynamic_bitset<std::size_t>& row =
                        bitsets[r0 + row_in_block].first;
                    uint8_t* dst = rsb_buf.data() + row_in_block * rsb_w;
                    for(std::size_t b = 0; b < row.num_blocks(); b++)
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
                // out_vec[r0+row_in_block] is owned by this block/thread -> no mutex.
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
                        if(row_set_bits[_p] + row_set_bits[_q] + row_set_bits[_r] +
                               row_set_bits[_s] !=
                           2)
                            return;
                    }

                    const unsigned int row_int =
                        bitset_ladder_int(row_set_bits, group_inds.data(), group_rowint_length[g]);
                    const std::size_t group_int_start = ladder(g * ladder_offset + row_int);
                    const std::size_t group_int_stop = ladder(g * ladder_offset + row_int + 1);
                    if(group_int_start >= group_int_stop)
                        return;

                    const boost::dynamic_bitset<std::size_t>& row =
                        bitsets[r0 + row_in_block].first;
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
                        sink(r0 + row_in_block, col_idx, val);
                };

                for(const auto& g : aa_groups)
                    for(std::size_t row_in_block = 0; row_in_block < bn; row_in_block++)
                        process_standard_group(g, row_in_block);
                for(const auto& g : aaaa_groups)
                    for(std::size_t row_in_block = 0; row_in_block < bn; row_in_block++)
                        process_standard_group(g, row_in_block);
                for(const auto& g : bb_groups)
                    for(std::size_t row_in_block = 0; row_in_block < bn; row_in_block++)
                        process_standard_group(g, row_in_block);

                // aabb fast path, ROW-major within the block to preserve the
                // per-row alpha/beta prefilter skip.
                if(!aabb_fast_groups.empty())
                {
                    for(std::size_t row_in_block = 0; row_in_block < bn; row_in_block++)
                    {
                        const boost::dynamic_bitset<std::size_t>& row =
                            bitsets[r0 + row_in_block].first;
                        const uint8_t* row_set_bits = rsb_buf.data() + row_in_block * rsb_w;

                        auto range_parity = [&](width_t lo, width_t hi) -> std::size_t {
                            if(lo >= hi)
                                return 0;
                            const width_t last = hi - 1; // inclusive last bit
                            const std::size_t lo_blk = lo >> BLOCK_EXPONENT;
                            const std::size_t hi_blk =
                                static_cast<std::size_t>(last) >> BLOCK_EXPONENT;
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

                        for(std::size_t i = 0; i < num_a_pairs; i++)
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
                        for(std::size_t j = 0; j < num_b_pairs; j++)
                        {
                            const width_t p0 = b_pairs[j][0];
                            const width_t p1 = b_pairs[j][1];
                            if(row_set_bits[p0] == row_set_bits[p1])
                            {
                                beta_ok[j] = 0;
                                continue;
                            }
                            beta_ok[j] = beta_half_hashes.contains(
                                region_hash(row, half_width, width, p0, p1));
                        }

                        for(std::size_t i = 0; i < num_a_pairs; i++)
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
                                    sink(r0 + row_in_block, col_idx, val);
                            }
                        }
                    }
                }

                for(const auto& g : aabb_slow_groups)
                    for(std::size_t row_in_block = 0; row_in_block < bn; row_in_block++)
                        process_standard_group(g, row_in_block);
                for(const auto& g : bbbb_groups)
                    for(std::size_t row_in_block = 0; row_in_block < bn; row_in_block++)
                        process_standard_group(g, row_in_block);
                for(const auto& g : other_groups)
                    for(std::size_t row_in_block = 0; row_in_block < bn; row_in_block++)
                        process_standard_group(g, row_in_block);
            } // end for-loop over blocks
        } // end if num_terms
    } // end parallel region
} // end type2_visit_groups
