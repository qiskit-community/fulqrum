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
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "external/hash_set8.hpp"
#include "external/hash_table8.hpp"

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include <boost/dynamic_bitset.hpp>

// Hermitian sparse matrix-vector product over a subspace Hamiltonian.
//
// Each row is owned by a single OpenMP iteration, so out_vec[kk]
// needs no locking.
// Group handling mirrors csrlike_builder2: groups are bucketed
// (aa / aaaa / bb / bbbb / aabb / other) and aabb groups whose
// terms share a single coeff + real_phase use the direct
// asign*bsign formula instead of the per-term accum_element loop.
// other and aabb_slow groups are here for safety.
// They are supposed to be empty.
template <typename T>
void omp_matvec2(const std::vector<OperatorTerm_t>& terms,
                 const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                 const T* __restrict diag_vec,
                 const width_t width,
                 const std::size_t subspace_dim,
                 const int has_nonzero_diag,
                 const std::size_t* __restrict group_ptrs,
                 const std::size_t* __restrict group_ladder_ptrs,
                 const width_t* __restrict group_rowint_length,
                 const std::vector<std::vector<width_t>>& group_offdiag_inds,
                 const unsigned int num_groups,
                 const unsigned int ladder_offset,
                 const T* __restrict in_vec,
                 T* __restrict out_vec)
{
    std::size_t kk;
    const auto* bitsets = subspace.get_bitsets();

    // See csrlike_builder2.hpp for the rationale.
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

    // See csrlike_builder2.hpp for the rationale.
    const std::size_t _ladder_len = static_cast<std::size_t>(num_groups) * ladder_offset + 1;
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

    std::size_t BLK = 128;
    if(const char* _blk_env = std::getenv("FQ_BLK"))
    {
        long _blk = std::atol(_blk_env);
        if(_blk > 0)
            BLK = static_cast<std::size_t>(_blk);
    }
    const std::size_t rsb_w = width; // one uint8 per qubit
    const std::size_t num_blocks = (subspace_dim + BLK - 1) / BLK;

    // Categorize groups into buckets based on offdiag indices (mirror of
    // csrlike_builder2): aa / aaaa / bb / bbbb / aabb / other.  half_width
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
    // See csrlike_builder2.hpp for details.
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

    // Take care of diagonal term first, if any (usually there is)
    if(has_nonzero_diag)
    {
#pragma omp for
        for(kk = 0; kk < subspace_dim; kk++)
        {
            out_vec[kk] = diag_vec[kk] * in_vec[kk];
        }
    }

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

            // see csrlike_builder2.hpp for details.
#pragma omp for schedule(dynamic)
            for(std::size_t blk = 0; blk < num_blocks; ++blk)
            {
                const std::size_t r0 = blk * BLK;
                const std::size_t r1 = std::min(r0 + BLK, subspace_dim);
                const std::size_t bn = r1 - r0;

                // row_set_bits for every row in the block, contiguous.
                rsb_buf.assign(bn * rsb_w, 0);
                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                {
                    const boost::dynamic_bitset<std::size_t>& row =
                        bitsets[r0 + row_in_block].first;
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
                        out_vec[r0 + row_in_block] += (val * in_vec[col_idx]);
                };

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
                // per-row alpha/beta prefilter skip.
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
                            beta_ok[j] = beta_half_hashes.contains(
                                region_hash(row, half_width, width, p0, p1));
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
                                    out_vec[r0 + row_in_block] += (val * in_vec[col_idx]);
                            }
                        }
                    }
                }

                for(const auto& g : aabb_slow_groups)
                    for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                        process_standard_group(g, row_in_block);
                for(const auto& g : bbbb_groups)
                    for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                        process_standard_group(g, row_in_block);
                for(const auto& g : other_groups)
                    for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                        process_standard_group(g, row_in_block);
            } // end for-loop over blocks
        } // end if num_terms
    } // end parallel region
} // end matvec

// ============================================================================
// Faster path for type-2 matvec for half-strs mode.
// ============================================================================

constexpr std::size_t NO_HALF_G = MAX_SIZE_T;

// One single-excitation connection within a spin sector.
struct HalfConnSingle
{
    std::size_t col; // connected alpha (or beta) string index
    std::size_t g; // group for the same-sector (aa/bb) value; NO_HALF_G if none
    int pair_rank; // -1 if pair not in any aabb group
    int sign; // factorized aabb sign (+/-1) for this string + pair
};

// One double-excitation connection within a spin sector.
struct HalfConnDouble
{
    std::size_t col; // connected string index
    std::size_t g; // group for the same-sector (aaaa/bbbb) value
};

template <typename T>
struct HalfStrContext
{
    bool usable = false;
    width_t width = 0;
    width_t half_width = 0;
    std::size_t num_alpha = 0;
    std::size_t num_beta = 0;
    std::size_t num_a_pairs = 0;
    std::size_t num_b_pairs = 0;
    std::size_t BLK = 128;
    std::size_t rsb_w = 0;
    std::size_t num_blocks = 0;
    // flattened offdiag inds + ladder (for same-sector value eval)
    std::vector<width_t> flat_inds;
    std::vector<std::size_t> inds_offsets;
    std::vector<std::uint32_t> ladder32;
    // per-string excitation lists
    std::vector<std::vector<HalfConnSingle>> alpha_single;
    std::vector<std::vector<HalfConnDouble>> alpha_double;
    std::vector<std::vector<HalfConnSingle>> beta_single;
    std::vector<std::vector<HalfConnDouble>> beta_double;
    // aabb coupling: aabb_val_2d[ra * num_b_pairs + rb] = direct coeff (0 if none)
    std::vector<T> aabb_val_2d;
};

template <std::size_t K>
struct HalfKey
{
    std::array<std::uint64_t, K> w{};
    bool operator==(const HalfKey& o) const noexcept
    {
        return w == o.w;
    }
    bool operator!=(const HalfKey& o) const noexcept
    {
        return !(w == o.w);
    }
};

template <std::size_t K>
struct HalfKeyHash
{
    std::size_t operator()(const HalfKey<K>& k) const noexcept
    {
        return static_cast<std::size_t>(rapidhashMicro(k.w.data(), K * sizeof(std::uint64_t)));
    }
};

template <std::size_t K>
inline HalfKey<K>
extract_half(const boost::dynamic_bitset<std::size_t>& bs, width_t lo, width_t len)
{
    HalfKey<K> out;
    const std::size_t nb = bs.num_blocks();
    const std::size_t word_shift = static_cast<std::size_t>(lo) >> 6;
    const unsigned bit_shift = static_cast<unsigned>(lo) & 63u;
    for(std::size_t wi = 0; wi < K; ++wi)
    {
        std::uint64_t v = (word_shift + wi < nb) ? (bs.m_bits[word_shift + wi] >> bit_shift) : 0;
        if(bit_shift && (word_shift + wi + 1 < nb))
            v |= bs.m_bits[word_shift + wi + 1] << (64u - bit_shift);
        out.w[wi] = v;
    }
    for(std::size_t wi = 0; wi < K; ++wi)
    {
        const std::size_t word_lo = wi * 64;
        if(word_lo >= len)
            out.w[wi] = 0;
        else if(static_cast<std::size_t>(len) - word_lo < 64)
            out.w[wi] &= (std::uint64_t(1) << (static_cast<std::size_t>(len) - word_lo)) - 1;
    }
    return out;
}

template <std::size_t K>
inline void hk_flip(HalfKey<K>& k, width_t p) noexcept
{
    k.w[static_cast<std::size_t>(p) >> 6] ^= std::uint64_t(1) << (static_cast<unsigned>(p) & 63u);
}

template <std::size_t K>
inline bool hk_test(const HalfKey<K>& k, width_t p) noexcept
{
    return (k.w[static_cast<std::size_t>(p) >> 6] >> (static_cast<unsigned>(p) & 63u)) & 1ULL;
}

// Parity of set bits in [lo, hi). Used for the factorized aabb sign.
template <std::size_t K>
inline int hk_range_parity(const HalfKey<K>& k, width_t lo, width_t hi) noexcept
{
    if(lo >= hi)
        return 0;
    const width_t last = static_cast<width_t>(hi - 1);
    const std::uint64_t lo_mask = ~std::uint64_t(0) << (static_cast<unsigned>(lo) & 63u);
    const std::uint64_t hi_mask = ~std::uint64_t(0) >> (63u - (static_cast<unsigned>(last) & 63u));
    if constexpr(K == 1)
    {
        return __builtin_popcountll(k.w[0] & lo_mask & hi_mask) & 1;
    }
    else
    {
        const std::size_t lo_w = static_cast<std::size_t>(lo) >> 6;
        const std::size_t hi_w = static_cast<std::size_t>(last) >> 6;
        int acc;
        if(lo_w == hi_w)
            acc = __builtin_popcountll(k.w[lo_w] & lo_mask & hi_mask);
        else
        {
            acc = __builtin_popcountll(k.w[lo_w] & lo_mask);
            for(std::size_t w = lo_w + 1; w < hi_w; ++w)
                acc += __builtin_popcountll(k.w[w]);
            acc += __builtin_popcountll(k.w[hi_w] & hi_mask);
        }
        return acc & 1;
    }
}

template <std::size_t K>
struct FixedKeyOps
{
    using Key = HalfKey<K>;
    template <typename V>
    using Map = emhash8::HashMap<Key, V, HalfKeyHash<K>>;
    static constexpr std::size_t words = K;
    static Key extract(const boost::dynamic_bitset<std::size_t>& bs, width_t lo, width_t len)
    {
        return extract_half<K>(bs, lo, len);
    }
    static void flip(Key& k, width_t p) noexcept
    {
        hk_flip<K>(k, p);
    }
    static bool test(const Key& k, width_t p) noexcept
    {
        return hk_test<K>(k, p);
    }
    static int range_parity(const Key& k, width_t lo, width_t hi) noexcept
    {
        return hk_range_parity<K>(k, lo, hi);
    }
};

struct DynBitsetHash
{
    std::size_t operator()(const boost::dynamic_bitset<std::size_t>& k) const noexcept
    {
        return static_cast<std::size_t>(
            rapidhashMicro(k.m_bits.data(), k.num_blocks() * sizeof(std::size_t)));
    }
};

struct DynKeyOps
{
    using Key = boost::dynamic_bitset<std::size_t>;
    template <typename V>
    using Map = std::unordered_map<Key, V, DynBitsetHash>;
    static constexpr std::size_t words = 0; // 0 => reported as "dyn" in verbose
    static Key extract(const boost::dynamic_bitset<std::size_t>& bs, width_t lo, width_t len)
    {
        Key out(len);
        for(width_t i = 0; i < len; ++i)
            if(bs.test(static_cast<std::size_t>(lo) + i))
                out.set(static_cast<std::size_t>(i));
        return out;
    }
    static void flip(Key& k, width_t p) noexcept
    {
        k.flip(static_cast<std::size_t>(p));
    }
    static bool test(const Key& k, width_t p) noexcept
    {
        return k.test(static_cast<std::size_t>(p));
    }
    static int range_parity(const Key& k, width_t lo, width_t hi) noexcept
    {
        int acc = 0;
        for(width_t p = lo; p < hi; ++p)
            acc ^= static_cast<int>(k.test(static_cast<std::size_t>(p)));
        return acc;
    }
};

template <typename T, typename Policy>
void build_halfstr_context_impl(const std::vector<OperatorTerm_t>& terms,
                                const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                                const width_t width,
                                const std::size_t subspace_dim,
                                const std::size_t* __restrict group_ptrs,
                                const std::size_t* __restrict group_ladder_ptrs,
                                const std::vector<std::vector<width_t>>& group_offdiag_inds,
                                const unsigned int num_groups,
                                const unsigned int ladder_offset,
                                HalfStrContext<T>& context)
{
    using Key = typename Policy::Key;
    const width_t hw = static_cast<width_t>(width / 2);
    const width_t bw = static_cast<width_t>(width - hw);
    context.width = width;
    context.half_width = hw;

    const auto* bitsets = subspace.get_bitsets();
    auto akey = [&](const boost::dynamic_bitset<std::size_t>& bs) -> Key {
        return Policy::extract(bs, 0, hw);
    };
    auto bkey = [&](const boost::dynamic_bitset<std::size_t>& bs) -> Key {
        return Policy::extract(bs, hw, bw);
    };

    const Key b0 = bkey(bitsets[0].first);
    std::size_t num_alpha = 1;
    while(num_alpha < subspace_dim && bkey(bitsets[num_alpha].first) == b0)
        ++num_alpha;
    if(num_alpha == 0 || subspace_dim % num_alpha != 0)
        return;
    const std::size_t num_beta = subspace_dim / num_alpha;

    std::vector<Key> akeys(num_alpha), bkeys(num_beta);
    for(std::size_t a = 0; a < num_alpha; ++a)
        akeys[a] = akey(bitsets[a].first);
    for(std::size_t b = 0; b < num_beta; ++b)
        bkeys[b] = bkey(bitsets[b * num_alpha].first);

    // Verify a clean Cartesian product. Otherwise fall back.
    for(std::size_t idx = 0; idx < subspace_dim; ++idx)
    {
        const std::size_t b = idx / num_alpha;
        const std::size_t a = idx % num_alpha;
        const auto& bs = bitsets[idx].first;
        if(akey(bs) != akeys[a] || bkey(bs) != bkeys[b])
            return;
    }

    typename Policy::template Map<std::uint32_t> amap, bmap;
    amap.reserve(num_alpha * 2 + 1);
    bmap.reserve(num_beta * 2 + 1);
    for(std::uint32_t a = 0; a < num_alpha; ++a)
        amap[akeys[a]] = a;
    for(std::uint32_t b = 0; b < num_beta; ++b)
        bmap[bkeys[b]] = b;

    flatten_offdiag_inds(group_offdiag_inds, context.flat_inds, context.inds_offsets);
    auto gview = [&](std::size_t g) -> GroupIndsView {
        const std::size_t off = context.inds_offsets[g];
        return GroupIndsView{context.flat_inds.data() + off, context.inds_offsets[g + 1] - off};
    };
    const std::size_t ladder_len = static_cast<std::size_t>(num_groups) * ladder_offset + 1;
    context.ladder32.resize(ladder_len);
    for(std::size_t i = 0; i < ladder_len; ++i)
        context.ladder32[i] = static_cast<std::uint32_t>(group_ladder_ptrs[i]);

    auto pack2 = [](width_t a, width_t b) -> std::uint32_t {
        return (static_cast<std::uint32_t>(a) << 16) | static_cast<std::uint32_t>(b);
    };
    auto pack4 = [](width_t a, width_t b, width_t c, width_t d) -> std::uint64_t {
        return (static_cast<std::uint64_t>(a) << 48) | (static_cast<std::uint64_t>(b) << 32) |
               (static_cast<std::uint64_t>(c) << 16) | static_cast<std::uint64_t>(d);
    };

    // Invert groups: flip-position set -> group id.
    emhash8::HashMap<std::uint32_t, std::size_t> aa_map, bb_map;
    emhash8::HashMap<std::uint64_t, std::size_t> aaaa_map, bbbb_map;
    std::unordered_map<std::uint32_t, int> a_pair_id, b_pair_id;
    struct AabbTmp
    {
        int ra;
        int rb;
        T val;
    };
    std::vector<AabbTmp> aabb_tmp;

    for(std::size_t g = 0; g < num_groups; ++g)
    {
        const GroupIndsView inds = gview(g);
        const std::size_t sz = inds.size();
        const bool nonempty = group_ptrs[g + 1] > group_ptrs[g];
        if(sz == 2)
        {
            if(inds[1] < hw)
                aa_map[pack2(inds[0], inds[1])] = g;
            else if(inds[0] >= hw)
                bb_map[pack2(static_cast<width_t>(inds[0] - hw),
                             static_cast<width_t>(inds[1] - hw))] = g;
            else if(nonempty)
                return; // mixed single excitation -> unsupported
        }
        else if(sz == 4)
        {
            if(inds[3] < hw)
                aaaa_map[pack4(inds[0], inds[1], inds[2], inds[3])] = g;
            else if(inds[0] >= hw)
                bbbb_map[pack4(static_cast<width_t>(inds[0] - hw),
                               static_cast<width_t>(inds[1] - hw),
                               static_cast<width_t>(inds[2] - hw),
                               static_cast<width_t>(inds[3] - hw))] = g;
            else if(inds[1] < hw && inds[2] >= hw)
            {
                if(!nonempty)
                    continue;
                // Direct-formula eligibility: all terms share coeff + real_phase.
                const OperatorTerm_t& first = terms[group_ptrs[g]];
                const std::complex<double> gc = first.coeff;
                const int gp = first.real_phase;
                for(std::size_t i = group_ptrs[g]; i < group_ptrs[g + 1]; ++i)
                    if(terms[i].real_phase != gp || std::abs(terms[i].coeff - gc) > 1e-14)
                        return; // aabb slow path -> unsupported
                T val;
                if constexpr(std::is_same_v<T, double>)
                    val = gp * gc.real();
                else
                    val = static_cast<T>(gp) * gc;
                const std::uint32_t ak = pack2(inds[0], inds[1]);
                const std::uint32_t bk =
                    pack2(static_cast<width_t>(inds[2] - hw), static_cast<width_t>(inds[3] - hw));
                auto ai = a_pair_id.find(ak);
                const int ra = (ai == a_pair_id.end())
                                   ? (a_pair_id[ak] = static_cast<int>(a_pair_id.size()))
                                   : ai->second;
                auto bi = b_pair_id.find(bk);
                const int rb = (bi == b_pair_id.end())
                                   ? (b_pair_id[bk] = static_cast<int>(b_pair_id.size()))
                                   : bi->second;
                aabb_tmp.push_back({ra, rb, val});
            }
            else if(nonempty)
                return; // other 4-index class -> unsupported
        }
        else if(nonempty)
            return; // weight not in {2,4} -> unsupported
    }

    const std::size_t num_a_pairs = a_pair_id.size();
    const std::size_t num_b_pairs = b_pair_id.size();
    context.aabb_val_2d.assign(num_a_pairs * num_b_pairs, T(0));
    for(const auto& e : aabb_tmp)
        context.aabb_val_2d[static_cast<std::size_t>(e.ra) * num_b_pairs + e.rb] = e.val;

    context.alpha_single.assign(num_alpha, {});
    context.alpha_double.assign(num_alpha, {});
    context.beta_single.assign(num_beta, {});
    context.beta_double.assign(num_beta, {});

    // parity of set bits strictly between p0 and p1 (i.e. in [p0+1, p1)).
    auto pair_sign = [&](const Key& key, width_t p0, width_t p1) -> int {
        return Policy::range_parity(key, static_cast<width_t>(p0 + 1), p1) ? -1 : 1;
    };

    auto build_sector = [&](std::size_t n,
                            const std::vector<Key>& keys,
                            width_t nbits,
                            typename Policy::template Map<std::uint32_t>& idxmap,
                            emhash8::HashMap<std::uint32_t, std::size_t>& single_map,
                            emhash8::HashMap<std::uint64_t, std::size_t>& double_map,
                            std::unordered_map<std::uint32_t, int>& pair_id,
                            std::vector<std::vector<HalfConnSingle>>& singles_out,
                            std::vector<std::vector<HalfConnDouble>>& doubles_out) {

#pragma omp parallel if(n > 256)
        {
            std::vector<width_t> occ(nbits), emp(nbits);
#pragma omp for schedule(dynamic)
            for(std::size_t s = 0; s < n; ++s)
            {
                const Key kv = keys[s];
                unsigned no = 0, ne = 0;
                for(width_t p = 0; p < nbits; ++p)
                {
                    if(Policy::test(kv, p))
                        occ[no++] = p;
                    else
                        emp[ne++] = p;
                }
                auto& singles = singles_out[s];
                auto& doubles = doubles_out[s];
                for(unsigned oi = 0; oi < no; ++oi)
                    for(unsigned ej = 0; ej < ne; ++ej)
                    {
                        const width_t i = occ[oi], j = emp[ej];
                        Key Kp = kv;
                        Policy::flip(Kp, i);
                        Policy::flip(Kp, j);
                        auto it = idxmap.find(Kp);
                        if(it == idxmap.end())
                            continue;
                        const width_t p0 = i < j ? i : j;
                        const width_t p1 = i < j ? j : i;
                        const std::uint32_t pk = pack2(p0, p1);
                        std::size_t g_same = NO_HALF_G;
                        auto gi = single_map.find(pk);
                        if(gi != single_map.end())
                            g_same = gi->second;
                        int rank = -1;
                        auto ri = pair_id.find(pk);
                        if(ri != pair_id.end())
                            rank = ri->second;
                        singles.push_back({it->second, g_same, rank, pair_sign(kv, p0, p1)});
                    }
                for(unsigned oi = 0; oi < no; ++oi)
                    for(unsigned oj = oi + 1; oj < no; ++oj)
                        for(unsigned ei = 0; ei < ne; ++ei)
                            for(unsigned ej = ei + 1; ej < ne; ++ej)
                            {
                                const width_t i1 = occ[oi], i2 = occ[oj], j1 = emp[ei],
                                              j2 = emp[ej];
                                Key Kp = kv;
                                Policy::flip(Kp, i1);
                                Policy::flip(Kp, i2);
                                Policy::flip(Kp, j1);
                                Policy::flip(Kp, j2);
                                auto it = idxmap.find(Kp);
                                if(it == idxmap.end())
                                    continue;
                                width_t sd[4] = {i1, i2, j1, j2};
                                std::sort(sd, sd + 4);
                                auto gi = double_map.find(pack4(sd[0], sd[1], sd[2], sd[3]));
                                if(gi == double_map.end())
                                    continue;
                                doubles.push_back({it->second, gi->second});
                            }
            }
        }
    };

    build_sector(num_alpha,
                 akeys,
                 hw,
                 amap,
                 aa_map,
                 aaaa_map,
                 a_pair_id,
                 context.alpha_single,
                 context.alpha_double);
    build_sector(num_beta,
                 bkeys,
                 bw,
                 bmap,
                 bb_map,
                 bbbb_map,
                 b_pair_id,
                 context.beta_single,
                 context.beta_double);

    context.num_alpha = num_alpha;
    context.num_beta = num_beta;
    context.num_a_pairs = num_a_pairs;
    context.num_b_pairs = num_b_pairs;
    context.BLK = 128;
    if(const char* e = std::getenv("FQ_BLK"))
    {
        long v = std::atol(e);
        if(v > 0)
            context.BLK = static_cast<std::size_t>(v);
    }
    context.rsb_w = width;
    context.num_blocks = (subspace_dim + context.BLK - 1) / context.BLK;
    context.usable = true;

    if(std::getenv("FQ_HALFSTR_VERBOSE"))
    {
        std::size_t asc = 0, adc = 0, bsc = 0, bdc = 0;
        for(const auto& v : context.alpha_single)
            asc += v.size();
        for(const auto& v : context.alpha_double)
            adc += v.size();
        for(const auto& v : context.beta_single)
            bsc += v.size();
        for(const auto& v : context.beta_double)
            bdc += v.size();
        char ktag[16];
        if(Policy::words)
            std::snprintf(ktag, sizeof(ktag), "%zu", Policy::words);
        else
            std::snprintf(ktag, sizeof(ktag), "dyn");
        std::fprintf(stderr,
                     "[halfstr] ACTIVE (K=%s): num_alpha=%zu num_beta=%zu a_pairs=%zu b_pairs=%zu"
                     " | conns alpha_single=%zu alpha_double=%zu beta_single=%zu beta_double=%zu\n",
                     ktag,
                     num_alpha,
                     num_beta,
                     num_a_pairs,
                     num_b_pairs,
                     asc,
                     adc,
                     bsc,
                     bdc);
    }
}

// Templated on half_width: pick the smallest key representation that holds a half
// string, then build with it. All tiers run the same connected-det algorithm
// (build_halfstr_context_impl).
//   FixedKeyOps<1>: <=64 orbitals/spin-sector  (full det <=128 bits)
//   FixedKeyOps<2>: 65-128                     (<=256 bits)
//   FixedKeyOps<4>: 129-256                    (<=512 bits)
//   DynKeyOps     : >256                       (arbitrary; dynamic_bitset)
// The only remaining fallback to the group-walk omp_matvec2 is a non-Cartesian subspace
// (build then leaves usable=false). FQ_HALFSTR_FORCE_K={1,2,4} forces a wider fixed
// tier instead of auto-picking; FQ_HALFSTR_FORCE_K=0 forces the dynamic tier.
template <typename T>
void build_halfstr_context(const std::vector<OperatorTerm_t>& terms,
                           const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                           const width_t width,
                           const std::size_t subspace_dim,
                           const std::size_t* __restrict group_ptrs,
                           const std::size_t* __restrict group_ladder_ptrs,
                           const std::vector<std::vector<width_t>>& group_offdiag_inds,
                           const unsigned int num_groups,
                           const unsigned int ladder_offset,
                           HalfStrContext<T>& context)
{
    context = HalfStrContext<T>{};
    const width_t hw = static_cast<width_t>(width / 2);
    const width_t bw = static_cast<width_t>(width - hw);
    context.width = width;
    context.half_width = hw;

    if(subspace_dim == 0 || hw == 0 || terms.size() > UINT32_MAX)
        return;

    const std::size_t half_bits = static_cast<std::size_t>(hw > bw ? hw : bw);
    const std::size_t words = (half_bits + 63) / 64;

    std::size_t K = (words <= 1) ? 1 : (words <= 2) ? 2 : (words <= 4) ? 4 : 0;

    if(const char* fk = std::getenv("FQ_HALFSTR_FORCE_K"))
    {
        const long v = std::atol(fk);
        if(v == 0)
            K = 0;
        else if((v == 1 || v == 2 || v == 4) && static_cast<std::size_t>(v) >= words)
            K = static_cast<std::size_t>(v);
    }
    switch(K)
    {
    case 1:
        build_halfstr_context_impl<T, FixedKeyOps<1>>(terms,
                                                      subspace,
                                                      width,
                                                      subspace_dim,
                                                      group_ptrs,
                                                      group_ladder_ptrs,
                                                      group_offdiag_inds,
                                                      num_groups,
                                                      ladder_offset,
                                                      context);
        break;
    case 2:
        build_halfstr_context_impl<T, FixedKeyOps<2>>(terms,
                                                      subspace,
                                                      width,
                                                      subspace_dim,
                                                      group_ptrs,
                                                      group_ladder_ptrs,
                                                      group_offdiag_inds,
                                                      num_groups,
                                                      ladder_offset,
                                                      context);
        break;
    case 4:
        build_halfstr_context_impl<T, FixedKeyOps<4>>(terms,
                                                      subspace,
                                                      width,
                                                      subspace_dim,
                                                      group_ptrs,
                                                      group_ladder_ptrs,
                                                      group_offdiag_inds,
                                                      num_groups,
                                                      ladder_offset,
                                                      context);
        break;
    default:
        // >256 orbitals/sector (or forced): arbitrary-width dynamic_bitset tier.
        build_halfstr_context_impl<T, DynKeyOps>(terms,
                                                 subspace,
                                                 width,
                                                 subspace_dim,
                                                 group_ptrs,
                                                 group_ladder_ptrs,
                                                 group_offdiag_inds,
                                                 num_groups,
                                                 ladder_offset,
                                                 context);
        break;
    }
}

template <typename T, typename Sink>
void halfstr_walk(const HalfStrContext<T>& context,
                  const std::vector<OperatorTerm_t>& terms,
                  const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                  const std::size_t subspace_dim,
                  const width_t* __restrict group_rowint_length,
                  const unsigned int ladder_offset,
                  Sink sink)
{
    const auto* bitsets = subspace.get_bitsets();
    const std::size_t num_alpha = context.num_alpha;
    const std::size_t num_b_pairs = context.num_b_pairs;
    const std::size_t BLK = context.BLK;
    const std::size_t rsb_w = context.rsb_w;
    const std::size_t num_blocks = context.num_blocks;
    const width_t* __restrict flat_inds = context.flat_inds.data();
    const std::size_t* __restrict inds_offsets = context.inds_offsets.data();
    const std::uint32_t* __restrict ladder32 = context.ladder32.data();
    const T* __restrict aabb_val_2d = context.aabb_val_2d.data();

    auto gview = [&](std::size_t g) -> GroupIndsView {
        const std::size_t off = inds_offsets[g];
        return GroupIndsView{flat_inds + off, inds_offsets[g + 1] - off};
    };

#pragma omp parallel if(subspace_dim > 4096)
    {
        std::vector<uint8_t> rsb_buf;
        boost::dynamic_bitset<std::size_t> col_vec;

        auto eval_same_sector = [&](std::size_t g,
                                    const boost::dynamic_bitset<std::size_t>& row,
                                    const uint8_t* row_set_bits,
                                    std::size_t col_idx,
                                    std::size_t out_idx) {
            if(g == NO_HALF_G)
                return;
            const GroupIndsView group_inds = gview(g);
            const unsigned int row_int =
                bitset_ladder_int(row_set_bits, group_inds.data(), group_rowint_length[g]);
            const std::size_t s = static_cast<std::size_t>(ladder32[g * ladder_offset + row_int]);
            const std::size_t e =
                static_cast<std::size_t>(ladder32[g * ladder_offset + row_int + 1]);
            if(s >= e)
                return;
            col_vec = row;
            flip_bits(col_vec, group_inds.data(), group_inds.size());
            T val = 0;
            for(std::size_t idx = s; idx < e; ++idx)
            {
                const OperatorTerm_t* term = &terms[idx];
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
                sink(out_idx, col_idx, val);
        };

#pragma omp for schedule(dynamic)
        for(std::size_t blk = 0; blk < num_blocks; ++blk)
        {
            const std::size_t r0 = blk * BLK;
            const std::size_t r1 = std::min(r0 + BLK, subspace_dim);
            const std::size_t bn = r1 - r0;

            rsb_buf.assign(bn * rsb_w, 0);
            for(std::size_t rib = 0; rib < bn; ++rib)
            {
                const auto& row = bitsets[r0 + rib].first;
                uint8_t* dst = rsb_buf.data() + rib * rsb_w;
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

            for(std::size_t rib = 0; rib < bn; ++rib)
            {
                const std::size_t idx = r0 + rib;
                const std::size_t b = idx / num_alpha;
                const std::size_t a = idx % num_alpha;
                const auto& row = bitsets[idx].first;
                const uint8_t* rsb = rsb_buf.data() + rib * rsb_w;
                const std::size_t beta_base = b * num_alpha;

                const auto& a_singles = context.alpha_single[a];
                const auto& a_doubles = context.alpha_double[a];
                const auto& b_singles = context.beta_single[b];
                const auto& b_doubles = context.beta_double[b];

                // aa: same beta, alpha neighbor -> col = beta_base + a_col
                for(const auto& c : a_singles)
                    eval_same_sector(c.g, row, rsb, beta_base + c.col, idx);
                // aaaa
                for(const auto& c : a_doubles)
                    eval_same_sector(c.g, row, rsb, beta_base + c.col, idx);
                // bb: same alpha, beta neighbor -> col = b_col*num_alpha + a
                for(const auto& c : b_singles)
                    eval_same_sector(c.g, row, rsb, c.col * num_alpha + a, idx);
                // bbbb
                for(const auto& c : b_doubles)
                    eval_same_sector(c.g, row, rsb, c.col * num_alpha + a, idx);

                // aabb: alpha-single x beta-single, direct coeff * sign_a * sign_b
                for(const auto& ca : a_singles)
                {
                    if(ca.pair_rank < 0)
                        continue;
                    const std::size_t base = static_cast<std::size_t>(ca.pair_rank) * num_b_pairs;
                    const int sa = ca.sign;
                    const std::size_t col_a = ca.col;
                    for(const auto& cb : b_singles)
                    {
                        if(cb.pair_rank < 0)
                            continue;
                        const T coeff = aabb_val_2d[base + static_cast<std::size_t>(cb.pair_rank)];
                        if(std::abs(coeff) <= ATOL)
                            continue;
                        sink(idx, cb.col * num_alpha + col_a, coeff * static_cast<T>(sa * cb.sign));
                    }
                }
            }
        }
    }
}

// Special path for half-strs mode
template <typename T>
void omp_matvec2_halfstr(const HalfStrContext<T>& context,
                         const std::vector<OperatorTerm_t>& terms,
                         const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                         const T* __restrict diag_vec,
                         const std::size_t subspace_dim,
                         const int has_nonzero_diag,
                         const width_t* __restrict group_rowint_length,
                         const unsigned int ladder_offset,
                         const T* __restrict in_vec,
                         T* __restrict out_vec)
{
    if(has_nonzero_diag)
    {
#pragma omp parallel for
        for(std::size_t kk = 0; kk < subspace_dim; ++kk)
            out_vec[kk] = diag_vec[kk] * in_vec[kk];
    }
    halfstr_walk<T>(context,
                    terms,
                    subspace,
                    subspace_dim,
                    group_rowint_length,
                    ladder_offset,
                    [&](std::size_t out_row, std::size_t col_idx, T val) {
                        out_vec[out_row] += val * in_vec[col_idx];
                    });
}
