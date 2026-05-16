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
#include <boost/dynamic_bitset.hpp>

// Builds the CSR structure for a subspace Hamiltonian.
//
// Group handling mirrors csrlike_builder2: groups are bucketed
// (aa / aaaa / bb / bbbb / aabb / other) and aabb groups whose
// terms share a single coeff + real_phase use the direct
// asign*bsign formula instead of the per-term accum_element loop.
// other and aabb_slow groups are here for safety.
// They are supposed to be empty.
//
// Called twice: compute_values == 0 counts entries per row and fills indptr;
// compute_values == 1 fills indices/data using the indptr from the first call.
//
// T is the index type, U is the value type, e.g. (int, complex).
template <typename T, typename U>
void csr_matrix_builder2(const std::vector<OperatorTerm_t>& terms,
                         const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                         const U* __restrict diag_vec,
                         const width_t width,
                         const std::size_t subspace_dim,
                         const int has_nonzero_diag,
                         const std::size_t* __restrict group_ptrs,
                         const std::size_t* __restrict group_ladder_ptrs,
                         const width_t* __restrict group_rowint_length,
                         const std::vector<std::vector<width_t>>& group_offdiag_inds,
                         const std::size_t num_groups,
                         const unsigned int ladder_offset,
                         T* __restrict indptr,
                         T* __restrict indices,
                         U* __restrict data,
                         const int compute_values)
{
    std::size_t kk;
    T temp, _sum;

    const auto* bitsets = subspace.get_bitsets();

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
        const auto& inds = group_offdiag_inds[g];
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
    // through accum_element per term.  In csr2 the value type is U.
    std::vector<U> aabb_direct(num_groups, 0);
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
            if constexpr(std::is_same_v<U, double>)
            {
                aabb_direct[g] = group_real_phase * group_coeff.real();
            }
            else
            {
                aabb_direct[g] = static_cast<U>(group_real_phase) * group_coeff;
            }
            aabb_fast_groups.push_back(g);
        }
        else
        {
            aabb_slow_groups.push_back(g);
        }
    }

    // ---- aabb fast-path prefilter setup -------------------------------------
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
            const auto& inds = group_offdiag_inds[g];
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

    // process diagonal
    std::vector<T> row_nnz_s(subspace_dim, 0);
    if(has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
        for(kk = 0; kk < subspace_dim; kk++)
        { // begin loop over all rows
            T& row_nnz = row_nnz_s[kk]; // reference T& is critical
            T& elem_start = indptr[kk];
            // do diagonal first, if any
            if(diag_vec[kk] != 0.0)
            {
                if(compute_values)
                {
                    indices[elem_start + row_nnz] = kk;
                    data[elem_start + row_nnz] = diag_vec[kk];
                }
                row_nnz += 1;
            }
        }
    }

#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    { // begin loop over all rows
        const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;

        // define variables locally for omp for loop
        std::size_t idx;
        std::size_t group_int_start, group_int_stop;
        const OperatorTerm_t* term;
        boost::dynamic_bitset<std::size_t> col_vec;
        const std::vector<width_t>* group_inds;
        std::size_t* col_ptr;
        std::size_t col_idx;
        U val;
        unsigned int row_int;

        T& row_nnz = row_nnz_s[kk];
        T& elem_start = indptr[kk];

        std::vector<uint8_t> row_set_bits(row.size(), 0);
        bitset_to_bitvec(row, row_set_bits);

        // Per-row aabb prefilter (distinct alpha/beta pair validity).
        std::vector<char> alpha_ok(num_a_pairs);
        std::vector<char> beta_ok(num_b_pairs);

        auto range_parity = [&](width_t lo, width_t hi) -> std::size_t {
            if(lo >= hi)
            {
                return 0;
            }
            const width_t last = hi - 1; // inclusive last bit
            const std::size_t lo_blk = lo >> BLOCK_EXPONENT;
            const std::size_t hi_blk = static_cast<std::size_t>(last) >> BLOCK_EXPONENT;
            const std::size_t lo_mask = ~std::size_t(0) << (lo & BLOCK_SHIFT);
            const std::size_t hi_mask = ~std::size_t(0) >> (BLOCK_SHIFT - (last & BLOCK_SHIFT));

            std::size_t acc;
            if(lo_blk == hi_blk)
            {
                acc = row.m_bits[lo_blk] & lo_mask & hi_mask;
            }
            else
            {
                acc = row.m_bits[lo_blk] & lo_mask;
                for(std::size_t b = lo_blk + 1; b < hi_blk; b++)
                {
                    acc ^= row.m_bits[b];
                }
                acc ^= row.m_bits[hi_blk] & hi_mask;
            }
            return static_cast<std::size_t>(
                __builtin_parityll(static_cast<unsigned long long>(acc)));
        };

        // Emit a single (kk, col_idx, v) entry for this row.  No transpose
        // write, so no mutex is needed: each row is owned by one thread.
        auto emit = [&](std::size_t cidx, const U& v) {
            if(compute_values)
            {
                indices[elem_start + row_nnz] = cidx;
                data[elem_start + row_nnz] = v;
            }
            row_nnz += 1;
        };

        // Standard per-group path: validity check -> ladder lookup -> col flip
        // -> subspace lookup -> term loop -> emit.  Shared by aa, aaaa, bb,
        // bbbb, aabb_slow and other_groups.
        auto process_standard_group = [&](std::size_t g) {
            group_inds = &group_offdiag_inds[g];

            // Hamming weight check.
            // Flip pos must have equal number of 1s and 0s.
            const width_t _p = (*group_inds)[0];
            const width_t _q = (*group_inds)[1];

            if(group_inds->size() == 2)
            {
                if(row_set_bits[_p] == row_set_bits[_q])
                {
                    return;
                }
            }
            else if(group_inds->size() == 4)
            {
                const width_t _r = (*group_inds)[2];
                const width_t _s = (*group_inds)[3];
                if(row_set_bits[_p] + row_set_bits[_q] + row_set_bits[_r] + row_set_bits[_s] != 2)
                {
                    return;
                }
            }

            row_int =
                bitset_ladder_int(row_set_bits.data(), group_inds->data(), group_rowint_length[g]);
            group_int_start = group_ladder_ptrs[g * ladder_offset + row_int];
            group_int_stop = group_ladder_ptrs[g * ladder_offset + row_int + 1];

            if(group_int_start >= group_int_stop)
            {
                return;
            }

            col_vec = row;
            flip_bits(col_vec, group_inds->data(), group_inds->size());

            col_ptr = subspace.get_ptr(col_vec);
            if(col_ptr == nullptr)
            {
                return;
            }
            col_idx = *col_ptr;

            val = 0;
            for(idx = group_int_start; idx < group_int_stop; idx++)
            {
                term = &terms[idx];
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
                emit(col_idx, val);
            }
        };

        for(const auto& g : aa_groups)
            process_standard_group(g);
        for(const auto& g : aaaa_groups)
            process_standard_group(g);
        for(const auto& g : bb_groups)
            process_standard_group(g);

        for(std::size_t i = 0; i < num_a_pairs; ++i)
        {
            const width_t p0 = a_pairs[i][0];
            const width_t p1 = a_pairs[i][1];
            if(row_set_bits[p0] == row_set_bits[p1])
            {
                alpha_ok[i] = 0;
                continue;
            }
            alpha_ok[i] = alpha_half_hashes.contains(region_hash(row, 0, half_width, p0, p1));
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
            beta_ok[j] = beta_half_hashes.contains(region_hash(row, half_width, width, p0, p1));
        }

        for(std::size_t i = 0; i < num_a_pairs; ++i)
        {
            if(!alpha_ok[i])
            {
                continue;
            }
            const width_t ap0 = a_pairs[i][0];
            const width_t ap1 = a_pairs[i][1];
            for(const auto& e : aabb_by_alpha[i])
            {
                if(!beta_ok[e.b_id])
                {
                    continue;
                }
                group_inds = &group_offdiag_inds[e.g];
                const width_t bp0 = (*group_inds)[2];
                const width_t bp1 = (*group_inds)[3];

                row_int = bitset_ladder_int(
                    row_set_bits.data(), group_inds->data(), group_rowint_length[e.g]);
                group_int_start = group_ladder_ptrs[e.g * ladder_offset + row_int];
                group_int_stop = group_ladder_ptrs[e.g * ladder_offset + row_int + 1];
                if(group_int_start >= group_int_stop)
                {
                    continue;
                }

                col_vec = row;
                flip_bits(col_vec, group_inds->data(), group_inds->size());
                col_ptr = subspace.get_ptr(col_vec);
                if(col_ptr == nullptr)
                {
                    continue;
                }
                col_idx = *col_ptr;

                const std::size_t aabb_parity =
                    range_parity(ap0 + 1, ap1) ^ range_parity(bp0 + 1, bp1);
                double sign = aabb_parity ? -1.0 : 1.0;

                val = aabb_direct[e.g] * static_cast<U>(sign);

                if(std::abs(val) > ATOL)
                {
                    emit(col_idx, val);
                }
            }
        }

        // aabb slow path: terms within a group have differing coeff or
        // real_phase, so the direct asign*bsign formula doesn't apply.
        for(const auto& g : aabb_slow_groups)
            process_standard_group(g);

        for(const auto& g : bbbb_groups)
            process_standard_group(g);

        // other_groups is supposed to be empty. Kept here for safety.
        for(const auto& g : other_groups)
            process_standard_group(g);
    } // end loop over all rows

    if(!compute_values) // Done with all rows so accumulate for correct indptr structure
    {
        _sum = 0;
        for(kk = 0; kk < (subspace_dim); kk++)
        {
            temp = _sum + row_nnz_s[kk];
            indptr[kk] = _sum;
            _sum = temp;
        }
        indptr[subspace_dim] = _sum;
    }
}
