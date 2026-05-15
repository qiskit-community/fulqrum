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
#include <complex>
#include <cstdlib>
#include <vector>

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "csr_utils.hpp"
#include "csrlike.hpp"
#include "elements.hpp"
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
        const auto& inds = group_offdiag_inds[g];
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
        T val;
        unsigned int row_int;

        std::vector<uint8_t> row_set_bits(row.size(), 0);
        bitset_to_bitvec(row, row_set_bits);

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

        // Standard per-group path: ladder lookup -> col flip -> subspace lookup
        // -> term loop -> emit.  Shared by aa, aaaa, bb, bbbb, aabb_slow (and
        // the other_groups).
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
                cols[kk].push_back(col_idx);
                data[kk].push_back(val);
            }
        };

        for(const auto& g : aa_groups)
            process_standard_group(g);
        for(const auto& g : aaaa_groups)
            process_standard_group(g);
        for(const auto& g : bb_groups)
            process_standard_group(g);

        // Process aabb fast groups (size==4, 2 alpha + 2 beta indices, every term
        // in the group shares a single coeff and real_phase).  Matrix element
        // = coeff * asign * bsign; skip the term loop.
        // asign / bsign are the Jordan-Wigner parities for the alpha and beta
        // excitations.
        for(const auto& group : aabb_fast_groups)
        {
            group_inds = &group_offdiag_inds[group];
            const width_t ap0 = (*group_inds)[0];
            const width_t ap1 = (*group_inds)[1];
            const width_t bp0 = (*group_inds)[2];
            const width_t bp1 = (*group_inds)[3];

            if((row_set_bits[ap0] == row_set_bits[ap1]) || (row_set_bits[bp0] == row_set_bits[bp1]))
            {
                continue;
            }

            row_int = bitset_ladder_int(
                row_set_bits.data(), group_inds->data(), group_rowint_length[group]);
            group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
            group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];

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

            const std::size_t aabb_parity = range_parity(ap0 + 1, ap1) ^ range_parity(bp0 + 1, bp1);
            double sign = aabb_parity ? -1.0 : 1.0;

            val = aabb_direct[group] * static_cast<T>(sign);

            if(std::abs(val) > ATOL)
            {
                cols[kk].push_back(col_idx);
                data[kk].push_back(val);
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
    }

    sort_paired(cols, data);
} // end function
