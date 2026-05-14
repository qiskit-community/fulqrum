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
    std::vector<std::size_t> other_groups;  // Groups that don't fit the 5 categories
    
    const width_t half_width = width / 2;
    for(std::size_t g = 0; g < num_groups; g++)
    {
        const auto& inds = group_offdiag_inds[g];
        const std::size_t ind_size = inds.size();
        
        if(ind_size == 2)
        {
            // Check if both indices are in lower half (aa) or upper half (bb)
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
            // Check if all 4 indices are in lower half (aaaa), upper half (bbbb), or split (aabb)
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
                // Also add to other_groups for standard processing until efficient method is debugged
                other_groups.push_back(g);
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

    // Validate hypothesis for aabb groups and compute grp4_direct coefficients
    std::vector<T> grp4_direct(num_groups, 0);
    
    for(const auto& g : aabb_groups)
    {
        std::size_t group_start = group_ptrs[g];
        std::size_t group_stop = group_ptrs[g + 1];
        
        if(group_start >= group_stop)
        {
            continue; // Empty group
        }
        
        // Get first term in group to extract coefficient and real_phase
        const OperatorTerm_t& first_term = terms[group_start];
        std::complex<double> group_coeff = first_term.coeff;
        int group_real_phase = first_term.real_phase;
        
        // Validate hypothesis: all terms in group have same coeff and real_phase = +1
        for(std::size_t idx = group_start; idx < group_stop; idx++)
        {
            const OperatorTerm_t& term = terms[idx];
            
            if(term.real_phase != 1)
            {
                throw std::runtime_error(
                    "Hypothesis failed: aabb group " + std::to_string(g) +
                    " has term with real_phase = " + std::to_string(term.real_phase) +
                    " (expected +1)");
            }
            
            if(std::abs(term.coeff - group_coeff) > 1e-14)
            {
                throw std::runtime_error(
                    "Hypothesis failed: aabb group " + std::to_string(g) +
                    " has terms with different coefficients");
            }
        }
        
        // Compute grp4_direct[group] = coeff × real_phase
        if constexpr(std::is_same_v<T, double>)
        {
            grp4_direct[g] = group_real_phase * group_coeff.real();
        }
        else
        {
            grp4_direct[g] = group_coeff;
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
        std::size_t group;
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

        // Variables to collect valid aa and bb groups
        std::vector<std::size_t> valid_aa;
        std::vector<std::size_t> valid_bb;

        // Process aa groups
        for(const auto& group : aa_groups)
        {
            group_inds = &group_offdiag_inds[group];
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
            valid_aa.push_back(group);

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
        }

        // Process aaaa groups
        for(const auto& group : aaaa_groups)
        {
            group_inds = &group_offdiag_inds[group];
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
        }

        // Process bb groups
        for(const auto& group : bb_groups)
        {
            group_inds = &group_offdiag_inds[group];
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
            valid_bb.push_back(group);

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
        }

        // TODO: Implement efficient aabb processing
        // For now, aabb groups are processed via other_groups using standard accum_element

        // Process bbbb groups
        for(const auto& group : bbbb_groups)
        {
            group_inds = &group_offdiag_inds[group];
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
        }

        // Process other groups (those that don't fit the 5 specific categories)
        for(const auto& group : other_groups)
        {
            group_inds = &group_offdiag_inds[group];
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
        }
    }

    sort_paired(cols, data);
} // end function
