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
#include <mutex>
#include <vector>

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include "operators.hpp"
#include <boost/dynamic_bitset.hpp>

template <typename T, typename U>
void csrlike_builder(const OperatorTerm_t* terms,
                     const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                     const T* __restrict diag_vec,
                     const unsigned int width,
                     const std::size_t subspace_dim,
                     const int has_nonzero_diag,
                     const std::size_t* __restrict group_ptrs,
                     const std::vector<std::vector<unsigned int>>& group_offdiag_inds,
                     const std::size_t num_groups,
                     std::vector<std::vector<U>>& cols,
                     std::vector<std::vector<T>>& data)
{
    std::size_t kk;

    const auto* bitsets = subspace.get_bitsets();

    cols.resize(subspace_dim);
    data.resize(subspace_dim);

    std::vector<std::mutex> mutex1(subspace_dim);

    std::vector<uint16_t> grp_max_inds(num_groups, width);
    get_group_max_inds(grp_max_inds, group_offdiag_inds, num_groups);

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

#pragma omp parallel for if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    { // begin loop over all rows
        // define variables locally for omp for loop
        std::size_t idx, col_idx;
        std::size_t group_start, group_stop, group;
        const OperatorTerm_t* term;
        const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;
        boost::dynamic_bitset<std::size_t> col_vec;
        std::size_t* col_ptr;
        const std::vector<unsigned int>* group_inds;
        T val;

        std::vector<uint8_t> row_set_bits(row.size(), 0);
        bitset_to_bitvec(row, row_set_bits);

        for(group = 0; group < num_groups; group++)
        { // begin loop over groups
            // Detects a lower or an upper
            // triangle matrix element.
            // See details in ``get_group_max_inds()``
            // in fulqrum/core/src/offdiag_grouping.hpp
            if(!row_set_bits[grp_max_inds[group]])
            {
                continue;
            }

            group_start = group_ptrs[group];
            group_stop = group_ptrs[group + 1];
            val = 0;

            if(group_start < group_stop)
            {
                group_inds = &group_offdiag_inds[group];
                col_vec = row;
                flip_bits(col_vec, group_inds->data(), group_inds->size());

                col_ptr = subspace.get_ptr(col_vec);
                if(col_ptr == nullptr)
                {
                    continue;
                } // column is NOT in the subspace so break group
                col_idx = *col_ptr;
            }

            for(idx = group_start; idx < group_stop; idx++)
            { // begin loop over terms in this group

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
            } // end loop over terms in this group
            if(std::abs(val) > ATOL)
            {
                // see fulqrum/core/src/csr.hpp for details
                // about these Mutex locks
                {
                    std::lock_guard<std::mutex> lock_kk(mutex1[kk]);
                    cols[kk].push_back(col_idx);
                    data[kk].push_back(val);
                }

                {
                    std::lock_guard<std::mutex> lock_col_idx(mutex1[col_idx]);
                    cols[col_idx].push_back(kk);
                    if constexpr(std::is_same_v<T, double>)
                    {
                        data[col_idx].push_back(val);
                    }
                    else
                    {
                        // for complex-valued matrix, the upper triangle
                        // element will be complex conjugate of the lower
                        // triangle element
                        data[col_idx].push_back(std::conj(val));
                    }
                }
            }
        } // end loop over groups
    } // end loop over all rows

    sort_paired(cols, data);
}
