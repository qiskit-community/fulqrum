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
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
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

// T is the data type, U is in the index type, e.g (complex, int)
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
    
    auto start = std::chrono::steady_clock::now();
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
        const std::vector<unsigned int>* group_inds;
        std::size_t* col_ptr;
        std::size_t col_idx;
        T val;
        unsigned int row_int;

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

            group_inds = &group_offdiag_inds[group];
            row_int = bitset_ladder_int(
                row_set_bits.data(), group_inds->data(), group_rowint_length[group]);
            group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
            group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];
            val = 0;

            if(group_int_start < group_int_stop)
            {
                col_vec = row;
                flip_bits(col_vec, group_inds->data(), group_inds->size());

                col_ptr = subspace.get_ptr(col_vec);
                if(col_ptr == nullptr)
                {
                    continue;
                } // column is NOT in the subspace so break group
                col_idx = *col_ptr;
            }

            for(idx = group_int_start; idx < group_int_stop; idx++)
            { // begin loop over terms in this group
                term = &terms[idx];
                if(passes_proj_validation(term, row))
                {
                    accum_element(row,
                                  col_vec,
                                  &term->indices[0],
                                  &term->values[0],
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
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Vector build time: " << duration.count() << " milliseconds" << std::endl;

    auto start3 = std::chrono::steady_clock::now();

    sort_paired(cols, data);
    
    auto end3 = std::chrono::steady_clock::now();
    auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3);

    std::cout << "Pair sort time: " << duration3.count() << " milliseconds" << std::endl;

// #pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
//     for(kk = 0; kk < subspace_dim; kk++)
//     {
//         // need two different types for sorting
//         int sort_start_int = 0;
//         int sort_end_int = 0;
//         long long sort_start_long = 0;
//         long long sort_end_long = 0;
//         // sort column indices and data in each row
//         if constexpr(std::is_same_v<U, int>)
//         {
//             sort_end_int = cols[kk].size() - 1;
//             quicksort_indices_data(cols[kk].data(), data[kk].data(), sort_start_int, sort_end_int);
//         }
//         else
//         {
//             sort_end_long = cols[kk].size() - 1;
//             quicksort_indices_data(
//                 cols[kk].data(), data[kk].data(), sort_start_long, sort_end_long);
//         }
//         cols[kk].shrink_to_fit();
//         data[kk].shrink_to_fit();
//     } // end loop over all rows
} // end function
