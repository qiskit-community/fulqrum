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
#include <boost/dynamic_bitset.hpp>

template <typename T, typename U>
void csr_matrix_builder2(const std::vector<OperatorTerm_t>& terms,
                         const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                         const U* __restrict diag_vec,
                         const unsigned int width,
                         const std::size_t subspace_dim,
                         const int has_nonzero_diag,
                         const std::size_t* __restrict group_ptrs,
                         const std::size_t* __restrict group_ladder_ptrs,
                         const unsigned int* __restrict group_rowint_length,
                         const std::vector<std::vector<unsigned int>>& group_offdiag_inds,
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

    std::vector<std::mutex> mutex1(subspace_dim);
    std::vector<std::mutex> mutex2(subspace_dim);

    std::vector<uint16_t> grp_max_inds(num_groups, width);
    get_group_max_inds(grp_max_inds, group_offdiag_inds, num_groups);

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
        std::size_t group;
        std::size_t group_int_start, group_int_stop;
        T row_nnz_col_idx, elem_start_col_idx;
        const OperatorTerm_t* term;
        boost::dynamic_bitset<std::size_t> col_vec;
        const std::vector<unsigned int>* group_inds;
        std::size_t* col_ptr;
        std::size_t col_idx;
        U val;
        unsigned int row_int;

        T& row_nnz = row_nnz_s[kk];
        T& elem_start = indptr[kk];

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

            val = 0;

            for(idx = group_int_start; idx < group_int_stop; idx++)
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
                if(compute_values)
                {
                    // see fulqrum/core/src/csr.hpp for details
                    // about these Mutex locks
                    {
                        std::lock_guard<std::mutex> lock_kk(mutex1[kk]);

                        indices[elem_start + row_nnz] = col_idx;
                        data[elem_start + row_nnz] = val;
                        row_nnz += 1;
                    }

                    {
                        std::lock_guard<std::mutex> lock_col_idx(mutex1[col_idx]);
                        row_nnz_col_idx = row_nnz_s[col_idx];
                        elem_start_col_idx = indptr[col_idx];
                        indices[elem_start_col_idx + row_nnz_col_idx] = kk;
                        row_nnz_s[col_idx] += 1;

                        if constexpr(std::is_same_v<U, double>)
                        {
                            data[elem_start_col_idx + row_nnz_col_idx] = val;
                        }
                        else
                        {
                            // for complex-valued matrix, the upper triangle
                            // element will be complex conjugate of the lower
                            // triangle element
                            data[elem_start_col_idx + row_nnz_col_idx] = std::conj(val);
                        }
                    }
                }
                if(!compute_values)
                {
                    {
                        std::lock_guard<std::mutex> lock1(mutex2[kk]);
                        row_nnz += 1;
                    }

                    {
                        std::lock_guard<std::mutex> lock2(mutex2[col_idx]);
                        row_nnz_s[col_idx] += 1;
                    }
                }
            }
        } // end loop over groups
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
