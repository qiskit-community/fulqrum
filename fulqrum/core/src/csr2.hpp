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
#include <cstddef>
#include <vector>

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "constants.hpp"
#include "halfstr_tables.hpp"
#include "type2_common.hpp"
#include "type2_visit_cartesian.hpp"
#include "type2_visit_groups.hpp"
#include "type2_visit_non_cartesian.hpp"
#ifdef _OPENMP
#    include <omp.h>
#endif

// Builds the CSR structure for a type-2 subspace Hamiltonian.
//
// The element visiting is shared with matvec2.hpp and csrlike_builder2.hpp, all
// three differ only in what they do with each (row, col, value). This file supplies the
// CSR sink, which is called twice:
//   compute_values == 0 -> count entries per row (fills indptr via a prefix sum)
//   compute_values == 1 -> write indices/data using the indptr from the first call
//
// T is the index type, U is the value type, e.g. (int, complex).

// Diagonal entries, written directly (they are not produced by the visitors).
template <typename T, typename U>
inline void csr2_diag(const U* __restrict diag_vec,
                      const std::size_t subspace_dim,
                      const int compute_values,
                      T* __restrict indptr,
                      T* __restrict indices,
                      U* __restrict data,
                      std::vector<T>& row_nnz_s)
{
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
    for(std::size_t kk = 0; kk < subspace_dim; kk++)
    {
        T& row_nnz = row_nnz_s[kk];
        if(diag_vec[kk] != 0.0)
        {
            if(compute_values)
            {
                const T elem_start = indptr[kk];
                indices[elem_start + row_nnz] = static_cast<T>(kk);
                data[elem_start + row_nnz] = diag_vec[kk];
            }
            row_nnz += 1;
        }
    }
}

// Turn the per-row counts into the CSR row pointer (first pass only).
template <typename T>
inline void csr2_prefix_sum(const std::vector<T>& row_nnz_s,
                            const std::size_t subspace_dim,
                            T* __restrict indptr)
{
    T _sum = 0, temp;
    for(std::size_t kk = 0; kk < subspace_dim; kk++)
    {
        temp = _sum + row_nnz_s[kk];
        indptr[kk] = _sum;
        _sum = temp;
    }
    indptr[subspace_dim] = _sum;
}

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
    std::vector<T> row_nnz_s(subspace_dim, 0);
    if(has_nonzero_diag)
        csr2_diag<T, U>(diag_vec, subspace_dim, compute_values, indptr, indices, data, row_nnz_s);

    auto sink = [&](std::size_t out_row, std::size_t col_idx, U val) {
        T& row_nnz = row_nnz_s[out_row];
        if(compute_values)
        {
            const T elem_start = indptr[out_row];
            indices[elem_start + row_nnz] = static_cast<T>(col_idx);
            data[elem_start + row_nnz] = val;
        }
        row_nnz += 1;
    };

    type2_visit_groups<U>(terms,
                          subspace,
                          width,
                          subspace_dim,
                          group_ptrs,
                          group_ladder_ptrs,
                          group_rowint_length,
                          group_offdiag_inds,
                          num_groups,
                          ladder_offset,
                          sink);

    if(!compute_values)
        csr2_prefix_sum<T>(row_nnz_s, subspace_dim, indptr);
}

// Fast path: a prebuilt HalfStrTables replaces the per-candidate hash probes with direct
// partner discovery. Cartesian subspace -> type2_visit_cartesian, otherwise type2_visit_non_cartesian.
template <typename T, typename U>
void csr_matrix_builder2_halfstr(const HalfStrTables<U>& tables,
                                 const std::vector<OperatorTerm_t>& terms,
                                 const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                                 const U* __restrict diag_vec,
                                 const std::size_t subspace_dim,
                                 const int has_nonzero_diag,
                                 const width_t* __restrict group_rowint_length,
                                 const unsigned int ladder_offset,
                                 T* __restrict indptr,
                                 T* __restrict indices,
                                 U* __restrict data,
                                 const int compute_values)
{
    std::vector<T> row_nnz_s(subspace_dim, 0);
    if(has_nonzero_diag)
        csr2_diag<T, U>(diag_vec, subspace_dim, compute_values, indptr, indices, data, row_nnz_s);

    auto sink = [&](std::size_t out_row, std::size_t col_idx, U val) {
        T& row_nnz = row_nnz_s[out_row];
        if(compute_values)
        {
            const T elem_start = indptr[out_row];
            indices[elem_start + row_nnz] = static_cast<T>(col_idx);
            data[elem_start + row_nnz] = val;
        }
        row_nnz += 1;
    };

    if(tables.cartesian)
        type2_visit_cartesian<U>(
            tables, terms, subspace, subspace_dim, group_rowint_length, ladder_offset, sink);
    else
        type2_visit_non_cartesian<U>(
            tables, terms, subspace, subspace_dim, group_rowint_length, ladder_offset, sink);

    if(!compute_values)
        csr2_prefix_sum<T>(row_nnz_s, subspace_dim, indptr);
}
