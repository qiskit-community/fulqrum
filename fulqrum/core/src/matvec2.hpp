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

// Hermitian sparse matrix-vector product over a type-2 subspace Hamiltonian.
//
// The element visiting itself is shared with csr2.hpp and csrlike_builder2.hpp --
// all three differ only in what they do with each (row, col, value). This file supplies
// the matvec sink:  out[row] += val * in[col].
//
//   omp_matvec2: generic; works on any subspace (type2_visit_groups)
//   omp_matvec2_halfstr: fast; needs HalfStrTables
//                          (type2_visit_cartesian / type2_visit_non_cartesian)
//
// Each row is owned by a single thread for the duration of the visit, so
// out_vec needs no locking.

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
    if(has_nonzero_diag)
    {
        for(std::size_t kk = 0; kk < subspace_dim; kk++)
            out_vec[kk] = diag_vec[kk] * in_vec[kk];
    }

    auto sink = [&](std::size_t out_row, std::size_t col_idx, T val) {
        out_vec[out_row] += val * in_vec[col_idx];
    };

    type2_visit_groups<T>(terms,
                          subspace,
                          width,
                          subspace_dim,
                          group_ptrs,
                          group_ladder_ptrs,
                          group_rowint_length,
                          group_offdiag_inds,
                          static_cast<std::size_t>(num_groups),
                          ladder_offset,
                          sink);
}

// Fast path: a prebuilt HalfStrTables replaces the per-candidate hash probes with direct
// partner discovery. Cartesian subspace -> type2_visit_cartesian, otherwise type2_visit_non_cartesian.
template <typename T>
void omp_matvec2_halfstr(const HalfStrTables<T>& tables,
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
        for(std::size_t kk = 0; kk < subspace_dim; kk++)
            out_vec[kk] = diag_vec[kk] * in_vec[kk];
    }

    auto sink = [&](std::size_t out_row, std::size_t col_idx, T val) {
        out_vec[out_row] += val * in_vec[col_idx];
    };

    if(tables.cartesian)
        type2_visit_cartesian<T>(
            tables, terms, subspace, subspace_dim, group_rowint_length, ladder_offset, sink);
    else
        type2_visit_non_cartesian<T>(
            tables, terms, subspace, subspace_dim, group_rowint_length, ladder_offset, sink);
}
