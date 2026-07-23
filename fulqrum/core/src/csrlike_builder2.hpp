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
#include "csr_utils.hpp"
#include "csrlike.hpp"
#include "halfstr_tables.hpp"
#include "type2_common.hpp"
#include "type2_visit_cartesian.hpp"
#include "type2_visit_groups.hpp"
#include "type2_visit_non_cartesian.hpp"
#ifdef _OPENMP
#    include <omp.h>
#endif

// Builds the CSR-like (per-row vector) structure for a type-2 subspace Hamiltonian.
//
// The element visiting is shared with matvec2.hpp and csr2.hpp, all three differ
// only in what they do with each (row, col, value). This file supplies the append sink:
// push (col, val) onto that row's vectors.
//
// T is the data type, U is the index type, e.g. (complex, int).

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
    cols.resize(subspace_dim);
    data.resize(subspace_dim);

    if(has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
        for(std::size_t kk = 0; kk < subspace_dim; kk++)
            if(diag_vec[kk] != 0.0)
            {
                cols[kk].push_back(static_cast<U>(kk));
                data[kk].push_back(diag_vec[kk]);
            }
    }

    auto sink = [&](std::size_t out_row, std::size_t col_idx, T val) {
        cols[out_row].push_back(static_cast<U>(col_idx));
        data[out_row].push_back(val);
    };

    type2_visit_groups<T>(terms,
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

    sort_paired(cols, data);
}

// Fast path: a prebuilt HalfStrTables replaces the per-candidate hash probes with direct
// partner discovery. Cartesian subspace -> type2_visit_cartesian, otherwise type2_visit_non_cartesian.
template <typename T, typename U>
void csrlike_builder2_halfstr(const HalfStrTables<T>& tables,
                              const std::vector<OperatorTerm_t>& terms,
                              const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                              const T* __restrict diag_vec,
                              const std::size_t subspace_dim,
                              const int has_nonzero_diag,
                              const width_t* __restrict group_rowint_length,
                              const unsigned int ladder_offset,
                              std::vector<std::vector<U>>& cols,
                              std::vector<std::vector<T>>& data)
{
    if(has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
        for(std::size_t kk = 0; kk < subspace_dim; kk++)
            if(diag_vec[kk] != T(0))
            {
                cols[kk].push_back(static_cast<U>(kk));
                data[kk].push_back(diag_vec[kk]);
            }
    }

    auto sink = [&](std::size_t out_row, std::size_t col_idx, T val) {
        cols[out_row].push_back(static_cast<U>(col_idx));
        data[out_row].push_back(val);
    };

    if(tables.cartesian)
        type2_visit_cartesian<T>(
            tables, terms, subspace, subspace_dim, group_rowint_length, ladder_offset, sink);
    else
        type2_visit_non_cartesian<T>(
            tables, terms, subspace, subspace_dim, group_rowint_length, ladder_offset, sink);

    sort_paired(cols, data);
}
