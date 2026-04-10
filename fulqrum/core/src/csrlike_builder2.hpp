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
#include <iostream>
#include <complex>
#include <cstdlib>
#include <mutex>
#include <vector>

// #include <gperftools/profiler.h>

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
    // size_t survived_groups0 = 0, survived_groups1 = 0, survived_groups2 = 0;
    cols.resize(subspace_dim);
    data.resize(subspace_dim);

    std::vector<std::mutex> mutex1(subspace_dim);

    // -------------------------------------
    // Convert group_rowint_length to uint16
    // -------------------------------------
    std::vector<uint16_t> group_rowint_length_u16(num_groups);
    for(std::size_t i = 0; i < num_groups; i++)
    {
        group_rowint_length_u16[i] = (uint16_t)group_rowint_length[i];
    }

    // -------------------------------------------------------------------------------------
    // Convert group_offdiag_inds to array-based structure; trim inds size: 32-bit -> 16-bit
    // -------------------------------------------------------------------------------------
    // last element of inner array is the group size
    std::vector<std::array<uint16_t, 5>> group_offdiag_inds_array(num_groups);
    for(std::size_t i = 0; i < num_groups; i++)
    {
        group_offdiag_inds_array[i][4] = static_cast<uint16_t>(group_offdiag_inds[i].size());
        for(std::size_t j = 0; j < group_offdiag_inds[i].size(); j++)
        {
            group_offdiag_inds_array[i][j] = static_cast<uint16_t>(group_offdiag_inds[i][j]);
        }
        // Fill unused slots with 0 (for size-2 groups)
        for(std::size_t j = group_offdiag_inds[i].size(); j < 4; j++)
        {
            group_offdiag_inds_array[i][j] = 0;
        }
    }

    // ------------------------------------------------
    // Collect max inds of a group in a separate vector
    // ------------------------------------------------
    std::vector<uint16_t> grp_max_inds(num_groups, width);
    get_group_max_inds(grp_max_inds, group_offdiag_inds, num_groups);

    // Collect groups with same max inds for the lower-triangle check.
    // groups_by_maxbit[b] holds all groups whose max flip-position == b.
    std::vector<std::vector<uint32_t>> groups_by_maxbit(width);
    for (std::size_t g = 0; g < num_groups; g++)
        groups_by_maxbit[grp_max_inds[g]].push_back(static_cast<uint32_t>(g));


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

// size_t row_no_for_stats = (subspace_dim - 1);
#pragma omp parallel for schedule(dynamic) if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    { // begin loop over all rows
        const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;

        // define variables locally for omp for loop
        std::size_t idx;
        std::size_t group_int_start, group_int_stop;
        const OperatorTerm_t* term;
        boost::dynamic_bitset<std::size_t> col_vec;
        const std::array<uint16_t, 5>* group_inds;
        uint8_t group_size;
        std::size_t* col_ptr;
        std::size_t col_idx;
        T val;
        unsigned int row_int=0;
        uint8_t _p, _q, _r = 0;
        uint16_t pos0, pos1, pos2 = 0;

        std::vector<uint8_t> row_set_bits(row.size(), 0);
        bitset_to_bitvec(row, row_set_bits);

        // Iterate over bits of the row and do the lower-triangle check.
        for (std::size_t b = 0; b < static_cast<std::size_t>(width); b++)
        {
            if (!row_set_bits[b]) continue;
            for (const uint32_t group : groups_by_maxbit[b])
            {
                // if (kk == row_no_for_stats) survived_groups0++;

                group_inds = &group_offdiag_inds_array[group];
                group_size = (uint8_t)(*group_inds)[4];

                pos0 = (*group_inds)[0];
                pos1 = (*group_inds)[1];

                _p = row_set_bits[pos0];
                _q = row_set_bits[pos1];

                // Connected determinants filter
                if (group_size == 4)
                {
                    pos2 = (*group_inds)[2];
                    _r = row_set_bits[pos2];

                    if ((_p + _q + _r) != 1)
                        continue;

                    if ((pos1 < width/2) && (pos2 >= width/2))
                        if ((_p == _q) || (_r))
                            continue;
                }
                else // group_size == 2
                {
                    if (_p)
                        continue;
                }

                // if (kk == row_no_for_stats) survived_groups1++;

                col_vec = row;
                flip_bits_u16(col_vec, group_inds->data(), group_size);

                col_ptr = subspace.get_ptr(col_vec);
                if (col_ptr == nullptr) continue;
                col_idx = *col_ptr;

                // if (kk == row_no_for_stats) survived_groups2++;

                row_int = bitset_ladder_int_u16(
                    row_set_bits.data(), group_inds->data(),
                    group_rowint_length_u16[group]);
                group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
                group_int_stop  = group_ladder_ptrs[group * ladder_offset + row_int + 1];

                val = 0;
                for (idx = group_int_start; idx < group_int_stop; idx++)
                {
                    term = &terms[idx];
                    if (passes_proj_validation(term, row))
                        accum_element(row, col_vec,
                                      &term->indices[0], &term->values[0],
                                      term->coeff, term->real_phase,
                                      term->indices.size(), val);
                }

                if (std::abs(val) > ATOL)
                {
                    {
                        std::lock_guard<std::mutex> lock_kk(mutex1[kk]);
                        cols[kk].push_back(col_idx);
                        data[kk].push_back(val);
                    }
                    {
                        std::lock_guard<std::mutex> lock_col_idx(mutex1[col_idx]);
                        cols[col_idx].push_back(kk);
                        if constexpr(std::is_same_v<T, double>)
                            data[col_idx].push_back(val);
                        else
                            data[col_idx].push_back(std::conj(val));
                    }
                }
            }
        }
    }

    std::cout << "Num groups: " << num_groups << std::endl;
    // std::cout << "[kk=" << row_no_for_stats << "] surviving groups lower-tri filter: " << survived_groups0 << std::endl;
    // std::cout << "[kk=" << row_no_for_stats << "] surviving groups conndets filter:  " << survived_groups1 << std::endl;
    // std::cout << "[kk=" << row_no_for_stats << "] surviving groups hashmap:          " << survived_groups2 << std::endl;
    sort_paired(cols, data);
} // end function
