/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <mutex>
#include <vector>
#include <complex>

#include "base.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "bitset_hashmap.hpp"
#include "elements.hpp"
#include "operators.hpp"
#include "csrlike.hpp"
#include "csr_utils.hpp"
#include <boost/dynamic_bitset.hpp>

// T is the data type, U is in the index type, e.g (complex, int)
template <typename T, typename U>
void csrlike_builder2(const OperatorTerm_t *terms,
                      const bitset_map_namespace::BitsetHashMapWrapper &subspace,
                      const T *__restrict diag_vec,
                      const unsigned int width,
                      const std::size_t subspace_dim,
                      const int has_nonzero_diag,
                      const std::size_t *__restrict group_ptrs,
                      const std::size_t *__restrict group_ladder_ptrs,
                      const unsigned int *__restrict group_rowint_length,
                      const std::vector<std::vector<unsigned int>> &group_offdiag_inds,
                      const std::size_t num_groups,
                      const unsigned int ladder_offset,
                      std::vector<std::vector<U>> &cols,
                      std::vector<std::vector<T>> &data)
{
    std::size_t kk;
    const auto *bitsets = subspace.get_bitsets();

    const auto smallest_bitset = bitsets[0].first;
    const auto largest_bitset = bitsets[(subspace_dim - 1)].first;

    cols.resize(subspace_dim);
    data.resize(subspace_dim);
    std::vector<std::mutex> row_mutex(subspace_dim);

    // Note: LOWER TRAINGLE ELEMENTS DETECTION
    // Following block gets the max item of ``group_offdiag_inds``
    // for each group.
    // Only checking the bit at max index position
    // can tell us if a potential column is greater than
    // the row.
    // Inspecting col_idx < row_idx (kk) helps us detect the location of
    // lower triangle elements of the matrix.
    // When subspace bitstrings are sorted in the ascending order,
    // a col_bitset < row_bitset means the corresponding col_idx < row_idx (kk)
    //
    // Our potential columns are found by flipping row bits at
    // ``group_offdiag_inds[group]`` positions.
    // If the row bit at max inds position is ``0``, it will be
    // flipped to ``1`` in the column, and thus, column will be > the row.
    // So, we do not need to construct the full col_vec and check
    // whether col_vec < row (or col_vec > row) to detect, whether we
    // are potentially in the lower triangle or not. Only, testing
    // max group inds bit location in row tells us this, and
    // we can avoid expensive computations.
    std::vector<uint16_t> grp_max_inds(num_groups, width);
    get_group_max_inds(grp_max_inds, group_offdiag_inds, num_groups);

    // Populate diagonal elements first separately
    if (has_nonzero_diag)
    {
#pragma omp parallel for schedule(dynamic) if (subspace_dim > 4096)
        for (kk = 0; kk < subspace_dim; kk++)
        { // begin loop over all rows

            if (diag_vec[kk] != 0.0)
            {
                cols[kk].push_back(kk);
                data[kk].push_back(diag_vec[kk]);
            }
        }
    }

#pragma omp parallel for schedule(dynamic) if (subspace_dim > 4096)
    for (kk = 0; kk < subspace_dim; kk++)
    { // begin loop over all rows

        const boost::dynamic_bitset<size_t> &row = bitsets[kk].first;

        std::vector<uint8_t> row_set_bits(row.size(), 0);
        bitset_to_bitvec(row, row_set_bits);

        // define variables locally for omp for loop
        std::size_t idx;
        std::size_t group;
        std::size_t group_int_start, group_int_stop;
        const OperatorTerm_t *term;
        boost::dynamic_bitset<std::size_t> col_vec;
        const std::vector<unsigned int> *group_inds;
        std::size_t *col_ptr;
        std::size_t col_idx;
        T val;
        unsigned int row_int;

        for (group = 0; group < num_groups; group++)
        { // begin loop over groups

            // see Note: LOWER TRAINGLE DETECTION at the beginning.
            // For lower triangle elements, col_idx < row_idx (kk).
            // For sorted bitstrings (must), it means col_vec < row.
            // Checking the bit location at the max index of group_offdiag_inds[group]
            // (it is the differing bit location) tells us if potential col_vec < row.
            // If potential col_vec > row, we are in the upper triangle, and we can skip
            // to the group without doing expensive col_vec construction and subsapce
            // look-up.

            if (!row_set_bits[grp_max_inds[group]])
            {
                continue;
            }

            group_inds = &group_offdiag_inds[group];
            //
            row_int = bitset_ladder_int(
                row_set_bits.data(),
                group_inds->data(),
                group_rowint_length[group]);

            group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
            group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];
            val = 0;

            if (group_int_start < group_int_stop)
            {
                col_vec = row;
                flip_bits(col_vec, group_inds->data(), group_inds->size());

                // if col_vec is smaller than the smallest bitset
                // it is already outside the subspace, and we can
                // safely skip the group. As we only do col_idx < row_idx
                // col_vec cannot be > largest_bitset
                // Shows minor speed-up for fragment and dimer case.
                if (col_vec < smallest_bitset)
                {
                    continue;
                }

                col_ptr = subspace.get_ptr(col_vec);
                if (col_ptr == nullptr)
                {
                    continue;
                } // column is NOT in the subspace so break group
                col_idx = *col_ptr;
            }

            for (idx = group_int_start; idx < group_int_stop; idx++)
            { // begin loop over terms in this group
                term = &terms[idx];
                if (passes_proj_validation(term, row))
                {
                    accum_element(row, col_vec, &term->indices[0], &term->values[0],
                                  term->coeff, term->real_phase, term->indices.size(), val);
                }
            } // end loop over terms in this group

            if (std::abs(val) > ATOL)
            {
                // Mutex locks to avoid write contention inside OMP
                // parallel for loop. After detecting an lower triangle
                // elements, we also populate corresponding upper triangle
                // position. It may lead to multiple parallel threads writing
                // into the same inner vector at the same time.
                // These scoped (inside each curly braces {}) Mutex-based locks
                // are supposed to prevent simulaneous writing into a same vector.
                {
                    std::lock_guard<std::mutex> lock_kk(row_mutex[kk]);
                    cols[kk].push_back(col_idx);
                    data[kk].push_back(val);
                }

                {
                    std::lock_guard<std::mutex> lock_col_idx(row_mutex[col_idx]);
                    cols[col_idx].push_back(kk);
                    if constexpr (std::is_same_v<T, double>)
                    {
                        data[col_idx].push_back(val);
                    }
                    else
                    {
                        // for complex-valued matrix, lower triangle
                        // element will be complex conjugate of the upper
                        // triangle element
                        data[col_idx].push_back(std::conj(val));
                    }
                }
            }
        } // end loop over groups
    }

#pragma omp parallel for schedule(dynamic) if (subspace_dim > 4096)
    for (kk = 0; kk < subspace_dim; kk++)
    {
        // need two different types for sorting
        int sort_start_int = 0;
        int sort_end_int = 0;
        long long sort_start_long = 0;
        long long sort_end_long = 0;
        // sort column indices and data in each row
        if constexpr (std::is_same_v<U, int>)
        {
            sort_end_int = cols[kk].size() - 1;
            quicksort_indices_data(cols[kk].data(), data[kk].data(), sort_start_int, sort_end_int);
        }
        else
        {
            sort_end_long = cols[kk].size() - 1;
            quicksort_indices_data(cols[kk].data(), data[kk].data(), sort_start_long, sort_end_long);
        }
        cols[kk].shrink_to_fit();
        data[kk].shrink_to_fit();
    } // end loop over all rows
} // end function
