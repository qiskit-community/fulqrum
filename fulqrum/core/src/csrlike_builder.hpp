/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <complex>

#include "base.hpp"
#include "bitset_utils.hpp"
#include "bitset_hashmap.hpp"
#include "elements.hpp"
#include "operators.hpp"
#include <boost/dynamic_bitset.hpp>

template <typename T, typename U>
void csrlike_builder(const OperatorTerm_t *terms,
                     const bitset_map_namespace::BitsetHashMapWrapper &subspace,
                     const T *__restrict diag_vec,
                     const unsigned int width,
                     const std::size_t subspace_dim,
                     const int has_nonzero_diag,
                     const std::size_t *__restrict group_ptrs,
                     const std::vector<std::vector<unsigned int>> &group_offdiag_inds,
                     const std::size_t num_groups,
                     std::vector<std::vector<U>>& cols,
                     std::vector<std::vector<T>>& data)
{
    std::size_t kk;
    T temp, _sum;

    const auto *bitsets = subspace.get_bitsets();

    #pragma omp parallel for if (subspace_dim > 4096)
    for (kk = 0; kk < subspace_dim; kk++)
    { // begin loop over all rows
        // define variables locally for omp for loop
        std::size_t idx;
        std::size_t group_start, group_stop, group;
        const OperatorTerm_t *term;
        const boost::dynamic_bitset<size_t> &row = bitsets[kk].first;
        boost::dynamic_bitset<std::size_t> col_vec;
        std::size_t *col_ptr;
        const std::vector<unsigned int> *group_inds;
        T val;
        std::vector<U> * row_cols = &cols[kk];
        std::vector<T> * row_data = &data[kk];

        // need two different types for sorting 
        int sort_start_int = 0;
        int sort_end_int = 0;
        long long sort_start_long = 0;
        long long sort_end_long = 0;
        
        int do_col_search;
        // do diagonal first, if any
        if (has_nonzero_diag)
        {
            if (diag_vec[kk] != 0.0)
            {
                row_cols->push_back(kk);
                row_data->push_back(diag_vec[kk]);
            }
        }
        for (group = 0; group < num_groups; group++)
        { // begin loop over groups
            group_start = group_ptrs[group];
            group_stop = group_ptrs[group + 1];
            do_col_search = 1;
            val = 0;
            group_inds = &group_offdiag_inds[group];
            for (idx = group_start; idx < group_stop; idx++)
            { // begin loop over terms in this group
                if (do_col_search)
                {
                    col_vec = row;
                    flip_bits(col_vec, group_inds->data(), group_inds->size());
                    col_ptr = subspace.get_ptr(col_vec);
                    if (col_ptr == nullptr)
                    {
                        break;
                    } // column is NOT in the subspace so break group
                    do_col_search = 0;
                }
                term = &terms[idx];
                if (passes_proj_validation(term, row))
                {
                    accum_element(row, col_vec, &term->indices[0], &term->values[0],
                                  term->coeff, term->real_phase, term->indices.size(), val);
                }
            } // end loop over terms in this group
            if (val != 0.0)
            {
                row_cols->push_back(*col_ptr);
                row_data->push_back(val);
            }
        } // end loop over groups
        // sort column indices and data in each row
        if constexpr (std::is_same_v<U, int>)
        {
            sort_end_int = row_cols->size() - 1;
            quicksort_indices_data(row_cols->data(), row_data->data(), sort_start_int, sort_end_int);
        }
        else
        {
            sort_end_long = row_cols->size() - 1;
            quicksort_indices_data(row_cols->data(), row_data->data(), sort_start_long, sort_end_long);
        }
    } // end loop over all rows
}
