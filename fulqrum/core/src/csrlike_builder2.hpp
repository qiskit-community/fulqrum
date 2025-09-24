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
                         std::vector<std::vector<U>>& cols,
                         std::vector<std::vector<T>>& data)
{
    std::size_t kk;
    T temp, _sum;

    const auto *bitsets = subspace.get_bitsets();

    #pragma omp parallel for schedule(dynamic) if (subspace_dim > 128)
    for (kk = 0; kk < subspace_dim; kk++)
    { // begin loop over all rows

        const boost::dynamic_bitset<size_t> &row = bitsets[kk].first;
        // creates a vector representation of the row bitset
        // with 1 at set-bit positions. This vector is easier to
        // look-up by index as looking up a bit in a bitset required
        // division followed modulo operations.
        // code from:
        // https://lemire.me/blog/2018/02/21/iterating-over-set-bits-quickly/
        std::vector<uint8_t> row_set_bits(row.size(), 0);
        for (size_t block = 0; block < row.num_blocks(); block++)
        {
            auto bitset = row.m_bits[block];
            while (bitset != 0)
            {
                uint64_t t = bitset & -bitset;
                int r = __builtin_ctzll(bitset);
                row_set_bits[block * BITS_PER_BLOCK + r] = 1;
                bitset ^= t;
            }
        }
        // define variables locally for omp for loop
        std::size_t idx;
        std::size_t group;
        std::size_t group_int_start, group_int_stop;
        const OperatorTerm_t *term;
        boost::dynamic_bitset<std::size_t> col_vec;
        const std::vector<unsigned int> *group_inds;
        std::size_t *col_ptr;
        T val;
        std::vector<U> * row_cols = &cols[kk];
        std::vector<T> * row_data = &data[kk];
        unsigned int row_int;
        int do_col_search;

        // need two different types for sorting 
        int sort_start_int = 0;
        int sort_end_int = 0;
        long long sort_start_long = 0;
        long long sort_end_long = 0;
        

        // do diagonal first, if any
        if(has_nonzero_diag)
        {
            if (diag_vec[kk] != 0.0)
            {
                row_cols->push_back(kk);
                row_data->push_back(diag_vec[kk]);
            }
        }
        for (group = 0; group < num_groups; group++)
        { // begin loop over groups
            group_inds = &group_offdiag_inds[group];
            row_int = bitset_ladder_int(
                row_set_bits.data(),
                group_inds->data(),
                group_rowint_length[group]);
            group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
            group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];
            do_col_search = 1;
            val = 0;
            for (idx = group_int_start; idx < group_int_stop; idx++)
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
                    accum_element(row, col_vec,
                                  &term->indices[0], &term->values[0], term->coeff,
                                  term->real_phase, term->indices.size(), val);
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
} // end function
