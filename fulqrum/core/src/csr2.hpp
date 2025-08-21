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


template <typename T, typename U> void csr_matrix_builder2(const OperatorTerm_t * terms,
                                              const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                                              const U * __restrict diag_vec,
                                              const unsigned int width,
                                              const std::size_t subspace_dim,
                                              const int has_nonzero_diag,
                                              const std::size_t * __restrict group_ptrs,
                                              const std::size_t *__restrict group_ladder_ptrs,
                                              const unsigned int *__restrict group_rowint_length,
                                              const std::vector<std::vector<unsigned int>>& group_offdiag_inds,
                                              const std::size_t num_groups,
                                              const unsigned int ladder_offset,
                                              T * __restrict indptr,
                                              T * __restrict indices,
                                              U * __restrict data,
                                              const int compute_values)
{
    std::size_t kk;
    T temp, _sum;

    const bitset_map_namespace::BitsetMap& subsapce_hash_map = subspace.get_map();
    const auto* bitsets = subsapce_hash_map.values();

    #pragma omp parallel for schedule(dynamic) if(subspace_dim > 128)
    for(kk=0; kk<subspace_dim; kk++)
    { // begin loop over all rows
        // define variables locally for omp for loop
        std::size_t idx;
        std::size_t group;
        std::size_t group_int_start, group_int_stop;
        T row_nnz, elem_start;
        const OperatorTerm_t * term;
        boost::dynamic_bitset<std::size_t> row, col_vec;
        const std::vector<unsigned int> * group_inds;
        std::size_t* col_ptr;
        U val;
        unsigned int row_int;
        int do_col_search;
        row_nnz = 0;
        row = bitsets[kk].first;
        elem_start = indptr[kk];
        // do diagonal first, if any
        if(has_nonzero_diag)
        {
            if(diag_vec[kk] != 0.0)
            {
                if(compute_values)
                {
                    indices[elem_start+row_nnz] = kk;
                    data[elem_start+row_nnz] = diag_vec[kk];
                }
                row_nnz += 1;
            }
        }
        for(group=0; group < num_groups; group++)
        { // begin loop over groups
            group_inds = &group_offdiag_inds[group];
            row_int = bitset_ladder_int(row, group_inds->data(), group_rowint_length[group]);
            group_int_start = group_ladder_ptrs[group*ladder_offset+row_int];
            group_int_stop = group_ladder_ptrs[group*ladder_offset+row_int+1];
            do_col_search = 1;
            val = 0;
            for(idx=group_int_start; idx < group_int_stop; idx++)
            { // begin loop over terms in this group
                if(do_col_search)
                {
                    col_vec = row;
                    flip_bits(col_vec, group_inds->data(), group_inds->size());
                    col_ptr = subspace.get_ptr(col_vec);
                    if(col_ptr == nullptr){break;} // column is NOT in the subspace so break group
                    do_col_search = 0;
                }
                term = &terms[idx];
                if(passes_proj_validation(term, row))
                {
                    accum_element(row, col_vec,
                                      &term->indices[0], &term->values[0], term->coeff, term->real_phase, 
                                      term->indices.size(), val);
                }
            } // end loop over terms in this group
            if(val!=0.0)
            {
                if(compute_values)
                {
                indices[elem_start+row_nnz] = *col_ptr;
                data[elem_start+row_nnz] = val;
                }
                row_nnz += 1;
            }
        } // end loop over groups
        if(!compute_values)  // done with row, add row_nnz to indptr
        {
            indptr[kk] = row_nnz;
        }
    } // end loop over all rows
    if(!compute_values) // Done with all rows so accumulate for correct indptr structure
    {
        _sum = 0;
        for(kk=0; kk < (subspace_dim+1); kk++)
        {
            temp = _sum + indptr[kk];
            indptr[kk] = _sum;
            _sum = temp;
        }
    }
}
