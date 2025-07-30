/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <cstddef>
#include <vector>
#include <complex>

#include "base.hpp"
#include "bitset_utils.hpp"
#include "elements.hpp"
#include "operators.hpp"
#include <boost/dynamic_bitset.hpp>


template <typename T> void omp_matvec(const std::vector<OperatorTerm_t>& terms,
                                        const std::vector<boost::dynamic_bitset<std::size_t> >& subspace,
                                        const T * diag_vec,
                                        const std::size_t width,
                                        const std::size_t subspace_dim,
                                        const int has_nonzero_diag,
                                        const std::size_t bin_width,
                                        const std::size_t *__restrict bin_ranges,
                                        const std::size_t *__restrict group_ptrs,
                                        const std::vector<std::vector<unsigned int>>& group_offdiag_inds,
                                        const std::size_t num_groups,
                                        const T *__restrict in_vec,
                                        T *__restrict out_vec)
{
    std::size_t kk;
    #pragma omp parallel if(subspace_dim > 128)
    {
    // Take care of diagonal term first, if any (usually there is)
    if(has_nonzero_diag){
        #pragma omp for
        for(kk=0; kk < subspace_dim; kk++){
            out_vec[kk] = diag_vec[kk]*in_vec[kk];
        }
    }

    std::size_t num_terms = terms.size();
    // Take care of off-diagonal terms
    if(num_terms)
    {
        #pragma omp for schedule(dynamic)
        for(kk=0; kk < subspace_dim; kk++)
        {
            boost::dynamic_bitset<std::size_t> row, col_vec;
            row = subspace[kk];
            T temp_val, val=0;
            const OperatorTerm_t * term;
            std::size_t start, stop;
            std::size_t group_start, group_stop, group;
            std::size_t idx, col_idx;
            std::size_t bin_num;
            const std::vector<unsigned int> * group_inds;
            int do_col_search;
            // Loop over all off-diagonal terms in operator
            for(group=0; group < num_groups; group++)
            {
                group_start = group_ptrs[group];
                group_stop = group_ptrs[group+1];
                do_col_search = 1;
                group_inds = &group_offdiag_inds[group];
                temp_val = 0;
                for(idx=group_start; idx < group_stop; idx++)
                {
                    if(do_col_search)
                    {
                        col_vec = row;
                        flip_bits(col_vec, group_inds->data(), group_inds->size());
                        bin_int(col_vec, bin_width, bin_num);
                        start = bin_ranges[bin_num];
                        stop = bin_ranges[bin_num+1];
                        bitset_column_index(start, stop, col_vec, subspace, col_idx);
                        if(col_idx == MAX_SIZE_T){break;} // column is NOT in the subspace so break group
                        do_col_search = 0;
                    }
                    term = &terms[idx];
                    if(passes_proj_validation(term, row))
                    {
                        accum_element(row, col_vec, &term->indices[0], &term->values[0],
                                      term->coeff, term->real_phase, term->indices.size(), temp_val);
                    }
                } // end loop for this group
                if(!do_col_search)
                {
                    val += temp_val * in_vec[col_idx];
                }
            } // end loop over all groups
            out_vec[kk] += val;
        } // end for-loop over rows
    } // end if num_terms
    } //end parallel region
} // end matvec
