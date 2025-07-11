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


void omp_matvec(const QubitOperator_t& ham,
    const std::vector<boost::dynamic_bitset<std::size_t> >& subspace,
    const std::complex<double> * diag_vec,
    const std::size_t width,
    const std::size_t subspace_dim,
    const int has_nonzero_diag,
    const std::size_t bin_width,
    const std::size_t *__restrict bin_ranges,
    const std::size_t *__restrict group_ptrs,
    const std::vector<std::vector<unsigned int>>& group_offdiag_inds,
    const std::size_t num_groups,
    const std::complex<double> *__restrict in_vec,
    std::complex<double> *__restrict out_vec)
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

    std::size_t num_terms = ham.terms.size();
    // Take care of off-diagonal terms
    if(num_terms)
    {
        #pragma omp for schedule(dynamic)
        for(kk=0; kk < subspace_dim; kk++)
        {
            boost::dynamic_bitset<std::size_t> col_vec;
            std::complex<double> temp_val, val=0;
            const OperatorTerm_t * term;
            std::size_t start, stop;
            std::size_t group_start, group_stop, group;
            std::size_t idx, weight, col_idx;
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
                for(idx=group_start; idx < group_stop; idx++)
                {
                    term = &ham.terms[idx];
                    weight = term->indices.size();
                    temp_val = 0;
                    if(do_col_search)
                    {
                        col_vec = subspace[kk];
                        flip_bits(col_vec, group_inds->data(), group_inds->size());
                        bin_int(col_vec, bin_width, bin_num);
                        start = bin_ranges[bin_num];
                        stop = bin_ranges[bin_num+1];
                        bitset_column_index(start, stop, col_vec, subspace, col_idx);
                        if(col_idx < MAX_SIZE_T) // column is in the subspace
                        {
                            do_col_search = 0; // do not search again for this group
                            if(term->extended) // check if extended term is zero
                            {
                                if(!nonzero_extended_bitset(term, subspace[kk])) // extended term is zero so move on to next term
                                {
                                    continue;
                                }
                            }
                            accum_element(subspace[kk], col_vec,
                                          &term->indices[0], &term->values[0],
                                          term->coeff, weight, temp_val);
                        }
                        else // column is not in the subspace so entire group does nothing, break
                        {
                            break;
                        }
                    }
                    else // column already found, process remaining terms
                    {
                        if(term->extended) // check if extended term is zero
                        {
                            if(!nonzero_extended_bitset(term, subspace[kk])) // extended term is zero so move on to next term
                            {
                                continue;
                            }
                        }
                        accum_element(subspace[kk], col_vec,
                                          &term->indices[0], &term->values[0],
                                          term->coeff, weight, temp_val);
                    }
                if(!do_col_search)
                {
                    val += temp_val * in_vec[col_idx];
                }
                } // end loop for this group
            } // end loop over all groups
            out_vec[kk] += val;
        } // end for-loop over rows
    } // end if num_terms
    } //end parallel region
} // end matvec
