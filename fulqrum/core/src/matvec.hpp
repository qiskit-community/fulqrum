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
#include "bitstrings.hpp"
#include "elements.hpp"
#include "operators.hpp"


void omp_matvec(QubitOperator_t& ham,
    std::vector<unsigned char>& subspace,
    std::complex<double> * diag_vec,
    std::size_t width,
    std::size_t subspace_dim,
    int has_nonzero_diag,
    std::size_t bin_width,
    std::size_t * bin_ranges,
    std::size_t * group_ptrs,
    std::size_t num_groups,
    const std::complex<double> * in_vec,
    std::complex<double> * out_vec)
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
            const unsigned char * row_start = &subspace[kk*width];
            std::vector<unsigned char> col_vec;
            std::complex<double> temp_val, val=0;
            OperatorTerm_t * term;
            std::size_t start, stop;
            std::size_t group_start, group_stop, group;
            std::size_t idx, weight, col_idx;
            int bin_num;
            int do_col_search;
            col_vec.resize(width);
            // Loop over all off-diagonal terms in operator
            for(group=0; group < num_groups; group++)
            {
                group_start = group_ptrs[group];
                group_stop = group_ptrs[group+1];
                do_col_search = 1;
                for(idx=group_start; idx < group_stop; idx++)
                {
                    term = &ham.terms[idx];
                    weight = term->indices.size();
                    if(do_col_search)
                    {
                        memcpy(&col_vec[0], row_start, width);
                        get_column_vec(row_start, &col_vec[0], width, &term->indices[0], &term->values[0], weight);
                        bin_num = bin_width_to_int(&col_vec[0], width, bin_width);
                        start = bin_ranges[bin_num];
                        stop = bin_ranges[bin_num+1];
                        col_idx = col_index(start, stop, &col_vec[0], &subspace[0], width);
                        if(col_idx < MAX_SIZE_T) // column is in the subspace
                        {
                            do_col_search = 0; // do not search again for this group
                            if(term->extended) // check if extended term is zero
                            {
                                if(!nonzero_extended_value(term, row_start, width)) // extended term is zero so move on to next term
                                {
                                    continue;
                                }
                            }
                            temp_val = compute_element_vec(row_start, &col_vec[0], width,
                                                        &term->indices[0], &term->values[0],
                                                        term->coeff, weight);
                            val += temp_val * in_vec[col_idx];
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
                            if(!nonzero_extended_value(term, row_start, width)) // extended term is zero so move on to next term
                            {
                                continue;
                            }
                        }
                        temp_val = compute_element_vec(row_start, &col_vec[0], width,
                                                       &term->indices[0], &term->values[0],
                                                       term->coeff, weight);
                        val += temp_val * in_vec[col_idx];
                    }
                } // end loop for this group
            } // end loop over all groups
            out_vec[kk] += val;
        } // end for-loop over rows
    } // end if num_terms
    } //end parallel region
} // end matvec
