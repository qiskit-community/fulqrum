/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <complex>

#include "base.hpp"
#include "bitstrings.hpp"
#include "elements.hpp"
#include "operators.hpp"


template <typename T> void csr_matrix_builder(const OperatorTerm_t * terms,
                                            std::vector<unsigned char>& subspace,
                                            const std::complex<double> * diag_vec,
                                            std::size_t width,
                                            std::size_t subspace_dim,
                                            int has_nonzero_diag,
                                            std::size_t bin_width,
                                            const std::size_t * bin_ranges,
                                            const std::size_t * group_ptrs,
                                            std::size_t num_groups,
                                            T * indptr,
                                            T * indices,
                                            std::complex<double> * data,
                                            int compute_values)
{
    std::size_t kk;
    T temp, _sum;
    #pragma omp parallel for if(subspace_dim > 128)
    for(kk=0; kk<subspace_dim; kk++)
    { // begin loop over all rows
        // define variables locally for omp for loop
        std::size_t idx;
        std::size_t group_start, group_stop, group;
        std::size_t start, stop;
        T row_nnz, elem_start;
        const OperatorTerm_t * term;
        std::vector<unsigned char> col_vec;
        std::size_t weight, col_idx;
        std::complex<double> val;
        const unsigned char * row_start;
        int do_col_search;
        std::size_t bin_num;
        col_vec.resize(width);
        row_start = &subspace[kk*width];
        row_nnz = 0;
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
            group_start = group_ptrs[group];
            group_stop = group_ptrs[group+1];
            do_col_search = 1;
            val = 0;
            for(idx=group_start; idx < group_stop; idx++)
            { // begin loop over terms in this group
                term = &terms[idx];
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
                        val += compute_element_vec(row_start, &col_vec[0], width,
                                                    &term->indices[0], &term->values[0], 
                                                    term->coeff, weight);
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
                    val += compute_element_vec(row_start, &col_vec[0], width,
                                               &term->indices[0], &term->values[0],
                                               term->coeff, weight);
                }
            } // end loop over terms in this group
            if(val!=0.0)
            {
                if(compute_values)
                {
                indices[elem_start+row_nnz] = col_idx;
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
    if(not compute_values) // Done all rows so cummulate for correct indptr structure
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


template <typename T>void csr_spmv(const T *__restrict indptr, const T *__restrict indices,
                                   const std::complex<double> *__restrict data, 
                                   const std::complex<double> *__restrict vec, 
                                   std::complex<double> *__restrict out, std::size_t dim)
    {
        std::size_t row;
        #pragma omp parallel for if(dim > 128)
        for(row=0; row < dim; row++)
        {   
            T jj;
            T row_start, row_end;
            std::complex<double> dot = 0;
            row_start = indptr[row];
            row_end = indptr[row+1];
            for(jj=row_start; jj < row_end; jj++)
            {
                dot += data[jj]*vec[indices[jj]];
            }
            out[row] += dot;
        }
    }
