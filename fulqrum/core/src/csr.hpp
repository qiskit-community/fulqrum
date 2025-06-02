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
#include "elements.hpp"
#include "operators.hpp"
#include <boost/dynamic_bitset.hpp>


template <typename T> void csr_matrix_builder(const OperatorTerm_t * terms,
                                              const std::vector<boost::dynamic_bitset<std::size_t> >& subspace,
                                              const std::complex<double> * diag_vec,
                                              const unsigned int width,
                                              const std::size_t subspace_dim,
                                              const int has_nonzero_diag,
                                              const unsigned int bin_width,
                                              const std::size_t * bin_ranges,
                                              const std::size_t * group_ptrs,
                                              const std::size_t num_groups,
                                              T * indptr,
                                              T * indices,
                                              std::complex<double> * data,
                                              const int compute_values)
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
        boost::dynamic_bitset<std::size_t> col_vec;
        std::size_t col_idx;
        unsigned int weight;
        std::complex<double> val;
        int do_col_search;
        std::size_t bin_num;
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
                    col_vec = subspace[kk];
                    get_column_bitset(col_vec, &term->indices[0], &term->values[0], weight);
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
                                      &term->indices[0], &term->values[0], term->coeff,
                                      weight, val);
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
                                  &term->indices[0], &term->values[0], term->coeff,
                                  weight, val);
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
    if(!compute_values) // Done all rows so cummulate for correct indptr structure
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
