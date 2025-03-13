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


/**
 * Perform SpMV for a CSR matrix with complex data
 *
 * @param data Complex data
 * @param indices Column indices for the data
 * @param indptr Row pointer indices
 * @param x Input vector
 * @param out Output vector
 * @param nrows Dimenion of the matrix
 */
 template <typename T> void csr_matvec(const std::complex<double> * data, const T * indices, const T * indptr,
                                       const std::complex<double> * x, std::complex<double> * out, std::size_t nrows)
{
    #pragma omp parallel if(nrows > 128)
    for (std::size_t row=0; row < nrows; row++)
    {
        std::complex<double> dot = 0;
        T row_start = indptr[row];
        T row_end = indptr[row+1];
        for (T jj=row_start; jj <row_end; jj++)
        {
            dot += data[jj]*x[indices[jj]];
        }
        out[row] += dot;
    }
}


/**
 * Build a CSR matrix directly from subspace Hamiltonian
 *
 * This routine is called in two steps, first with compute_values=0
 * which finds the number of nonzero elements in each row.  A second
 * run with compute_values=1 populates the matrix with the actual
 * numerical values.
 *
 * @param ham Off-diagonal Hamiltonian terms
 * @param subspace Subspace of bit-strings
 * @param diag_vec Cached vector of diagonal entries
 * @param with Operator width
 * @param subspace_dim Dimensionality (size) of subspace
 * @param has_nonzero_diag Flag indicating a nonzero diagonal vector
 * @param bin_width Number of bits used for binning
 * @param bin_ranges Array giving start and stop indices for bins
 * @param indptr Row indices pointer array
 * @param indices Column indices of nonzero elements
 * @param data Complex data for matrix
 * @param compute_values Flag for indicating that values of matrix should be populated
 */
template <typename T> void csr_builder(QubitOperator_t& ham, std::vector<unsigned char>& subspace,
                                       std::complex<double> * diag_vec,
                                       std::size_t width, std::size_t subspace_dim,
                                       int has_nonzero_diag,
                                       std::size_t bin_width, std::size_t * bin_ranges,
                                       T * indptr, T * indices, std::complex<double> * data,
                                       int compute_values)
{
    std::size_t kk;
    T temp, _sum;
    std::size_t num_terms = ham.terms.size();
    OperatorTerm_t * terms = &ham.terms[0];

    #pragma omp parallel if(nrows > 128)
    for(kk=0; kk < subspace_dim; kk++) //do this loop in openmp
    {
        // Define local variables for openmp
        const unsigned char * row_start = &subspace[kk*width];
        std::vector<unsigned char> col_vec;
        col_vec.resize(width);
        OperatorTerm_t * term;
        int elem_start = indptr[kk];
        int elem_offset = 0;
        T nnz = 0;
        std::size_t idx = 0;
        int bin_num;
        std::complex<double> val;
        int current_group, col_found, col_check;
        std::size_t weight, col_idx, start, stop;
        // Check if diagonal term is nonzero
        if(has_nonzero_diag)
        {
            if (diag_vec[kk] != 0.0)
            {
                nnz += 1;
                // populate values if requested
                if(compute_values)
                {
                    indices[elem_start+elem_offset] = kk;
                    data[elem_start+elem_offset] = diag_vec[kk];
                    elem_offset += 1;
                }
            }
        } // end diagonal check

        // While loop over all terms in Hamiltonian
        while(idx < num_terms)
        {
            val = 0;
            col_found = 0;
            col_check = 0;
            current_group = terms[idx].group;
            // Iterate over all terms with the same group label, i.e. point to same column
            while (terms[idx].group == current_group && idx < num_terms)
            {
                term = &terms[idx];
                weight = term->indices.size();
                // If the column was found in the subspace
                if(col_found)
                {
                    val += compute_element_vec(row_start, &col_vec[0], width,
                           &term->indices[0], &term->values[0], term->coeff,
                           weight);
                    idx += 1;
                    continue;
                }
                // # If we did not check if the column is in the subspace yet
                if(!col_check)
                {
                    // check if element for group is in the subspace
                    if(term->extended)
                    {
                        // extended term is zero so move on to next term
                        if(!nonzero_extended_value(term, row_start, width))
                        {
                            idx += 1;
                            continue;
                        }
                    } // end extended term check
                    memcpy(&col_vec[0], row_start, width);
                    get_column_vec(row_start, &col_vec[0], width, 
                                   &term->indices[0], &term->values[0], weight);
                    bin_num = bin_width_to_int(&col_vec[0], width, bin_width);
                    start = bin_ranges[bin_num];
                    stop = bin_ranges[bin_num+1];
                    col_idx = col_index(start, stop, &col_vec[0], &subspace[0], width);
                    col_check = 1;
                    // If column has been found
                    if(col_idx < MAX_SIZE_T)
                    {
                        col_found = 1;
                        val += compute_element_vec(row_start, &col_vec[0], width,
                                                   &term->indices[0], &term->values[0], term->coeff,
                                                   weight);
                    }
                } // end if !col_check
                idx += 1; // move onto the next term
            } // end while loop over all terms in group
            if(val != 0.0)
            {
                nnz += 1;
                if(compute_values)
                {
                    indices[elem_start+elem_offset] = col_idx;
                    data[elem_start+elem_offset] = val;
                    elem_offset += 1;
                }
            }
        } // end while loop over operator terms
        // Add nnz to indptr for this row
        if(!compute_values)
        {
            indptr[kk] = nnz;
        }
    } // end for-loop over all rows (end omp loop)

    // Done all rows so cummulate for correct indptr if doing structure only
    if(!compute_values)
    {
        _sum = 0;
        for(kk=0; kk < subspace_dim+1; kk++)
        {
            temp = _sum + indptr[kk];
            indptr[kk] = _sum;
            _sum = temp;
        }
    }
} // End
