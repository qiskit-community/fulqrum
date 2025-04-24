# Fulqrum
# Copyright (C) 2024, IBM
cimport cython
from libcpp.vector cimport vector
from fulqrum.core.spmv cimport FulqrumSpMV
from libc.string cimport memcpy 

import numpy as np
import scipy.sparse as sp

include "includes/base_header.pxi"
include "includes/bitstrings_header.pxi"
include "includes/elements_header.pxi"
include "includes/operators_header.pxi"



cdef void csr_matrix_builder(QubitOperator_t& ham,
                              vector[unsigned char]& subspace,
                              double complex * diag_vec,
                              size_t width,
                              size_t subspace_dim,
                              int has_nonzero_diag,
                              size_t bin_width,
                              size_t * bin_ranges,
                              size_t * group_ptrs,
                              size_t num_groups,
                              fused_int * indptr,
                              fused_int * indices,
                              double complex * data,
                              int compute_values) noexcept nogil:

    cdef size_t kk, idx;
    cdef size_t group_start, group_stop, group;
    cdef size_t start, stop;
    cdef fused_int temp, _sum;
    cdef size_t num_terms = ham.terms.size();
    cdef OperatorTerm_t * terms = &ham.terms[0];
    cdef fused_int row_nnz;
    cdef OperatorTerm_t * term;
    cdef vector[unsigned char] col_vec;
    cdef size_t weight, col_idx;
    cdef double complex val;
    cdef const unsigned char * row_start;
    cdef int do_col_search;
    cdef fused_int elem_start;
    cdef size_t bin_num;
    
    col_vec.resize(width);
    for kk in range(subspace_dim):
        row_start = &subspace[kk*width];
        row_nnz = 0;
        elem_start = indptr[kk];
        # do diagonal first, if any
        if(has_nonzero_diag):
            if(diag_vec[kk] != 0.0):
                if compute_values:
                    indices[elem_start+row_nnz] = kk;
                    data[elem_start+row_nnz] = diag_vec[kk];
                row_nnz += 1;
        for group in range(num_groups):
            group_start = group_ptrs[group];
            group_stop = group_ptrs[group+1];
            do_col_search = 1;
            val = 0;
            for idx in range(group_start, group_stop):
                term = &ham.terms[idx];
                weight = term.indices.size();
                if(term.extended):
                    if(not nonzero_extended_value(term, row_start, width)):
                        continue;
                if(do_col_search):
                    memcpy(&col_vec[0], row_start, width);
                    get_column_vec(row_start, &col_vec[0], width, 
                                   &term.indices[0], &term.values[0], weight);
                    bin_num = bin_width_to_int(&col_vec[0], width, bin_width);
                    start = bin_ranges[bin_num];
                    stop = bin_ranges[bin_num+1];
                    col_idx = col_index(start, stop, &col_vec[0], &subspace[0], width);

                if(col_idx < MAX_SIZE_T):
                    do_col_search = 0; # do not search again for this group
                    val += compute_element_vec(row_start, &col_vec[0], width,
                                               &term.indices[0], &term.values[0],
                                               term.coeff, weight);
                else:
                    break;
            if(val!=0):
                if(compute_values):
                    indices[elem_start+row_nnz] = col_idx;
                    data[elem_start+row_nnz] = val;
                row_nnz += 1;
        if(not compute_values):
            # done with row, add row_nnz to indptr
            indptr[kk] = row_nnz;
    
    if(not compute_values):
        # Done all rows so cummulate for correct indptr structure
        _sum = 0;
        for kk in range(subspace_dim+1):
            temp = _sum + indptr[kk];
            indptr[kk] = _sum;
            _sum = temp;
