/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <cstddef>
#include <complex>

#include "base.hpp"
#include "elements.hpp"


void compute_diag_vector(const unsigned char * __restrict data,
                         std::complex<double> * __restrict diag_vec,
                         QubitOperator_t& diag_oper,
                         std::size_t width,
                         std::size_t subspace_dim){
        std::size_t kk;
        const std::size_t num_terms = diag_oper.terms.size();
        #pragma omp parallel for if(subspace_dim > 100)
        for(kk=0; kk < subspace_dim; kk++){
            std::size_t ll, weight, row_start;
            std::complex<double> val = 0;
            OperatorTerm_t * term;
            row_start = width*kk;
            for(ll=0; ll < num_terms; ll++){
                term = &diag_oper.terms[ll];
                weight = term->indices.size();
                compute_element_vec(&data[row_start], &data[row_start], width,
                                           &term->indices[0],
                                           &term->values[0],
                                           term->coeff, weight, val);
                }
            diag_vec[kk] = val;
            }
    }
