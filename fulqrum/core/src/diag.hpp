/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <cstddef>
#include <complex>
#include <vector>

#include "base.hpp"
#include "elements.hpp"
#include <boost/dynamic_bitset.hpp>


template <typename T> void compute_diag_vector(const std::vector<boost::dynamic_bitset<std::size_t> >& data,
                                               T * __restrict diag_vec,
                                               const QubitOperator_t& diag_oper,
                                               const unsigned int width,
                                               const std::size_t subspace_dim){
        std::size_t kk;
        const std::size_t num_terms = diag_oper.terms.size();
        #pragma omp parallel for if(subspace_dim > 100)
        for(kk=0; kk < subspace_dim; kk++){
            std::size_t ll;
            unsigned int weight;
            T val = 0;
            OperatorTerm_t term;
            for(ll=0; ll < num_terms; ll++){
                term = diag_oper.terms[ll];
                weight = term.indices.size();
                accum_element(data[kk], data[kk],
                              &term.indices[0], &term.values[0], term.coeff, term.real_phase,
                              weight, val);
                }
            diag_vec[kk] = val;
            }
    }
