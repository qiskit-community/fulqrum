/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <cstddef>
#include <complex>
#include <vector>
#include <iostream>
#include <typeinfo>

#include "base.hpp"
#include "elements.hpp"
#include "bitset_hashmap.hpp"
#include <boost/dynamic_bitset.hpp>


template <typename T> void compute_diag_vector(const bitset_map_namespace::BitsetHashMapWrapper& data,
                                               T * __restrict diag_vec,
                                               const QubitOperator_t& diag_oper,
                                               const unsigned int width,
                                               const std::size_t subspace_dim){
        std::size_t kk;
        const std::size_t num_terms = diag_oper.terms.size();
        const auto* bitsets = data.get_bitsets();

        #pragma omp parallel for if(subspace_dim > 100)
        for(kk=0; kk < subspace_dim; kk++){
            std::size_t ll;
            unsigned int weight;
            T val = 0;
            OperatorTerm_t term;
            for(ll=0; ll < num_terms; ll++){
                term = diag_oper.terms[ll];
                weight = term.indices.size();
                accum_element(bitsets[kk].first, bitsets[kk].first,
                              &term.indices[0], &term.values[0], term.coeff, term.real_phase,
                              weight, val);
                }
            diag_vec[kk] = val;
            }
    }
