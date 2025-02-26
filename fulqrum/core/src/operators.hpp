/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "base.hpp"

/**
 * Sorting of indices and values for Operator term data
 *
 * @param inds The term indices (qubits) array
 * @param vals The term values (operators) array
 */
void sort_term_data(std::vector<std::size_t>& inds, std::vector<unsigned char>& vals) {
    std::size_t n = inds.size();
    for (std::size_t i = 1; i < n; i++) {
        std::size_t key = inds[i];
        char val = vals[i];
        std::size_t j = std::lower_bound(inds.begin(), inds.begin() + i, key) - inds.begin();
        
        for (std::size_t k = i; k > j; k--) {
            inds[k] = inds[k-1];
            vals[k] = vals[k-1];
        }
        inds[j] = key;
        vals[j] = val;
    }
}

/**
 * Comparitor for term grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 * 
 * @return comparitor value
 */
int group_comp(OperatorTerm_t& term1, OperatorTerm_t& term2){
    return term1.group < term2.group;
}


/**
 * In-place term sorting by off-diagonal structure
 *
 * @param oper Hamiltonian operator
 * 
 */
void offdiag_term_sort(QubitOperator_t& oper){
    OperatorTerm_t * term;
    OperatorTerm_t * term2;
    std::size_t kk, ll, idx;
    std::size_t ind_size;
    int match;

    // Reset all groupings
    for(kk=0; kk< oper.terms.size(); kk++){
        term = &oper.terms[kk];
        term->group = 0;
        for(ll=0; ll < term->values.size(); ll++){
            if(term->values[ll] > 2){
                term->group = -1;
                break;
            }
        }
    } // end for-loop

    int group_idx = 0;
    for(kk=0; kk < oper.terms.size(); kk++){
        term = &oper.terms[kk];
        ind_size = term->indices.size();
        if(term->group < 0){
            group_idx += 1; // diags are group zero, so go to 1 first
            term->group = group_idx;
        }
        // Loop over all terms from kk+1 on up
        for(ll=kk+1; ll<oper.terms.size(); ll++){
            term2 = &oper.terms[ll];
            // term2 is not matched and number of non-id operators match
            if((term2->group < 0) && (ind_size == term2->indices.size())){
                match = 1;
                for(idx=0; idx<ind_size; idx++){
                    // mis-match between indices
                    if(term->indices[idx] != term2->indices[idx]){
                        match = 0;
                        break;
                    }
                    // mismatch between off-diag structures
                    if((term->values[idx] > 2) != (term2->values[idx] > 2)){
                        match = 0;
                        break;
                    }
                } // end idx for-loop
                if(match){ // If match
                    term2->group = group_idx;
                }
            } // end non-id match
        } // end ll for-loop
    } // end kk for-loop

    // sort by group index
    sort(oper.terms.begin(), oper.terms.end(), group_comp);
}
