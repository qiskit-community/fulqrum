/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "base.hpp"

// Z, 0, 1, X, Y, -, +
const int REV_EXT_MASK[7] = {1, 0, 0, 1, 1, 0, 0};


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
    std::vector<std::size_t>::iterator inds_it;
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

    std::ptrdiff_t dist;
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
            // term2 is not matched and number of off-diag ops is equal
            if((term2->group < 0) && (term2->offdiag_weight == term->offdiag_weight)){
                match = 1;
                for(idx=0; idx<ind_size; idx++){
                    // found off-diag term at idx
                    if(term->values[idx] > 2){
                        // Tell me if the index is alsco found in term2
                        inds_it = std::find(term2->indices.begin(),
                                       term2->indices.end(), term->indices[idx]);
                        if(inds_it == term2->indices.end()){
                            match = 0;
                            break;
                        }
                        // if the index is in term2, find out its location and check for off-diag there
                        else{
                            dist = std::distance(term2->indices.begin(), inds_it);
                            if(not (term2->values[dist] > 2)){
                                match = 0;
                                break;
                            }
                        }
                    } // end found off-diag term
                } // end idx for-loop
                if(match){ // If match
                    term2->group = group_idx;
                }
            } // end non-id match
        } // end ll for-loop
    } // end kk for-loop

    // sort by group index
    std::sort(oper.terms.begin(), oper.terms.end(), group_comp);
}


/**
 * In-place marks a term as extended or not
 *
 * @param term Hamiltonian term
 * 
 */
void set_extended_flag(OperatorTerm_t& term){
    std::size_t kk;
    int out = 1;
    for(kk=0; kk < term.values.size(); kk++){
        out *= REV_EXT_MASK[term.values[kk]];
    }
    term.extended = (not out);
}
