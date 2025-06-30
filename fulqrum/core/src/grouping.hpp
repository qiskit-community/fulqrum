/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <complex>
#include <vector>
#include <algorithm>
#include "base.hpp"


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
    OperatorTerm_t *__restrict term;
    OperatorTerm_t *__restrict term2;
    std::size_t ii, kk, ll, idx;
    std::size_t ind_size;
    std::vector<unsigned int>::iterator inds_it;
    int match;

    // Reset all groupings
    for(ii=0; ii< oper.terms.size(); ii++)
    {
        term = &oper.terms[ii];
        term->group = 0; // diagonals are group 0 by convention
        if(term->offdiag_weight > 0)
        {
            term->group = -1;
        }
    } // end reset

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
                            if(!(term2->values[dist] > 2)){
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
