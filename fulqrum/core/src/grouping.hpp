/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "base.hpp"
#include "operators.hpp"


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
    std::size_t ii;
    OperatorTerm_t *__restrict temp_term;
    std::vector<std::size_t> weight_ptrs;
    set_offdiag_weight_ptrs(oper.terms, weight_ptrs);

    // Reset all groupings
    for(ii=0; ii< oper.terms.size(); ii++)
    {
        temp_term = &oper.terms[ii];
        temp_term->group = 0; // diagonals are group 0 by convention
        if(temp_term->offdiag_weight > 0)
        {
            temp_term->group = -1;
        }
    } // end reset

    if(!weight_ptrs.size()) // return if no off-diagonal terms are present
    {
        oper.sorted = 1;
        return;
    }
    unsigned int step_size = max_offdiag_ptr_size(&weight_ptrs[0], weight_ptrs.size());
    std::ptrdiff_t dist;
    std::size_t num_terms = oper.terms.size();
    #pragma omp parallel for schedule(dynamic) if(num_terms > 1024)
    for(ii=0; ii < weight_ptrs.size()-1; ii++)
    {
        std::size_t start = weight_ptrs[ii];
        std::size_t stop = weight_ptrs[ii+1];
        int group_idx = ii*step_size;
        std::size_t kk, ll, idx;
        OperatorTerm_t *__restrict term;
        OperatorTerm_t *__restrict term2;
        std::vector<unsigned int>::iterator inds_it;
        int match;
        std::size_t ind_size;

        for(kk=start; kk < stop; kk++){
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
        // sort by group index within the start and stop indices
        std::sort(&oper.terms[start], &oper.terms[stop], group_comp);
    } // end ii loop
    
     // relabel groups into continuous integers
    int current_group=1, current_idx=1, next_idx = 2;
    for(ii=0; ii < oper.terms.size(); ii++)
    {
        if(oper.terms[ii].group == 0) // diagonal term
        {
            continue;
        }
        if(oper.terms[ii].group > current_group)
        {
            current_group = oper.terms[ii].group;
            current_idx = next_idx;
            oper.terms[ii].group = current_idx;
            next_idx += 1;
        }
        else{
            oper.terms[ii].group = current_idx;
        }
    }
    // set the grouping flag for the operator
    oper.sorted = 1;
}


/**
 * Compute the ladder indices for the first term in a group and add it to the group
 * ladder indices vector
 *
 * @param term Operator term
 * @param ladder_inds Pre-sized array (size=off-diag weight) to store indices in
 * @param ladder_width Number of elements to consider for appending
 * 
 */
inline void compute_term_ladder_inds(const OperatorTerm_t& term, 
                                      unsigned int * ladder_inds, 
                                      unsigned int ladder_width)
{
    unsigned int kk, counter = 1;
    for(kk=0; kk < term.indices.size(); kk++)
    {
        if(counter > ladder_width)
        {
            break;
        }

        if(term.values[kk] > 4)
        {
            
            ladder_inds[kk] = term.indices[kk];
            counter += 1;
        }
    }
}


inline void sort_groups_by_ladder_int(QubitOperator_t& oper,
                                      std::size_t * group_ptrs,
                                      unsigned int num_groups,
                                      unsigned int ladder_width)
    {
        
        unsigned int kk;
        #pragma omp parallel for if(num_groups > 128)
        for(kk=0; kk < num_groups; kk++)
        {
            std::sort(&oper.terms[group_ptrs[kk]], &oper.terms[group_ptrs[kk+1]], [=](const OperatorTerm_t& a, OperatorTerm_t& b)
                                  {
                                    unsigned int res_a, res_b;  
                                    res_a = term_ladder_int(a, ladder_width);
                                    res_b = term_ladder_int(b, ladder_width);
                                    return res_a < res_b;
                                  });
        }
    }
