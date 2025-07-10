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
 * Compute the offdiag indices for the first term in a group and add it to the group
 * offdiag indices vector
 *
 * @param term Operator term
 * @param ladder_inds Pre-sized array (size=num_inds) to store indices in
 * @param num_inds Number of elements to consider for appending
 * 
 */
inline void compute_term_offdiag_inds(const OperatorTerm_t& term, 
                                     unsigned int * offdiag_inds, 
                                     unsigned int num_inds)
{
    unsigned int kk, counter = 1;
    for(kk=0; kk < term.indices.size(); kk++)
    {
        if(counter > num_inds)
        {
            break;
        }

        if(term.values[kk] > 2)
        {
            offdiag_inds[kk] = term.indices[kk];
            counter += 1;
        }
    }
}


/**
 * Set the offdiag indices for each group in a off-diagonal type=2 Hamiltonian
 *
 * @param terms Operator terms
 * @param group_indices Vector of vectors of group_indices
 * @param group_ptrs Pointer of array of group pointers
 * @param num_groups Number of groups = len(group_ptrs) - 1
 * @param ladder_width Target ladder indices width for type=2 operators
 * @param oper_type Type of operator, 1 or 2
 * 
 */
void set_group_offdiag_indices(const std::vector<OperatorTerm_t>& terms,
                              std::vector<std::vector<unsigned int>>& group_indices,
                              const std::size_t * group_ptrs,
                              unsigned int num_groups,
                              unsigned int ladder_width, int oper_type)
{
    unsigned int kk;
    unsigned int inds_len;
    group_indices.resize(num_groups);
    for(kk=0; kk<num_groups; kk++)
    {
        if(oper_type==2)
        {
            inds_len = std::min(terms[group_ptrs[kk]].offdiag_weight, ladder_width);
        }
        else
        {
            inds_len = terms[group_ptrs[kk]].offdiag_weight;
        }
        group_indices[kk].resize(inds_len);
        compute_term_offdiag_inds(terms[group_ptrs[kk]], &(group_indices[kk])[0], inds_len);
    }
}



inline void sort_groups_by_ladder_int(QubitOperator_t& oper,
                                      const std::size_t * group_ptrs,
                                      unsigned int num_groups,
                                      unsigned int ladder_width)
    {
        
        unsigned int kk;
        std::size_t start, stop;
        #pragma omp parallel for if(num_groups > 128)
        for(kk=0; kk < num_groups; kk++)
        {
            
            start = group_ptrs[kk];
            stop = group_ptrs[kk+1];
            if(!oper.terms[start].group)
            {
                continue;
            }
            std::sort(&oper.terms[start], &oper.terms[stop], [=](const OperatorTerm_t& a, OperatorTerm_t& b)
                                  {
                                    unsigned int res_a, res_b;  
                                    res_a = term_ladder_int(a, ladder_width);
                                    res_b = term_ladder_int(b, ladder_width);
                                    return res_a < res_b;
                                  });
        }
    }


void ladder_bin_starts(const OperatorTerm_t * terms, const std::size_t * group_ptrs,
                        unsigned int * group_counts, unsigned int * group_ranges,
                        unsigned int num_groups, unsigned int num_bins, unsigned int ladder_width)
{

    std::size_t start, stop, kk, mm;
    unsigned int ptr_size = num_bins + 1;
    unsigned int term_int;
    unsigned int total;
    for(kk=0; kk<num_groups; kk++)
    {
        start = group_ptrs[kk];
        stop = group_ptrs[kk+1];
        
        for(mm=start; mm < stop; mm++)
        {
            term_int = term_ladder_int(terms[mm], ladder_width);
            group_counts[kk*num_bins+term_int] += 1;
        }
        group_ranges[kk*ptr_size] = start;
        total = start + group_counts[kk*num_bins];

        for(mm=1; mm < ptr_size; mm++)
        {
            group_ranges[kk*ptr_size+mm] = total;
            if(mm != num_bins)
            {
                total += group_counts[kk*num_bins+mm];
            }
        }
    }
}