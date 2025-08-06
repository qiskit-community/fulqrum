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
 * Compute an integer value from the off-diagonal structure of a term
 *
 * @param term The term
 * 
 * @return Structure value
 */
std::size_t term_offdiag_structure(const OperatorTerm_t& term)
{
    std::size_t kk;
    std::size_t out = 0;
    for(kk=0; kk < term.values.size(); ++kk)
    {
        out += term.indices[kk] * (term.values[kk] > 2);
    }
    return out;
}


/**
 * Comparator for off-diagonal grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 * 
 * @return comparitor value
 */
int offdiag_comp(const OperatorTerm_t& term1, const OperatorTerm_t& term2){
    return term_offdiag_structure(term1) < term_offdiag_structure(term2);
}


/**
 * Sort terms in operator by their off-diagonal structure value
 *
 * @param terms Vector of operator terms
 *
 */
void term_offdiag_sort(std::vector<OperatorTerm_t>& terms)
{
    std::sort(terms.begin(), terms.end(), offdiag_comp);
}


unsigned int _max_offdiag_group_size(std::size_t * __restrict ptrs, std::size_t num_elems)
{
    std::size_t kk, max_size = 0;
    for(kk=0; kk < num_elems-1; kk++)
    {
        if((ptrs[kk+1]-ptrs[kk]) > max_size)
        {
            max_size = (ptrs[kk+1]-ptrs[kk]);
        }
    }
    return static_cast<unsigned int>(max_size);
}


/**
 * Comparitor for term grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 * 
 * @return comparitor value
 */
int group_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
    return term1.group < term2.group;
}


void term_group_sort(std::vector<OperatorTerm_t>& terms, std::size_t * __restrict weight_ptrs, std::size_t len_ptrs, 
                     unsigned int max_group_size)
{
    std::size_t ii;
    // Reset all groupings
    for(ii=0; ii < terms.size(); ii++)
    {
        terms[ii].group = 0; // diagonals are group 0 by convention
        if(terms[ii].offdiag_weight > 0)
        {
            terms[ii].group = -1;
        }
    } // end reset

    std::ptrdiff_t dist;
    //#pragma omp parallel for schedule(dynamic) if(num_terms > 1024)
    for(ii=0; ii < len_ptrs-1; ii++)
    {
        std::size_t start = weight_ptrs[ii];
        std::size_t stop = weight_ptrs[ii+1];
        int group_idx = ii*max_group_size;
        std::size_t kk, ll, idx;
        OperatorTerm_t * term;
        OperatorTerm_t * term2;
        std::vector<unsigned int>::iterator inds_it;
        int match;
        std::size_t ind_size;
        if(terms[start].group == 0) // group is a diagonal group
        {
            continue;
        }

        for(kk=start; kk < stop; kk++)
        {
            term = &terms[kk];
            ind_size = term->indices.size();
            if(term->group < 0) // term is not touched yet
            {
                group_idx += 1; // diags are group zero, so go to 1 first
                term->group = group_idx;
            }
            // Loop over all terms from kk+1 on up t ostop
            for(ll=kk+1; ll<stop; ll++)
            {
                term2 = &terms[ll];
                // term2 is not matched and number of off-diag ops is equal
                if((term2->group < 0) && (term2->offdiag_weight == term->offdiag_weight))
                {
                    match = 1;
                    for(idx=0; idx<ind_size; idx++)
                    {
                        // found off-diag term at idx
                        if(term->values[idx] > 2)
                        {
                            // Tell me if the index is also found in term2
                            inds_it = std::find(term2->indices.begin(),
                                        term2->indices.end(), term->indices[idx]);
                            if(inds_it == term2->indices.end())
                            {
                                match = 0;
                                break;
                            }
                            // if the index is in term2, find out its location and check for off-diag there
                            else
                            {
                                dist = std::distance(term2->indices.begin(), inds_it);
                                if(!(term2->values[dist] > 2))
                                {
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
        std::sort(&terms[start], &terms[stop], group_comp);
    } // end ii loop
    
     // relabel groups into continuous integers
    int current_group=1, current_idx=1, next_idx = 2;
    for(ii=0; ii < terms.size(); ii++)
    {
        if(terms[ii].group == 0) // diagonal term
        {
            continue;
        }
        if(terms[ii].group > current_group)
        {
            current_group = terms[ii].group;
            current_idx = next_idx;
            terms[ii].group = current_idx;
            next_idx += 1;
        }
        else{
            terms[ii].group = current_idx;
        }
    }
}