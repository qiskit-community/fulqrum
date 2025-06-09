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

// Reverse mask for marking terms as extended or not
// Z, 0, 1, X, Y, -, +
const int REV_EXT_MASK[7] = {1, 0, 0, 1, 1, 0, 0};




/**
 * Sorting of indices and values for Operator term data
 *
 * @param inds The term indices (qubits) array
 * @param vals The term values (operators) array
 */
void sort_term_data(std::vector<unsigned int>& inds, std::vector<unsigned char>& vals) {
    std::size_t n = inds.size();
    for (std::size_t i = 1; i < n; i++) {
        unsigned int key = inds[i];
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
    OperatorTerm_t *__restrict term;
    OperatorTerm_t *__restrict term2;
    std::size_t kk, ll, idx;
    std::size_t ind_size;
    std::vector<unsigned int>::iterator inds_it;
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
    term.extended = (!out);
}


/**
 * In-place set off-diagonal weight
 *
 * @param term Hamiltonian term
 * 
 */
 void set_offdiag_weight(OperatorTerm_t& term){
    std::size_t kk;
    std::size_t weight = 0;
    unsigned char * values = &term.values[0];
    for(kk=0; kk < term.values.size(); kk++){
        weight += (values[kk] > 2);
    }
    term.offdiag_weight = weight;
}


/**
 * Combine repeated terms that represent same
 * operators, dropping terms smaller than requested tolerance.
 *
 * @param terms Terms for input operator
 * @param out_terms Terms for ouput operator (to push_back to)
 * @param touched pointer array indicating if term has been touched
 * @param num_terms Number of terms in input operator
 * @param atol Absolute tolerance for term truncation
 * 
 */
void combine_qubit_terms(std::vector<OperatorTerm_t>&__restrict terms,
                         std::vector<OperatorTerm_t>&__restrict out_terms,
                         unsigned char * touched,
                         std::size_t num_terms,
                         double atol)
{
    std::size_t kk, jj, mm;
    OperatorTerm_t target_term;
    OperatorTerm_t * current_term;
    int do_combine;
    for(kk=0; kk<num_terms; kk++)
    {
        if(touched[kk]) // If touched, move onto next term
        {
            continue;
        }
        touched[kk] = 1;
        target_term = terms[kk];
        for(jj=kk+1; jj < num_terms; jj++)
        {
            if(touched[jj])
            {
                continue;
            }
            current_term = &terms[jj];
            // filter if offdiag weights differ
            if(target_term.offdiag_weight != current_term->offdiag_weight)
            {
                continue;
            }
            // filter if weights are different
            if(target_term.indices.size() != current_term->indices.size())
            {
                continue;
            }
            do_combine = 1;
            // look to see if indices and values match
            for(mm=0; mm < target_term.indices.size(); mm++)
            {
                if((target_term.indices[mm] != current_term->indices[mm]) || (target_term.values[mm] != current_term->values[mm]))
                {
                    do_combine = 0;
                    break;
                }
            }
            if(do_combine)
            {
                touched[jj] = 1;
                target_term.coeff += current_term->coeff;
            }
        } // end jj for-loop
        // Add term to output if either real or imag parts are greater than atol
        if(std::abs(target_term.coeff) > atol)
        {
            out_terms.push_back(target_term);
        }
    } // end main kk loop
} // end combine_qubit_terms



inline unsigned int term_ladder_int(const OperatorTerm_t& term, unsigned int num_bits)
{
    unsigned int subset = 0;
    unsigned int kk, counter = 0;
    for(kk=0; kk < term.indices.size(); kk++)
    {
        if(term.values[kk] > 4)
        {
            subset = subset | ((unsigned int)term.values[kk]-5U) << counter;
            counter += 1;
            if(counter == num_bits)
            {
                break;
            }
        }
    }
    return subset;
}