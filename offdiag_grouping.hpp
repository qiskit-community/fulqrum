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


size_t term_offdiag_structure(const OperatorTerm_t& term)
{
    std::size_t kk;
    std::size_t out = 0;
    for(kk=0; k < term.values.size(); ++kk)
    {
        out += term.indices[kk] * (term.values[kk] > 2);
    }
    return out;
}

/**
 * Comparitor for term grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 * 
 * @return comparitor value
 */
int offdiag_comp(const OperatorTerm_t& term1, const OperatorTerm_t& term2){
    return term_offdiag_structure(term1) < term_offdiag_structure(term2);
}


void offdiag_term_sort(const std::vector<OperatorTerm_t>& terms)
{
    std::sort(terms.begin(), terms.end(), offdiag_comp);
}