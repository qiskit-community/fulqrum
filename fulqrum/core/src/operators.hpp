/**
 * This code is part of Fulqrum.
 *
 * (C) Copyright IBM 2024.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */
#pragma once
#include "base.hpp"
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <vector>




inline unsigned int term_ladder_int(const OperatorTerm_t& term, unsigned int ladder_width)
{
    unsigned int subset = 0;
    unsigned int kk, counter = 0;
    for(kk = 0; kk < term.indices.size(); kk++)
    {
        if(term.values[kk] > 4)
        {
            subset = subset | ((unsigned int)term.values[kk] - 5U) << counter;
            counter += 1;
        }
    }
    if(counter < ladder_width)
    {
        ladder_width = counter;
    }
    if(!counter)
    {
        subset = MAX_UINT;
    }
    else
    {
        subset = subset & ((1U << ladder_width) - 1U);
    }
    return subset;
}

void offdiag_weight_sort(QubitOperator_t& oper)
{
    // sort by group index
    std::sort(oper.terms.begin(), oper.terms.end(), offweight_comp);
    oper.off_weight_sorted = 1;
    oper.weight_sorted = 0;
}
