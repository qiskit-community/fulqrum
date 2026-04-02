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

// Reverse mask for marking terms as extended or not
// Z, 0, 1, X, Y, -, +
const int REV_EXT_MASK[7] = {1, 0, 0, 1, 1, 0, 0};



/**
 * Find max. number of elements with same off-diag weight
 *
 * Used for offsetting the group counter for parallel execution
 *
 * @param vec Vector of off-diagonal pointers
 *
 * @returns Unsigned int for max. number of terms
 *
 */
unsigned int max_offdiag_ptr_size(std::size_t* vec, std::size_t size)
{
    std::size_t kk;
    unsigned int temp, max = 0;
    for(kk = 0; kk < size - 1; kk++)
    {
        temp = vec[kk + 1] - vec[kk];
        if(temp > max)
        {
            max = temp;
        }
    }
    return max;
}

/**
 * In-place marks a term as extended or not
 *
 * @param term Hamiltonian term
 *
 */
void set_extended_flag(OperatorTerm_t& term)
{
    std::size_t kk;
    int out = 1;
    for(kk = 0; kk < term.values.size(); kk++)
    {
        out *= REV_EXT_MASK[term.values[kk]];
    }
    term.extended = (!out);
}

/**
 * In-place set off-diagonal weight and real_phase
 *
 * @param term Hamiltonian term
 *
 */
void set_offdiag_weight(OperatorTerm_t& term)
{
    std::size_t kk;
    unsigned int weight = 0;
    unsigned int temp, num_y = 0;
    unsigned char* values = &term.values[0];
    for(kk = 0; kk < term.values.size(); kk++)
    {
        weight += (values[kk] > 2);
        num_y += (values[kk] == 4);
    }
    term.offdiag_weight = weight;
    // Do the real_phase for checking if operator itself can be cast as symmetric (real)
    temp = num_y % 4;
    if(temp)
    {
        term.real_phase = (temp % 2) - 1;
    }
}


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
