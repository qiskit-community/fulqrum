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
#include "operators.hpp"
#include <algorithm>
#include <cstdlib>
#include <vector>





void ladder_bin_starts(const std::vector<OperatorTerm_t>& terms,
                       const std::size_t* group_ptrs,
                       unsigned int* group_counts,
                       std::size_t* group_ranges,
                       unsigned int num_groups,
                       unsigned int num_ladder_bins,
                       unsigned int ladder_width)
{

    std::size_t start, stop, kk, mm;
    unsigned int term_int;
    std::size_t total;
    total = 0;
    for(kk = 0; kk < num_groups; kk++)
    {
        start = group_ptrs[kk];
        stop = group_ptrs[kk + 1];

        for(mm = start; mm < stop; mm++)
        {
            term_int = term_ladder_int(terms[mm], ladder_width);
            group_counts[kk * num_ladder_bins + term_int] += 1;
        }
        group_ranges[kk * num_ladder_bins] = total;
        for(mm = 1; mm < num_ladder_bins + 1; mm++)
        {
            total += group_counts[kk * num_ladder_bins + mm - 1];
            group_ranges[kk * num_ladder_bins + mm] = total;
        }
    }
}
