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
#include <vector>

template <typename T>
inline void set_weight_ptrs(std::vector<T>& __restrict terms,
                            std::vector<std::size_t>& vec)
{
    vec.resize(0);
    vec.push_back(0);
    std::size_t kk;
    std::size_t val = terms[0].indices.size();
    for(kk = 1; kk < terms.size(); kk++)
    {
        if(terms[kk].indices.size() > val)
        {
            vec.push_back(kk);
            val = terms[kk].indices.size();
        }
    }
    vec.push_back(terms.size());
}


/**
 * Combine repeated terms that represent same
 * operators, dropping terms smaller than requested tolerance.
 *
 * Input terms must be sorted by weight before calling this routine
 *
 * @param[in] terms Terms for input operator
 * @param[in] out_terms Terms for output operator (to push_back to)
 * @param[in] touched pointer array indicating if term has been touched
 * @param[in] num_terms Number of terms in input operator
 * @param[in] atol Absolute tolerance for term truncation
 *
 */
template <typename T>
inline void combine_terms(std::vector<T>& __restrict terms,
                          std::vector<T>& __restrict out_terms,
                                width_t* touched,
                                double atol)
{
    std::size_t kk, qq, num_terms = terms.size();
    std::vector<std::size_t> weight_ptrs;
    set_weight_ptrs(terms, weight_ptrs);
    std::vector<std::vector<T>> temp_terms;
    temp_terms.resize(weight_ptrs.size() - 1);
    // do sort over each collection of terms with same weight
#pragma omp parallel for schedule(dynamic) if(num_terms > 1024)
    for(kk = 0; kk < weight_ptrs.size() - 1; kk++)
    {
        std::size_t jj, mm, pp;
        std::size_t start, stop;
        T target_term;
        T* current_term;
        int do_combine;
        // set start and stop for terms of the same weight
        start = weight_ptrs[kk];
        stop = weight_ptrs[kk + 1];
        for(jj = start; jj < stop; jj++)
        {
            if(touched[jj]) // If touched, move onto next term
            {
                continue;
            }
            touched[jj] = 1;
            target_term = terms[jj];
            for(mm = jj + 1; mm < stop; mm++)
            {
                if(touched[mm])
                {
                    continue;
                }
                current_term = &terms[mm];
                // filter if offdiag weights differ
                if(target_term.offdiag_weight != current_term->offdiag_weight)
                {
                    continue;
                }

                do_combine = 1;
                // look to see if indices and values match
                for(pp = 0; pp < target_term.indices.size(); pp++)
                {
                    if((target_term.indices[pp] != current_term->indices[pp]) ||
                       (target_term.values[pp] != current_term->values[pp]))
                    {
                        do_combine = 0;
                        break;
                    }
                }
                if(do_combine)
                {
                    touched[mm] = 1;
                    target_term.coeff += current_term->coeff;
                }
            } // end mm for-loop
            // Add term to output if either real or imag parts are greater than atol
            if(std::abs(target_term.coeff) > atol)
            {
                temp_terms[kk].push_back(target_term);
            }
        } // end main jj loop
    } //end kk-loop

    // at end of all, add to output terms
    for(kk = 0; kk < weight_ptrs.size() - 1; kk++)
    {
        for(qq = 0; qq < temp_terms[kk].size(); qq++)
        {
            out_terms.push_back(temp_terms[kk][qq]);
        }
    }

} // end combine_terms