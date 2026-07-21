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
#include "constants.hpp"
#include <boost/sort/pdqsort/pdqsort.hpp>
#include <vector>

/**
 * Compute an integer value from the off-diagonal structure of a term
 *
 * @param term The term
 *
 * @return Structure value
 */
template <typename T>
inline std::size_t term_offdiag_structure(const T& term)
{
    std::size_t kk;
    std::size_t out = 0;
#pragma omp simd reduction(+ : out)
    for(kk = 0; kk < term.values.size(); ++kk)
    {
        out += (term.indices[kk] + 1) * (term.values[kk] > 2);
    }
    return out;
}

template <typename T>
inline void set_weight_ptrs(std::vector<T>& __restrict terms, std::vector<std::size_t>& vec)
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
 * Set the pointers for the off-diagonal structure
 *
 * @param terms Operator terms
 * @param vec Vector to add pointers to
 *
 */
template <typename T>
inline void set_offdiag_structure_ptrs(const std::vector<T>& __restrict terms,
                                       std::vector<std::size_t>& vec)
{
    vec.resize(0);
    std::size_t kk;
    if(terms.size())
    {
        std::size_t val = term_offdiag_structure(terms[0]);
        vec.push_back(0);
        for(kk = 1; kk < terms.size(); kk++)
        {
            if(term_offdiag_structure(terms[kk]) > val)
            {
                vec.push_back(kk);
                val = term_offdiag_structure(terms[kk]);
            }
        }
        vec.push_back(terms.size());
    }
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
                          std::vector<std::size_t>& __restrict sort_ptrs,
                          double atol)
{
    std::size_t kk, num_terms = terms.size();
// do sort over each collection of terms with same weight
#pragma omp parallel for if(num_terms > 4096) schedule(guided)
    for(kk = 0; kk < sort_ptrs.size() - 1; kk++)
    {
        std::size_t jj, mm, qq;
        std::size_t start, stop, sub_start, sub_stop;
        T target_term;
        T* current_term;
        // set start and stop for terms based on ptrs
        start = sort_ptrs[kk];
        stop = sort_ptrs[kk + 1];
        std::vector<T> temp_terms;
        temp_terms.reserve(stop - start);
        std::vector<std::size_t> new_ptrs;
        new_ptrs.push_back(start);
        unsigned int val;
        // If the number of terms in the bucket is greater than this value then
        // do an additional sort based on the projector structure to make smaller buckets
        // to search over
        if(stop - start > 10000)
        {
            boost::sort::pdqsort(terms.begin() + start, terms.begin() + stop, [](T term1, T term2) {
                return term1.proj_structure < term2.proj_structure;
            });
            val = terms[start].proj_structure;
            for(jj = start + 1; jj < stop; jj++)
            {
                if(terms[jj].proj_structure > val)
                {
                    new_ptrs.push_back(jj);
                    val = terms[jj].proj_structure;
                }
            }
        }
        new_ptrs.push_back(stop);
        for(jj = 0; jj < new_ptrs.size() - 1; jj++)
        {
            sub_start = new_ptrs[jj];
            sub_stop = new_ptrs[jj + 1];
            std::vector<bool> touched(sub_stop - sub_start, false);
            for(qq = 0; qq < (sub_stop - sub_start); qq++)
            {
                if(touched[qq]) // If touched, move onto next term
                {
                    continue;
                }
                touched[qq] = true;
                T target_term = terms[qq + sub_start];
                const std::size_t target_size = target_term.indices.size();
                for(mm = qq + 1; mm < (sub_stop - sub_start); mm++)
                {
                    if(touched[mm])
                    {
                        continue;
                    }
                    current_term = &terms[mm + sub_start];
                    // move on if number of indices does not match
                    if(target_size != current_term->indices.size())
                    {
                        continue;
                    }
                    // look to see if indices and values match
                    if((target_term.indices == current_term->indices) &&
                       (target_term.values == current_term->values))
                    {
                        touched[mm] = true;
                        target_term.coeff += current_term->coeff;
                    }
                } // end mm for-loop
                // Add term to output if either real or imag parts are greater than atol
                if(std::abs(target_term.coeff) > atol)
                {
                    temp_terms.push_back(std::move(target_term));
                }
            }
        } // end main jj loop
#pragma omp critical
        {
            out_terms.insert(out_terms.end(),
                             std::make_move_iterator(temp_terms.begin()),
                             std::make_move_iterator(temp_terms.end()));
        }
    } //end kk-loop

} // end combine_terms
