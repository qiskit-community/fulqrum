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
        std::size_t val = terms[0].offdiag_structure;
        vec.push_back(0);
        for(kk = 1; kk < terms.size(); kk++)
        {
            if(terms[kk].offdiag_structure > val)
            {
                vec.push_back(kk);
                val = terms[kk].offdiag_structure;
            }
        }
        vec.push_back(terms.size());
    }
}

/**
 * Combine repeated terms that represent same
 * operators, dropping terms smaller than requested tolerance.
 *
 * Input terms must be sorted before calling this routine
 *
 * @param[in] terms Terms for input operator
 * @param[in] out_terms Terms for output operator (to push_back to)
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
    const std::size_t num_blocks = sort_ptrs.size() - 1;
    // Do a double sorting here so that we can iterate once through the terms
    // in each block
    auto term_data_sort = [](const T& a, const T& b) -> bool {
        if(a.indices.size() != b.indices.size())
            return a.indices.size() < b.indices.size();
        if(a.indices != b.indices)
            return a.indices < b.indices;
        return a.values < b.values;
    };
    // term data equality check
    auto term_eq = [](const T& a, const T& b) -> bool {
        return a.indices == b.indices && a.values == b.values;
    };

    std::vector<std::vector<T>> temp_results(num_blocks);

// do sort over each collection of terms with same weight
#pragma omp parallel for if(num_terms > 4096)
    for(kk = 0; kk < num_blocks; kk++)
    {
        std::size_t start, stop;
        start = sort_ptrs[kk];
        stop = sort_ptrs[kk + 1];

        std::vector<T>& temp_terms = temp_results[kk];
        temp_terms.reserve(stop - start);

        // Sort by proj_structure, if different, otherwise sort by data
        boost::sort::pdqsort(
            terms.begin() + start, terms.begin() + stop, [&](const T& a, const T& b) {
                if(a.proj_structure != b.proj_structure)
                    return a.proj_structure < b.proj_structure;
                return term_data_sort(a, b);
            });

        for(std::size_t qq = start; qq < stop;)
        {
            T accum = terms[qq];
            std::size_t next = qq + 1;
            while(next < stop && term_eq(terms[next], accum))
            {
                accum.coeff += terms[next].coeff;
                ++next;
            }
            if(std::abs(accum.coeff) > atol)
            {
                temp_terms.push_back(std::move(accum));
            }
            qq = next;
        }
    } //end kk-loop

    // merge all block results together into ouput operator terms
    std::size_t total = 0;
    for(const auto& item : temp_results)
        total += item.size();
    out_terms.reserve(out_terms.size() + total);
    for(auto& item : temp_results)
        out_terms.insert(out_terms.end(),
                         std::make_move_iterator(item.begin()),
                         std::make_move_iterator(item.end()));

} // end combine_terms