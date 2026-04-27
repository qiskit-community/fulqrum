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
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <map>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "constants.hpp"
#include "qubit_term.hpp"
#include "io.hpp"

struct QubitOperator;

// forward definitions
void set_sorting_flags(QubitOperator& oper, std::string kind);
inline void term_offdiag_sort(QubitOperator& oper);

/**
 * Comparator for weight grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int weight_comp(OperatorTerm& term1, OperatorTerm& term2)
{
    return term1.indices.size() < term2.indices.size();
}

/**
 * Comparator for off-diagonal weight grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int offweight_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
    return term1.offdiag_weight < term2.offdiag_weight;
}

inline void set_weight_ptrs(std::vector<OperatorTerm>& __restrict terms,
                            std::vector<std::size_t>& vec)
{
    vec.resize(0);
    vec.push_back(0);
    std::size_t kk;
    unsigned int val = terms[0].indices.size();
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

inline void set_group_ptrs(const std::vector<OperatorTerm>& __restrict terms,
                           std::vector<std::size_t>& vec)
{
    vec.resize(0);
    vec.push_back(0);
    std::size_t kk;
    int val = terms[0].group;
    for(kk = 1; kk < terms.size(); kk++)
    {
        if(terms[kk].group > val)
        {
            vec.push_back(kk);
            val = terms[kk].group;
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
inline void combine_qubit_terms(std::vector<OperatorTerm>& __restrict terms,
                                std::vector<OperatorTerm>& __restrict out_terms,
                                unsigned int* touched,
                                double atol)
{
    std::size_t kk, qq, num_terms = terms.size();
    std::vector<std::size_t> weight_ptrs;
    set_weight_ptrs(terms, weight_ptrs);
    std::vector<std::vector<OperatorTerm_t>> temp_terms;
    temp_terms.resize(weight_ptrs.size() - 1);
    // do sort over each collection of terms with same weight
#pragma omp parallel for schedule(dynamic) if(num_terms > 1024)
    for(kk = 0; kk < weight_ptrs.size() - 1; kk++)
    {
        std::size_t jj, mm, pp;
        std::size_t start, stop;
        OperatorTerm_t target_term;
        OperatorTerm_t* current_term;
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

} // end combine_qubit_terms

/**
 * Compute an integer value from the off-diagonal structure of a term
 *
 * @param term The term
 *
 * @return Structure value
 */
inline std::size_t term_offdiag_structure(const OperatorTerm_t& term)
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

/**
 * Comparator for off-diagonal grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int offdiag_comp(const OperatorTerm& term1, const OperatorTerm& term2)
{
    return term_offdiag_structure(term1) < term_offdiag_structure(term2);
}

/**
 * Set the pointers for the off-diagonal weights
 *
 * @param terms Operator terms
 * @param vec Vector to add pointers to
 *
 */
inline void set_offdiag_weight_ptrs(const std::vector<OperatorTerm>& __restrict terms,
                                    std::vector<std::size_t>& vec)
{
    vec.resize(0);
    std::size_t kk;
    unsigned int val = terms[0].offdiag_weight;
    vec.push_back(0);
    for(kk = 1; kk < terms.size(); kk++)
    {
        if(terms[kk].offdiag_weight > val)
        {
            vec.push_back(kk);
            val = terms[kk].offdiag_weight;
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
inline void set_offdiag_structure_ptrs(const std::vector<OperatorTerm>& __restrict terms,
                                       std::vector<std::size_t>& vec)
{
    vec.resize(0);
    std::size_t kk;
    if(terms.size())
    {
        unsigned int val = term_offdiag_structure(terms[0]);
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
 * Find max. number of elements with same off-diag weight
 *
 * Used for offsetting the group counter for parallel execution
 *
 * @param[in] vec Vector of off-diagonal pointers
 * @param[in] size Number of elements in vec
 * 
 * @returns size_t for max. number of terms
 *
 */
inline std::size_t max_offdiag_ptr_size(std::vector<std::size_t>& vec)
{
    std::size_t kk;
    std::size_t temp, max = 0;
    if(!vec.size()) // This is the case for all diagonals operator
    {
        max = 0;
    }
    else
    {
        for(kk = 0; kk < vec.size() - 1; kk++)
        {
            temp = vec[kk + 1] - vec[kk];
            if(temp > max)
            {
                max = temp;
            }
        }
    }
    return max;
}

// Reverse mask for marking terms as extended or not
// Z, 0, 1, X, Y, -, +
const int REV_EXT_MASK[7] = {1, 0, 0, 1, 1, 0, 0};

/**
 * In-place marks a term as extended or not
 *
 * @param term Hamiltonian term
 *
 */
inline void set_extended_flag(OperatorTerm_t& term)
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
inline void set_offdiag_weight_and_phase(OperatorTerm_t& term)
{
    if(!term.values.size())
    {
        return;
    }
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

/**
 * Comparator for term grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int offdiag_group_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
    return term1.group < term2.group;
}

/**
 * Sort terms with same off-diagonal weight into groups that share the
 * same off-diagonal structure
 *
 */
inline void term_group_sort(std::vector<OperatorTerm_t>& terms,
                            std::size_t* __restrict weight_ptrs,
                            std::size_t len_ptrs,
                            unsigned int max_group_size)
{
    std::size_t ii;
#pragma omp parallel if(terms.size() > 1024)
    {
// Reset all groupings
#pragma omp for schedule(dynamic)
        for(ii = 0; ii < terms.size(); ii++)
        {
            terms[ii].group = 0; // diagonals are group 0 by convention
            if(terms[ii].offdiag_weight > 0)
            {
                terms[ii].group = -1;
            }
        } // end reset

#pragma omp for schedule(dynamic)
        for(ii = 0; ii < len_ptrs - 1; ii++)
        {
            std::size_t start = weight_ptrs[ii];
            std::size_t stop = weight_ptrs[ii + 1];
            int group_idx = ii * (max_group_size);

            if(terms[start].group == 0) // group is the diagonal group
            {
                continue;
            }

            std::map<std::vector<unsigned int>, int> pattern_to_group;

            for(std::size_t kk = start; kk < stop; kk++)
            {
                OperatorTerm_t* term = &terms[kk];

                // Build canonical key: sorted off-diagonal qubit indices
                std::vector<unsigned int> key;
                key.reserve(term->offdiag_weight);
                for(std::size_t idx = 0; idx < term->values.size(); idx++)
                    if(term->values[idx] > 2)
                        key.push_back(term->indices[idx]);
                std::sort(key.begin(), key.end());

                auto result = pattern_to_group.emplace(key, 0);
                if(result.second) // new pattern: allocate a new group
                {
                    group_idx += 1;
                    result.first->second = group_idx;
                }
                term->group = result.first->second;
            } // end kk loop

            // sort by group index within the start and stop indices
            std::sort(terms.begin() + start, terms.begin() + stop, offdiag_group_comp);
        } // end ii loop

    } // end omp parallel

    // relabel groups into continuous integers
    int current_group = 0;
    int current_idx = 0, next_idx = 1;
    for(ii = 0; ii < terms.size(); ii++)
    {
        if(terms[ii].group != current_group)
        {
            current_group = terms[ii].group;
            current_idx = next_idx;
            terms[ii].group = current_idx;
            next_idx += 1;
        }
        else
        {
            terms[ii].group = current_idx;
        }
    }
}

/**
 * Compute the ladder integer value for a given qubit term
 *
 */
inline unsigned int term_ladder_int(const OperatorTerm& term, unsigned int ladder_width)
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

/**
 * Sort terms within each group by their ladder integer values
 *
 */
inline void sort_groups_by_ladder_int(std::vector<OperatorTerm>& terms,
                                      const std::size_t* group_ptrs,
                                      unsigned int num_groups,
                                      unsigned int ladder_width)
{

    unsigned int kk;
#pragma omp parallel for if(num_groups > 128)
    for(kk = 0; kk < num_groups; kk++)
    {
        std::size_t start, stop;
        start = group_ptrs[kk];
        stop = group_ptrs[kk + 1];
        if(!terms[start].group) // This is true if the group=0 and thus are diagonal terms
        {
            continue;
        }
        std::sort(terms.begin() + start,
                  terms.begin() + stop,
                  [=](const OperatorTerm& a, const OperatorTerm& b) {
                      unsigned int res_a, res_b;
                      res_a = term_ladder_int(a, ladder_width);
                      res_b = term_ladder_int(b, ladder_width);
                      return res_a < res_b;
                  });
    }
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
inline void compute_term_offdiag_inds(const OperatorTerm_t& term, unsigned int* offdiag_inds)
{
    unsigned int kk;
    unsigned int counter = 0;
    for(kk = 0; kk < term.indices.size(); kk++)
    {
        if(term.values[kk] > 2)
        {
            offdiag_inds[counter] = term.indices[kk];
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
inline void set_group_offdiag_indices(const std::vector<OperatorTerm_t>& terms,
                                      std::vector<std::vector<unsigned int>>& group_indices,
                                      const std::size_t* group_ptrs,
                                      unsigned int num_groups)
{
    unsigned int kk;
    unsigned int inds_len;
    group_indices.resize(num_groups);
    for(kk = 0; kk < num_groups; kk++)
    {
        inds_len = terms[group_ptrs[kk]].offdiag_weight;
        group_indices[kk].resize(inds_len);
        compute_term_offdiag_inds(terms[group_ptrs[kk]], &(group_indices[kk])[0]);
    }
}

/**
 * Term indices corresponding to ladder integers for each group
 *
 */
inline void ladder_int_starts(const std::vector<OperatorTerm>& terms,
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

/** @struct QubitOperator
 * @brief Data structure for each a qubit operator, i.e. a collection of 'words'
 *
 * @var width is the number of qubits
 * @var terms is a vector of OperatorTerms that make up the operator
 * @var sorted is a flag that indicates the term is sorted (NOT USED AT PRESENT)
 */
typedef struct QubitOperator
{
    unsigned int width;
    std::vector<OperatorTerm_t> terms;
    int type{1};
    unsigned int ladder_width{DEFAULT_LADDER_WIDTH};
    int sorted{0};              // Are the operator terms group sorted
    int weight_sorted{0};       // Are the operator terms weight sorted 
    int off_weight_sorted{0};   // Are the operator terms off-diagonal weight sorted
    int ladder_sorted{0};       // Are the operator terms ladder int sorted within their groups?
    int structure_sorted{0};    // Are the operator terms sorted by (non-unique) off-diagonal structure?

    QubitOperator() {}
    /**
     * Constructor building an empty operator with a given width
     *
     * @param[in] width The width (number of qubits) of the operator
     */
    QubitOperator(unsigned int x)
    {
        width = x;
    }

    QubitOperator(unsigned int x, std::vector<TermData> data)
        : width(x)
    {
        unsigned int num_terms = data.size();
        std::size_t kk;
        TermData tdata;
        OperatorTerm term;
        std::complex<double> coeff = 1.0;
        for(kk = 0; kk < num_terms; kk++)
        {
            tdata = data[kk];
            _validate_indices(std::get<1>(tdata),
                              width); // validate that all indices are less than operator width
            // If there are no indices and the coeff==0 then the term should be an identity term with coeff=1
            if(std::get<1>(tdata).size() == 0 && std::get<2>(tdata) == std::complex<double>(0, 0))
            {
                coeff = 1.0;
            }
            else
            {
                coeff = std::get<2>(tdata);
            }
            term = OperatorTerm(std::get<0>(tdata), std::get<1>(tdata), coeff);
            term.set_proj_indices();
            set_offdiag_weight_and_phase(term);
            set_extended_flag(term);
            terms.push_back(term);
        }
    }
    // destructor
    ~QubitOperator()
    {
        std::vector<OperatorTerm_t>().swap(terms);
    }
    /**
     * QubitOperator from string label
     */
    static QubitOperator from_label(std::string label)
    {
        unsigned int width = label.size();
        unsigned char val;
        std::size_t counter = 0;
        QubitOperator out = QubitOperator(width);
        OperatorTerm term = OperatorTerm(1.0); // start with term set with coeff = 1.0
        for(auto it = label.rbegin(); it != label.rend(); it++)
        {
            if(*it != 73)
            {
                val = oper_map[*it];
                term.values.push_back(val);
                term.indices.push_back(counter);
            }
            counter += 1;
        }
        set_offdiag_weight_and_phase(term);
        term.set_proj_indices();
        set_extended_flag(term);
        out.terms.push_back(term);
        return out;
    }
    /**
     * Grab a single term by index
     * 
     * @param[in] Index of term to grab
     * 
     * @return OperatorTerm at the given index
     */
    OperatorTerm_t operator[](std::size_t index) const
    {
        if(index >= this->size())
        {
            throw std::runtime_error("Index is larger than operator size");
        }
        return terms[index];
    }
    /**
     * Inplace multiplication by a complex value
     */
    QubitOperator& operator*=(std::complex<double> c)
    {
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            terms[kk] *= c;
        }
        return *this;
    }
    /**
     * multiplication by a complex value (need one for mult on each side)
     */
    friend QubitOperator operator*(QubitOperator& op, std::complex<double> c)
    {
        QubitOperator out = op.copy();
        for(std::size_t kk = 0; kk < out.size(); kk++)
        {
            out.terms[kk] *= c;
        }
        return out;
    }
    friend QubitOperator operator*(std::complex<double> c, QubitOperator& op)
    {
        QubitOperator out = op.copy();
        for(std::size_t kk = 0; kk < out.size(); kk++)
        {
            out.terms[kk] *= c;
        }
        return out;
    }
    /**
     * Inplace addition by another QubitOperator
     * 
     * @param[in] other Operator to add to this one
     * @throw Error if operators do not share the same width
     */
    QubitOperator& operator+=(QubitOperator other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        for(std::size_t kk = 0; kk < other.size(); kk++)
        {
            this->terms.push_back(other.terms[kk]);
        }
        this->sorted = 0;
        return *this;
    }
    /**
     * Inplace subtraction by another QubitOperator
     * 
     * @param[in] other Operator to add to this one
     * @throw Error if operators do not share the same width
     */
    QubitOperator& operator-=(QubitOperator other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }

        OperatorTerm term;
        for(std::size_t kk = 0; kk < other.size(); kk++)
        {
            term = other.terms[kk];
            term.coeff *= -1;
            this->terms.push_back(term);
        }
        this->sorted = 0;
        return *this;
    }
    /**
     * Subtraction by another QubitOperator
     * 
     * @param[in] other Operator to subject to this one
     * @return New operator
     * @throw Error if operators do not share the same width
     */
    QubitOperator operator-(QubitOperator other)
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }

        OperatorTerm term;
        QubitOperator out = this->copy();
        for(std::size_t kk = 0; kk < other.size(); kk++)
        {
            term = other.terms[kk];
            term.coeff *= -1;
            out.terms.push_back(term);
        }
        return out;
    }
    /**
     * Addition by another QubitOperator
     * 
     * @param[in] other Operator to add to this one
     * @return The new operator
     * @throw Error if operators do not share the same width
     */
    QubitOperator operator+(QubitOperator other) const
    {
        if(other.width != this->width)
        {
            throw std::runtime_error("Operators must have the same width");
        }
        QubitOperator out = this->copy();
        for(std::size_t kk = 0; kk < other.size(); kk++)
        {
            out.terms.push_back(other.terms[kk]);
        }
        return out;
    }
    /**
     * Print object to standard output stream
     */
    friend auto operator<<(std::ostream& os, const QubitOperator& self) -> std::ostream&
    {
        std::size_t num_terms = self.size();
        std::size_t total_terms = num_terms;
        OperatorTerm_t term;
        int too_many_terms = 0;
        std::size_t kk, jj;

        // restrict to outputting at most 100 terms
        if(num_terms > 100)
        {
            too_many_terms = 1;
            num_terms = 100;
        }
        os << "<QubitOperator["; // start output here
        for(kk = 0; kk < num_terms; kk++)
        {
            term = self.terms[kk];
            os << "{";
            for(jj = 0; jj < term.indices.size(); jj++)
            {
                os << rev_oper_map[term.values[jj]] << ":" << term.indices[jj];
                if(jj != term.indices.size() - 1)
                {
                    os << " ";
                }
            }
            os << ", " << term.coeff;
            os << "}";
            if(kk != num_terms - 1)
            {
                os << ", ";
            }
        }
        if(too_many_terms)
        {
            os << " + " << (total_terms - 100) << "terms";
        }
        return os << ", width=" << self.width << "]>";
    }
    auto begin()
    {
        return terms.begin();
    }
    auto end()
    {
        return terms.end();
    }
    /**
     * The number of terms in the operator
     *
     * @return The number of terms in the operator
     */
    std::size_t size() const
    {
        return terms.size();
    }
    std::size_t num_terms() const
    {
        return terms.size();
    }
    /**
     * The number of terms in the operator
     *
     * @return A copy of the current operator
     */
    QubitOperator copy() const
    {
        QubitOperator out = QubitOperator(this->width);
        out.terms = this->terms;
        out.type = this->type;
        return out;
    }
    /**
     * Is the operator diagonal
     */
    bool is_diagonal() const
    {
        std::size_t kk;
        bool diag = 1;
        for(kk = 0; kk < terms.size(); kk++)
        {
            if(!terms[kk].is_diagonal())
            {
                diag = 0;
                break;
            }
        }
        return diag;
    }
    /**
     * Can operator be described via a symmetric matrix
     */
    bool is_real() const
    {
        bool out = true;
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            if(std::abs(terms[kk].coeff.imag()) > ATOL || !terms[kk].real_phase)
            {
                out = false;
                break;
            }
        }
        return out;
    }
    /**
     * Set operator type inplace
     * 
     * @param[in] x Integer type of operator
     * @throw Error if type is not 1 or 2
     */
    QubitOperator& set_type(int x)
    {
        if(x > 2 || x < 1)
        {
            throw std::runtime_error("Type must be 1 or 2");
        }
        this->type = x;
        return *this;
    }
    /**
    * Return vector of coefficients for each term
    * 
    * @return Vector of coefficients for terms
    */
    std::vector<std::complex<double>> coefficients() const
    {
        std::vector<std::complex<double>> out;
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            out.push_back(this->terms[kk].coeff);
        }
        return out;
    }
    /**
    * Return vector of weights for each term
    * 
    * @return Vector of weights for terms
    * 
    */
    std::vector<unsigned int> weights() const
    {
        std::vector<unsigned int> out;
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            out.push_back(this->terms[kk].weight());
        }
        return out;
    }
    /**
    * Return vector of real-phases for each term
    * 
    * @return Vector of real-phases for terms
    */
    std::vector<int> real_phases() const
    {
        std::vector<int> out;
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            out.push_back(this->terms[kk].real_phase);
        }
        return out;
    }
    /**
    * Return vector of showing which terms are extended alphabet
    * 
    * @return Vector of showing which terms are extended alphabet
    */
    std::vector<int> extended_terms() const
    {
        std::vector<int> out;
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            out.push_back(this->terms[kk].extended);
        }
        return out;
    }
    /**
     * Split operator into diagonal and off-diagonal components
     * 
     * @return Diagonal and off-diagonal operators
     */
    std::pair<QubitOperator, QubitOperator> split_diagonal() const
    {
        QubitOperator diag = QubitOperator(this->width);
        QubitOperator off = QubitOperator(this->width);
        for(auto term : this->terms)
        {
            if(term.is_diagonal())
            {
                diag.terms.push_back(term);
            }
            else
            {
                off.terms.push_back(term);
            }
        }
        off.type = this->type;
        diag.type = this->type;
        return {diag, off};
    }
    /**Constant energy of operator
    * 
    */
    double constant_energy() const
    {
        double out = 0;
        for(std::size_t kk = 0; kk < terms.size(); kk++)
        {
            if(!terms[kk].indices.size())
            {
                out += terms[kk].coeff.real();
            }
        }
        return out;
    }
    /**
    * Remove constant terms from operator
    * 
    */
    QubitOperator remove_constant_terms()
    {
        QubitOperator out = QubitOperator(this->width);
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            if(terms[kk].indices.size())
            {
                out.terms.push_back(terms[kk]);
            }
        }
        out.type = this->type;
        return out;
    }
    /**
    * In-place sorting of terms by weight
    * 
    */
    QubitOperator& weight_sort()
    {
        // sort by weight
        std::sort(terms.begin(), terms.end(), weight_comp);
        set_sorting_flags(*this, "weight");
        return *this;
    }
    /**
    * In-place sorting of terms by off-diagonal weight
    * 
    */
    QubitOperator& offdiag_weight_sort()
    {
        // sort by off-diagonal weight
        std::sort(terms.begin(), terms.end(), offweight_comp);
        set_sorting_flags(*this, "off_weight");
        return *this;
    }
    /**
    * Pointers to starting indices for off-diagonally sorted operator
    * 
    */
    std::vector<std::size_t> offdiag_weight_ptrs()
    {
        std::vector<std::size_t> ptrs;
        if(!this->off_weight_sorted)
        {
            this->offdiag_weight_sort();
        }
        set_offdiag_weight_ptrs(terms, ptrs);
        return ptrs;
    }
    /**
    * Pointers to starting indices for off-diagonally sorted operator
    * 
    */
    std::vector<std::size_t> offdiag_structure_ptrs()
    {
        std::vector<std::size_t> ptrs;
        if(!this->structure_sorted)
        {
            term_offdiag_sort(*this);
        }
        set_offdiag_structure_ptrs(terms, ptrs);
        return ptrs;
    }
    /**
    * In-place sorting of terms into groups (shared off-diagonal structure)
    * 
    */
    QubitOperator& group_sort()
    {
        if(this->size()) // do stuff only if there are terms in the operator
        {
            if(!this->structure_sorted)
            {
                term_offdiag_sort(*this);
            }
            std::vector<std::size_t> ptrs = this->offdiag_structure_ptrs();
            std::size_t max_group_size = max_offdiag_ptr_size(ptrs);
            term_group_sort(this->terms, &ptrs[0], ptrs.size(), max_group_size);
        }
        set_sorting_flags(*this, "group");
        return *this;
    }
    /**
    * Return a vector of all the term group labels
    * 
    */
    std::vector<int> groups() const
    {
        std::vector<int> out;
        out.resize(terms.size());
        for(std::size_t kk = 0; kk < terms.size(); kk++)
        {
            out[kk] = terms[kk].group;
        }
        return out;
    }
    /**
    * Return a vector of pointers to all the groups
    * 
    */
    std::vector<std::size_t> group_ptrs()
    {
        std::vector<std::size_t> out;
        if(!this->size()) // return empty vector if no terms
        {
            return out;
        }
        if(!this->sorted)
        {
            this->group_sort();
        }
        set_group_ptrs(terms, out);
        return out;
    }
    /**
    * Return a vector of pointers to all the groups
    * 
    */
    QubitOperator terms_by_group(int idx)
    {
        if(!this->sorted)
        {
            throw std::runtime_error("Operator must be group sorted first");
        }
        QubitOperator out = QubitOperator(this->width);
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            if(terms[kk].group == idx)
            {
                out.terms.push_back(terms[kk]);
            }
            else if(terms[kk].group > idx)
            {
                break;
            }
        }
        if(!out.size())
        {
            throw std::runtime_error("No terms with given group index found");
        }
        out.sorted = 1;
        out.type = this->type;
        return out;
    }
    /**
    * Off-diagonal indices for each group of terms
    */
    std::vector<std::vector<unsigned int>> group_offdiag_indices()
    {
        if(!this->sorted)
        {
            throw std::runtime_error("Operator must be group sorted first");
        }

        std::vector<std::vector<unsigned int>> out;
        std::vector<std::size_t> ptrs = this->group_ptrs();
        set_group_offdiag_indices(this->terms, out, &ptrs[0], ptrs.size() - 1);
        return out;
    }
    /**
    * Combine repeated terms in operator
    * 
    * @param[in] atol Tolerance for determining if a combined coefficient is zero
    * 
    * @return Output QubitOperator with terms combined
    * 
    */
    QubitOperator combine_repeated_terms(double atol = 1e-12)
    {
        QubitOperator out = QubitOperator(this->width);
        if(!this->size())
        {
            return out;
        }
        if(!this->weight_sorted)
        {
            this->weight_sort();
        }
        std::vector<unsigned int> touched;
        touched.resize(this->size());
        combine_qubit_terms(this->terms, out.terms, &touched[0], atol);
        out.type = this->type;
        return out;
    }
    /**
    * Return vector of off-diagonal weights for each term
    * 
    * @return Vector of off-diagonal weights for terms
    * 
    */
    std::vector<unsigned int> offdiag_weights() const
    {
        std::vector<unsigned int> out;
        out.resize(terms.size());
        std::size_t kk;
        for(kk = 0; kk < terms.size(); kk++)
        {
            out[kk] = terms[kk].offdiag_weight;
        }
        return out;
    }
    /**
    * In-place sort terms in groups by their ladder integer values
    * 
    */
    QubitOperator& group_term_sort_by_ladder_int(unsigned int ladder_width = 4)
    {
        if(!(this->type == 2))
        {
            throw std::runtime_error("Operator must be type=2");
        }
        if(!this->sorted)
        {
            this->group_sort();
        }
        std::vector<std::size_t> ptrs = this->group_ptrs();
        sort_groups_by_ladder_int(this->terms, &ptrs[0], ptrs.size() - 1, ladder_width);
        this->ladder_width = ladder_width;
        this->ladder_sorted = 1;
        return *this;
    }
    /**
    * Vector of ladder integer values for terms in operators
    * 
    * If no ladder ops present then default int is max(uint32)
    * 
    */
    std::vector<unsigned int> ladder_integers()
    {
        std::vector<unsigned int> out;
        if(!this->ladder_sorted)
        {
            this->group_term_sort_by_ladder_int();
        }
        for(std::size_t kk = 0; kk < this->size(); kk++)
        {
            out.push_back(term_ladder_int(terms[kk], this->ladder_width));
        }
        return out;
    }
    /**
    * Vector of group ladder integer bit lengths
    * 
    */
    std::vector<unsigned int> group_ladder_int_bit_lengths()
    {
        std::vector<unsigned int> out;
        if(!this->ladder_sorted)
        {
            this->group_term_sort_by_ladder_int();
        }
        std::vector<std::size_t> ptrs = this->group_ptrs();
        unsigned int num_groups = ptrs.size() - 1;
        out.resize(num_groups);
        for(std::size_t kk = 0; kk < num_groups; kk++)
        {
            out[kk] = std::min(this->ladder_width, terms[ptrs[kk]].offdiag_weight);
        }
        return out;
    }
    /**
    * Flat vector containing pointers to terms within a group with given 
    * ladder integer value
    * 
    */
    std::vector<std::size_t> group_ladder_int_ptrs()
    {
        if(!this->ladder_sorted)
        {
            this->group_term_sort_by_ladder_int();
        }
        std::vector<std::size_t> ptrs = this->group_ptrs();
        unsigned int num_ladder_ints = std::pow(2, this->ladder_width);
        unsigned int num_groups = ptrs.size() - 1;
        std::vector<unsigned int> group_counts(num_ladder_ints * num_groups);
        std::vector<std::size_t> group_ranges(num_ladder_ints * num_groups + 1);
        ladder_int_starts(terms,
                          &ptrs[0],
                          &group_counts[0],
                          &group_ranges[0],
                          num_groups,
                          num_ladder_ints,
                          ladder_width);
        return group_ranges;
    }

} QubitOperator_t;

/**
 * Set the QubitOperator flags when performing sorting of various kinds
 * 
 * @param[in, out] oper The operator whose flags to set
 * @param[in] kind Sting indicating the type of sorting that was performed
 * 
 * @throws Error if sorting type is not a valid kind
 */
inline void set_sorting_flags(QubitOperator& oper, std::string kind)
{
    if(kind == "group")
    {
        oper.sorted = 1;
        oper.weight_sorted = 0;
        oper.off_weight_sorted = 0;
        oper.ladder_sorted = 0; // since group sorting could modify the in-group ordering
        oper.structure_sorted = 0;
    }
    else if(kind == "weight")
    {
        oper.sorted = 0;
        oper.weight_sorted = 1;
        oper.off_weight_sorted = 0;
        oper.ladder_sorted = 0;
        oper.structure_sorted = 0;
    }
    else if(kind == "off_weight")
    {
        oper.sorted = 0;
        oper.weight_sorted = 0;
        oper.off_weight_sorted = 1;
        oper.ladder_sorted = 0;
        oper.structure_sorted = 0;
    }
    else if(kind == "ladder")
    {
        oper.sorted = 1; // since ladder sorting requires group sorting
        oper.weight_sorted = 0;
        oper.off_weight_sorted = 0;
        oper.ladder_sorted = 1;
        oper.structure_sorted = 0;
    }
    else if(kind == "structure")
    {
        oper.sorted = 0;
        oper.weight_sorted = 0;
        oper.off_weight_sorted = 0;
        oper.ladder_sorted = 0;
        oper.structure_sorted = 1;
    }
    else
    {
        throw std::runtime_error("Invalid sorting type.");
    }
}

/**
 * Sort terms in operator by their off-diagonal structure value
 *
 * @param terms Vector of operator terms
 *
 */
inline void term_offdiag_sort(QubitOperator& oper)
{
    std::vector<OperatorTerm_t>& terms = oper.terms;
    const std::size_t n = terms.size();

    // Precompute structure key for each term once instead of recomputing
    // it on every comparison during sort.
    std::vector<std::size_t> keys(n);
    for(std::size_t ii = 0; ii < n; ++ii)
        keys[ii] = term_offdiag_structure(terms[ii]);

    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&keys](std::size_t a, std::size_t b) {
        return keys[a] < keys[b];
    });

    std::vector<OperatorTerm_t> tmp(n);
    for(std::size_t ii = 0; ii < n; ++ii)
        tmp[ii] = std::move(terms[order[ii]]);
    terms = std::move(tmp);
    set_sorting_flags(oper, "structure");
}
