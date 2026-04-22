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
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "elements.hpp"
#include <boost/dynamic_bitset.hpp>

/**
 * Populate the diagonal vector for a given diagonal operator
 *
 *
 * @param data The subspace data
 * @param diag_vec The diagonal vector to store information to
 * @param val Variable storing the element value
 * @param diag_oper The diagonal operator
 * @param width The width of the operator
 * @param subspace_dim The dimension of the subspace
 */
template <typename T>
void compute_diag_vector(const bitset_map_namespace::BitsetHashMapWrapper& data,
                         T* __restrict diag_vec,
                         const QubitOperator_t& diag_oper,
                         const std::size_t subspace_dim)
{
    std::size_t kk;
    const auto* bitsets = data.get_bitsets();

#pragma omp parallel for if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    {
        T val = 0;
        const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;
        single_bitstring_diagonal(row, diag_oper.terms, val);
        diag_vec[kk] = val;
    }
}

/**
 * Compute the diagonal matrix-element for a single bit-string
 *
 *
 * @param row The row bit-string
 * @param diag_terms The diagonal operator
 * @param val Variable storing the element value
 */
template <typename T>
inline void single_bitstring_diagonal(const boost::dynamic_bitset<size_t>& row,
                                      const std::vector<OperatorTerm_t>& diag_terms,
                                      T& val)
{
    val = 0;
    const std::size_t num_terms = diag_terms.size();
    const OperatorTerm_t* term;
    width_t weight;
    std::size_t ll;
    for(ll = 0; ll < num_terms; ll++)
    {
        term = &diag_terms[ll];
        weight = term->indices.size();
        if(passes_proj_validation(term, row))
        {
            accum_element(
                row, row, term->indices, term->values, term->coeff, term->real_phase, weight, val);
        }
    }
}

/**
* Is diagonal fast_proj compatible 
*/
bool fast_diag_compatible(const QubitOperator& oper)
{
    bool out = true;
    if(oper.type != 2)
    {
        out = false;
    }
    else
    {
        std::size_t kk;
        for(auto term : oper.terms)
        {
            for(kk = 0; kk < term.proj_indices.size(); kk++)
            {
                if(term.proj_bits[kk] == 0)
                {
                    out = false;
                    break;
                }
            }
        }
    }
    return out;
}

/**
 * Compare terms based on the index of their 1st projector index, if any
 * 
 * This is used for sorting terms in a diagonal Hamiltonian so that computing
 * the energy along the diagonal is more efficient for type=2 Hamiltonians
 * 
 * If there are no projectors then the index value = -1, i.e. no projector terms
 * come first
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
inline int proj_index_term_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
    int term1_index = -1;
    int term2_index = -1;
    if(term1.proj_indices.size())
    {
        term1_index = term1.proj_indices[0];
    }
    if(term2.proj_indices.size())
    {
        term2_index = term2.proj_indices[0];
    }
    return term1_index < term2_index;
}

/**
* In-place sorting of terms by projector index
*/
QubitOperator& diag_proj_index_sort(QubitOperator& oper)
{
    if(oper.type != 2)
    {
        throw std::runtime_error("Operator must be type=2 for this to make sense");
    }
    if(!oper.is_diagonal())
    {
        throw std::runtime_error("Operator must be diagonal");
    }
    std::sort(oper.terms.begin(), oper.terms.end(), proj_index_term_comp);
    oper.off_weight_sorted = 0;
    oper.weight_sorted = 0;
    oper.sorted = 0;
    return oper;
}

std::pair<std::vector<std::pair<std::size_t, std::size_t>>, std::size_t>
projector_ptrs_and_offset(const QubitOperator& oper)
{
    std::pair<std::vector<std::pair<std::size_t, std::size_t>>, std::size_t> out;
    std::vector<std::pair<std::size_t, std::size_t>> ptrs(oper.width);
    std::size_t offset = 0;
    std::size_t kk;
    width_t current_ind;

    // compute the index (offset) at which terms begin to have projection operators
    for(kk = 0; kk < oper.size(); kk++)
    {
        if(oper.terms[kk].proj_indices.size())
        {
            break;
        }
        offset += 1;
    }
    out.second = offset; // set output second
    if(offset != oper.size()) // if there are terms with projectors do stuff
    {
        // set the current index to be the first projector index at terms[offset]
        current_ind = oper.terms[offset].proj_indices[0];
        // set the start pointer for this current index to be offset
        ptrs[current_ind].first = offset;
        for(kk = offset; kk < oper.size(); kk++)
        {
            if(oper.terms[kk].proj_indices[0] > current_ind)
            {
                ptrs[current_ind].second = kk;
                current_ind = oper.terms[kk].proj_indices[0];
                ptrs[current_ind].first = kk;
            }
        }
        ptrs[current_ind].second = oper.size();
    }
    out.first = ptrs; // set output first
    return out;
}

/**
 * Compute the diagonal matrix-element for a single bit-string
 *
 *
 * @param row The row bit-string
 * @param diag_terms The diagonal operator
 * @param val Variable storing the element value
 */
template <typename T>
inline void
single_bitstring_diagonal_fast(const boost::dynamic_bitset<size_t>& row,
                               const std::vector<OperatorTerm_t>& diag_terms,
                               const std::vector<std::pair<std::size_t, std::size_t>>& ptrs,
                               const std::size_t offset,
                               T& val)
{
    val = 0;
    //const std::size_t num_terms = diag_terms.size();
    const OperatorTerm_t* term;
    width_t weight;
    std::size_t kk;
    std::size_t ll;
    std::size_t start, stop;

    std::vector<width_t> set_bits = set_bit_indices(row);

    // take care of all terms with no projectors
    for(kk = 0; kk < offset; kk++)
    {
        term = &diag_terms[kk];
        weight = term->indices.size();
        accum_element(
            row, row, term->indices, term->values, term->coeff, term->real_phase, weight, val);
    }

    for(auto bit : set_bits)
    {
        start = ptrs[bit].first;
        stop = ptrs[bit].second;
        for(ll = start; ll < stop; ll++)
        {
            term = &diag_terms[ll];
            weight = term->indices.size();

            accum_element(
                row, row, term->indices, term->values, term->coeff, term->real_phase, weight, val);
        }
    }
}

/**
 * Compute the diagonal matrix-element for a single bit-string using fast projector mode
 *
 */
template <typename T>
void compute_diag_vector_fast(const bitset_map_namespace::BitsetHashMapWrapper& data,
                              T* __restrict diag_vec,
                              const QubitOperator_t& diag_oper,
                              const std::vector<std::pair<std::size_t, std::size_t>>& ptrs,
                              const std::size_t offset,
                              const std::size_t subspace_dim)
{
    std::size_t kk;
    const auto* bitsets = data.get_bitsets();

#pragma omp parallel for if(subspace_dim > 4096) schedule(dynamic)
    for(kk = 0; kk < subspace_dim; kk++)
    {
        T val = 0;
        const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;
        single_bitstring_diagonal_fast(row, diag_oper.terms, ptrs, offset, val);
        diag_vec[kk] = val;
    }
}
