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
#include <cassert>
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
inline void compute_diag_vector(const bitset_map_namespace::BitsetHashMapWrapper& data,
                                T* __restrict diag_vec,
                                const QubitOperator& diag_oper,
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
        diag_vec[kk] += val; // += here since const_energy (if any) is already included in diag
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
                                      const std::vector<OperatorTerm>& diag_terms,
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
* Here we assume any constant offset terms are removed before this
* function is called
*
*/
inline bool fast_diag_compatible(const QubitOperator& oper)
{
    bool out = true;
    if(oper.type != 2)
    {
        out = false;
    }
    else if(!oper.is_diagonal())
    {
        out = false;
    }
    // Number of non-constant terms in diag must be W * (W + 1) / 2
    else if(oper.size() != static_cast<std::size_t>(oper.width * (oper.width + 1) / 2))
    {
        out = false;
    }
    else
    {
        std::size_t kk;
        for(auto term : oper.terms)
        {
            // if condition is already false, or current term has no projectors, e.g. constant term,
            // 'Z' ops only
            if(!out)
            {
                break;
            }
            else if(term.proj_indices.size() == 0)
            {
                out = false;
                break;
            }
            // All projectors must be '1' for this to work so break if '0' is found
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
inline int proj_index_term_comp(OperatorTerm& term1, OperatorTerm& term2)
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

inline int proj_second_index_term_comp(OperatorTerm& term1, OperatorTerm& term2)
{
    int term1_index = -1;
    int term2_index = -1;
    if(term1.proj_indices.size() > 1)
    {
        term1_index = term1.proj_indices[1];
    }
    if(term2.proj_indices.size() > 1)
    {
        term2_index = term2.proj_indices[1];
    }
    return term1_index < term2_index;
}

/**
* In-place sorting of terms by projector index
*/
inline QubitOperator& diag_proj_index_sort(QubitOperator& oper)
{
    if(oper.type != 2)
    {
        throw std::runtime_error("Operator must be type=2 for this to make sense");
    }
    std::sort(oper.terms.begin(), oper.terms.end(), proj_index_term_comp);
    oper.off_weight_sorted = 0;
    oper.weight_sorted = 0;
    oper.sorted = 0;
    return oper;
}

inline void fast_diag_term_sort(QubitOperator& oper)
{
    if(!oper.terms.size())
    {
        return; // return if there are no terms
    }
    if(!oper.is_diagonal())
    {
        throw std::runtime_error("Operator must be diagonal");
    }
    width_t width = oper.width;
    diag_proj_index_sort(oper);
    std::vector<std::size_t> ptrs = {0};
    std::size_t current = 0;
    for(width_t kk = 0; kk < width; kk++)
    {
        current += (width - kk);
        ptrs.push_back(current);
    }
#pragma omp parallel for if(width > 31) schedule(dynamic)
    for(std::size_t ll = 0; ll < (ptrs.size() - 1); ll++)
    {
        std::size_t start = ptrs[ll];
        std::size_t stop = ptrs[ll + 1];
        // sort by group index within the start and stop indices
        std::sort(
            oper.terms.begin() + start, oper.terms.begin() + stop, proj_second_index_term_comp);
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
inline void single_bitstring_diagonal_fast(const boost::dynamic_bitset<size_t>& row,
                                           const std::vector<OperatorTerm>& diag_terms,
                                           const std::vector<std::size_t>& row_ptrs,
                                           T& val)
{
    val = 0;
    //const std::size_t num_terms = diag_terms.size();
    const OperatorTerm* term;
    width_t weight;
    std::size_t start, offset;

    std::vector<width_t> set_bits = set_bit_indices(row);

    for(std::size_t ll = 0; ll < set_bits.size(); ll++)
    {
        width_t current_bit = set_bits[ll];
        start = row_ptrs[current_bit];
        for(std::size_t mm = ll; mm < set_bits.size(); mm++)
        {
            offset = set_bits[mm] - current_bit;
            term = &diag_terms[start + offset];
            weight = term->indices.size();
            accum_element(
                row, row, term->indices, term->values, term->coeff, term->real_phase, weight, val);
        }
    }
}

/**
 * Incremental fast-diagonal energy col bitset. E(row) must be known.
 *
 * row and col bitsets only differs on offdiag_grp_inds positions (flip_inds here).
 * When we know the diagonal energy for a row bitset, E(row), we can incrementally
 * correct the E(row) to compute the E(col).
 * 
 * single_bitstring_diagonal_fast uses two nested loops over set bits indices (= num
 * of electrons) of a bitset, which takes nelecs * (nelecs + 1) / 2 iterations. 
 * Consider a row bitset with set bits at [0, 2, 3, 5] (nelecs = 4). E(row) iterates
 * over or has contribution from following 10 (set_bits[ll], set_bits[mm]) pairs:
 * (0, 0), (0, 2), (0, 3), (0, 5), (2, 2), (2, 3), (2, 5), (3, 3), (3, 5), and (5, 5).
 * 
 * Now, consider a group with flip_inds [2, 4]. The corresponding col for this group
 * and row will have set bits at [0, 3, 4, 5] ("1" bit at row pos 2 flips to "0", and
 * "0" bit at row pos 4 flips to "1"). E(col) needs contribution from following 10
 * pairs:
 * (0, 0), (0, 3), (0, 4), (0, 5), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5), and (5, 5).
 * 
 * If we inspect pairs from E(row) and E(col), we can find three buckets of pairs:
 * A. Common in both row and col: (0, 0), (0, 3), (0, 5), (3, 3), (3, 5), and (5, 5).
 * B. In col, but not in row: (0, 4), (3, 4), (4, 4), (4, 5)
 * C. Not in col, but in row: (0, 2), (2, 2), (2, 3), (2, 5)
 * 
 * To compute E(col) by editing E(row), we need to subtract contributions from pairs
 * that are not in col but in row (Bucket # C), and add from pairs that are in col but not in row (Bucket # B).
 * 
 * E(col) = E(row) + delta
 *        = E(row) + Bucket B - Bucket C
 * 
 * When we already know E(row), we just need to compute delta to find E(col).
 * Incremental delta needs 4 pairs in Bucket B + 4 pairs in Bucket C = 8 pairs
 * instead of full 10 pairs from scratch to get E(col).
 * 
 * In general, incremental delta needs nflips * nelecs steps instead of
 * (nelecs * (nelecs + 1)) / 2 steps, which can pay more dividends when
 * nelecs is large.
 * For example, Fe4S4 has 54 electrons requiring (54 * 55) / 2 = 1485 steps per det.
 * With incremental logic that drops to 2 * 54 = 108 for single excitation groups
 * and 4 * 54 + 2 = 218 for double excitation groups.
 *
 * `row_occ` must be set_bit_indices(row) (passed in so it can be reused across all
 * connected cols of a row).`flip_inds` need not be
 * sorted; nflip must be <=4 (2 or 4).
 */
template <typename T>
inline T single_bitstring_diagonal_delta_fast(const boost::dynamic_bitset<size_t>& row,
                                              const boost::dynamic_bitset<size_t>& col,
                                              const std::vector<width_t>& row_occ,
                                              const std::vector<OperatorTerm>& diag_terms,
                                              const std::vector<std::size_t>& row_ptrs,
                                              const width_t* flip_inds,
                                              const unsigned int nflip,
                                              const T e_row)
{
    auto g = [&](width_t a, width_t b, const boost::dynamic_bitset<size_t>& det) -> T {
        const width_t lo = a < b ? a : b;
        const width_t hi = a < b ? b : a;
        const OperatorTerm* term = &diag_terms[row_ptrs[lo] + (hi - lo)];
        T v = 0;
        accum_element(det,
                      det,
                      term->indices,
                      term->values,
                      term->coeff,
                      term->real_phase,
                      term->indices.size(),
                      v);
        return v;
    };

    // Split flips into positions added (0->1) and removed (1->0) going row -> col.
    // Type-2 off-diagonal groups are single (nflip==2) or double (nflip==4)
    // excitations, so na + nr == nflip <= 4 and each buffer holds at most 4.
    assert(nflip <= 4);
    width_t added[4];
    width_t removed[4];
    unsigned int na = 0, nr = 0;
    for(unsigned int f = 0; f < nflip; f++)
    {
        const width_t p = flip_inds[f];
        if(row[p])
        {
            removed[nr++] = p;
        }
        else
        {
            added[na++] = p;
        }
    }

    auto is_removed = [&](width_t y) {
        for(unsigned int i = 0; i < nr; i++)
        {
            if(removed[i] == y)
            {
                return true;
            }
        }
        return false;
    };

    T delta = 0;
    for(unsigned int i = 0; i < na; i++)
    {
        const width_t a = added[i];
        for(std::size_t t = 0; t < row_occ.size(); t++)
        {
            const width_t y = row_occ[t];
            if(!is_removed(y))
            {
                delta += g(a, y, col);
            }
        }
        for(unsigned int j = 0; j < na; j++) // added-added (incl self g(a,a))
        {
            delta += g(a, added[j], col);
        }
    }
    for(unsigned int i = 0; i < na; i++) // undo double-counted added-added pairs
    {
        for(unsigned int j = i + 1; j < na; j++)
        {
            delta -= g(added[i], added[j], col);
        }
    }

    for(unsigned int i = 0; i < nr; i++)
    {
        const width_t r = removed[i];
        for(std::size_t t = 0; t < row_occ.size(); t++)
        {
            delta -= g(r, row_occ[t], row);
        }
    }
    for(unsigned int i = 0; i < nr; i++) // undo double-counted removed-removed pairs
    {
        for(unsigned int j = i + 1; j < nr; j++)
        {
            delta += g(removed[i], removed[j], row);
        }
    }

    return e_row + delta;
}

/**
 * Compute the diagonal matrix-element for a single bit-string using fast projector mode
 *
 */
template <typename T>
inline void compute_diag_vector_fast(const bitset_map_namespace::BitsetHashMapWrapper& data,
                                     T* __restrict diag_vec,
                                     const QubitOperator& diag_oper,
                                     const std::size_t subspace_dim)
{
    std::size_t kk;
    width_t width = diag_oper.width;
    const auto* bitsets = data.get_bitsets();

    // set row_pointers
    std::vector<std::size_t> row_ptrs;
    row_ptrs.reserve(width + 1);
    row_ptrs.push_back(0);
    std::size_t current = 0;
    for(width_t kk = 0; kk < width; kk++)
    {
        current += (width - kk);
        row_ptrs.push_back(current);
    }

#pragma omp parallel for if(subspace_dim > 4096)
    for(kk = 0; kk < subspace_dim; kk++)
    {
        T val = 0;
        const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;
        single_bitstring_diagonal_fast(row, diag_oper.terms, row_ptrs, val);
        diag_vec[kk] += val; // += here since const_energy (if any) is already included in diag
    }
}