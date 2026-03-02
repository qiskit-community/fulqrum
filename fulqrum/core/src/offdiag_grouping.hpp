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
#include <cstdlib>
#include <vector>

/**
 * Compute an integer value from the off-diagonal structure of a term
 *
 * @param term The term
 *
 * @return Structure value
 */
std::size_t term_offdiag_structure(const OperatorTerm_t& term)
{
    std::size_t kk;
    std::size_t out = 0;
    for(kk = 0; kk < term.values.size(); ++kk)
    {
        out +=
            (term.indices[kk] + 1) *
            (term.values[kk] >
             2); // need plus one here so that an offdiag on 0 index does not look like a diagonal term
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
int offdiag_comp(const OperatorTerm_t& term1, const OperatorTerm_t& term2)
{
    return term_offdiag_structure(term1) < term_offdiag_structure(term2);
}

/**
 * Sort terms in operator by their off-diagonal structure value
 *
 * @param terms Vector of operator terms
 *
 */
void term_offdiag_sort(std::vector<OperatorTerm_t>& terms)
{
    std::sort(terms.begin(), terms.end(), offdiag_comp);
}

unsigned int _max_offdiag_group_size(std::size_t* __restrict ptrs, std::size_t num_elems)
{
    std::size_t kk, max_size = 0;
    for(kk = 0; kk < num_elems - 1; kk++)
    {
        if((ptrs[kk + 1] - ptrs[kk]) > max_size)
        {
            max_size = (ptrs[kk + 1] - ptrs[kk]);
        }
    }
    return static_cast<unsigned int>(max_size);
}

/**
 * Comparator for term grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparator value
 */
int offdiag_group_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
    return term1.group < term2.group;
}

void term_group_sort(std::vector<OperatorTerm_t>& terms,
                     std::size_t* __restrict weight_ptrs,
                     std::size_t len_ptrs,
                     unsigned int max_group_size)
{
    std::size_t ii;
    // Reset all groupings
    for(ii = 0; ii < terms.size(); ii++)
    {
        terms[ii].group = 0; // diagonals are group 0 by convention
        if(terms[ii].offdiag_weight > 0)
        {
            terms[ii].group = -1;
        }
    } // end reset

    std::ptrdiff_t dist;
#pragma omp parallel for schedule(dynamic) if(terms.size() > 1024)
    for(ii = 0; ii < len_ptrs - 1; ii++)
    {
        std::size_t start = weight_ptrs[ii];
        std::size_t stop = weight_ptrs[ii + 1];
        int group_idx = ii * (max_group_size);
        std::size_t kk, ll, idx;
        OperatorTerm_t* term;
        OperatorTerm_t* term2;
        std::vector<unsigned int>::iterator inds_it;
        int match;
        std::size_t ind_size;

        if(terms[start].group == 0) // group is the diagonal group
        {
            continue;
        }

        for(kk = start; kk < stop; kk++)
        {
            term = &terms[kk];
            ind_size = term->indices.size();
            if(term->group < 0) // term is not touched yet
            {
                group_idx += 1; // diags are group zero, so go to 1 first
                term->group = group_idx;
            }
            // Loop over all terms from kk+1 on up t ostop
            for(ll = kk + 1; ll < stop; ll++)
            {
                term2 = &terms[ll];
                // term2 is not matched and number of off-diag ops is equal
                if((term2->group < 0) && (term2->offdiag_weight == term->offdiag_weight))
                {
                    match = 1;
                    for(idx = 0; idx < ind_size; idx++)
                    {
                        // found off-diag term at idx
                        if(term->values[idx] > 2)
                        {
                            // Tell me if the index is also found in term2
                            inds_it = std::find(
                                term2->indices.begin(), term2->indices.end(), term->indices[idx]);
                            if(inds_it == term2->indices.end())
                            {
                                match = 0;
                                break;
                            }
                            // if the index is in term2, find out its location and check for off-diag there
                            else
                            {
                                dist = std::distance(term2->indices.begin(), inds_it);
                                if(!(term2->values[dist] > 2))
                                {
                                    match = 0;
                                    break;
                                }
                            }
                        } // end found off-diag term
                    } // end idx for-loop

                    if(match)
                    { // If match
                        term2->group = group_idx;
                    }
                } // end non-id match
            } // end ll for-loop
        } // end kk for-loop
        // sort by group index within the start and stop indices
        std::sort(&terms[start], &terms[stop], offdiag_group_comp);
    } // end ii loop

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
 * Constructs a vector of max offdiagonal group indices.
 *
 * The vector allows us to check only one bit position in the row bitset/bitvec to
 * determine whether the candidate column bitset will be smaller or greater than the
 * row bitset. In the process, it tells us if the matrix elements belong to lower (or
 * upper) triangle matrix. In Fulqrum, we only process lower triangle elements to
 * speed up the computation.
 *
 * DETAILS: LOWER TRIANGLE ELEMENTS DETECTION
 * Fulqrum deals with Hamiltonians with Hermitian matrices only where upper
 * triangle of a matrix is the complex conjugate of the lower triangle (or vice versa).
 * We can evaluate only lower half of a matrix and readily know the upper
 * half to construct the full matrix. This can cut down number of iterations or matrix
 * element evaluation operations by half and accelerate computations. Therefore, we need
 * a way to detect lower triangle elements to harness the speed up.
 *
 * For a matrix element to be in the lower triangle, ``col_idx < row_idx``.
 * When bitsets that spans rows and columns of the matrix are sorted in the ascending
 * order, the condition col_idx < row_idx translates into col_bitset < row_bitset.
 * In all of our CSR and MATVEC logics, we implicitly check for col_bitset < row_bitset
 * by testing only one bit position in the row_bitset.
 *
 * For a row_bitset, Fulqrum finds candidate col_bitsets with non-zero elements by
 * flipping certain bit positions in the row_bitset. The bit positions to be flipped
 * are stored in the 2D vector ``group_offdiag_inds``.
 *
 * Suppose, row_bitset = 110110 (little-endian bit ordering, i.e., b5b4b3b2b1b0)
 * and bit-flip positions for a group are [1, 3], i.e., max group inds or bit-flip
 * position is 3. The row bit at position 3 (b3) is 0. In the col_bitset, this
 * position will be flipped to 1. Now, a bitset 111xxx will always be greater than
 * 110xxx regardless of trailing bits. As col_bitset (111xxx) > row_bitset (110xxx),
 * this matrix element belongs to the upper triangle, and we skip evaluating it.
 *
 * Now, consider bit-flip positions for another group is [1 ,2], i.e., max index is 2.
 * The row bit at position 2 (b2) is 1, and it will be flipped to 0 in the col_bitset.
 * The col_bitset 1100xx will always be smaller than row_bitset 1101xx regardless of
 * trailing bits. Therefore, this group/matrix element corresponds to a lower triangle
 * element. We need to evaluate this element and populate corresponding upper triangle
 * entry.
 *
 * To summarize:
 * - Bitsets must be sorted in the ascening order.
 * - Row bit = 1 at max inds position -> lower triangle element. Do next steps.
 * - Row bit = 0 at max inds position -> upper triangle element. Skip.
 * - No need for explicit col bitset construction for lower half detection.
 *
 * @param grp_max_inds Vector to hold max index of each group. As the index
 * is represented by ``uint16_t``, we can solve max 2^16 = 65536 qubit problem
 * using Fulqrum.
 * @param group_offdiag_inds Offdiagonal indices of a group. This list determines
 * which bit positions in a row bitset will be flipped to construct a candidate
 * column bitset.
 * @param num_groups The number of groups.
 */
void get_group_max_inds(std::vector<uint16_t>& grp_max_inds,
                        const std::vector<std::vector<unsigned int>>& group_offdiag_inds,
                        const std::size_t& num_groups)
{
#pragma omp parallel for schedule(dynamic) if(num_groups > 4096)
    for(size_t group = 0; group < num_groups; group++)
    {
        auto group_inds = &group_offdiag_inds[group];

        // It is likely that the ``group_inds`` is already sorted, and
        // the last element is the maximum. However, as size of ``group_inds``
        // is moderate (2 or 4) and computing max element is cheap, we make use
        // of ``std::max_element()`` to be safe.
        auto max_element = std::max_element((*group_inds).begin(), (*group_inds).end());
        grp_max_inds[group] = *max_element;
    }
}
