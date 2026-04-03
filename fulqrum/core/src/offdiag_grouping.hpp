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
inline void get_group_max_inds(std::vector<uint16_t>& grp_max_inds,
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
