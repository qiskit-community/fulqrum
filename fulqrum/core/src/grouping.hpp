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

/**
 * Compute the offdiag indices for the first term in a group and add it to the group
 * offdiag indices vector
 *
 * @param term Operator term
 * @param ladder_inds Pre-sized array (size=num_inds) to store indices in
 * @param num_inds Number of elements to consider for appending
 *
 */
inline void compute_term_offdiag_inds(const OperatorTerm_t& term,
									  unsigned int* offdiag_inds,
									  unsigned int num_inds)
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
void set_group_offdiag_indices(const std::vector<OperatorTerm_t>& terms,
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
		compute_term_offdiag_inds(terms[group_ptrs[kk]], &(group_indices[kk])[0], inds_len);
	}
}

inline void sort_groups_by_ladder_int(QubitOperator_t& oper,
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
		if(!oper.terms[start].group)
		{
			continue;
		}
		std::sort(&oper.terms[start],
				  &oper.terms[stop],
				  [=](const OperatorTerm_t& a, const OperatorTerm_t& b) {
					  unsigned int res_a, res_b;
					  res_a = term_ladder_int(a, ladder_width);
					  res_b = term_ladder_int(b, ladder_width);
					  return res_a < res_b;
				  });
	}
}

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
