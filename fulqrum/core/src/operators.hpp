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
#include <complex>
#include <cstdlib>
#include <vector>

// Reverse mask for marking terms as extended or not
// Z, 0, 1, X, Y, -, +
const int REV_EXT_MASK[7] = {1, 0, 0, 1, 1, 0, 0};

/**
 * Sorting of indices and values for Operator term data
 *
 * @param inds The term indices (qubits) array
 * @param vals The term values (operators) array
 */
void sort_term_data(std::vector<unsigned int>& inds, std::vector<unsigned char>& vals)
{
	std::size_t n = inds.size();
	for(std::size_t i = 1; i < n; i++)
	{
		unsigned int key = inds[i];
		char val = vals[i];
		std::size_t j = std::lower_bound(inds.begin(), inds.begin() + i, key) - inds.begin();

		for(std::size_t k = i; k > j; k--)
		{
			inds[k] = inds[k - 1];
			vals[k] = vals[k - 1];
		}
		inds[j] = key;
		vals[j] = val;
	}
}

/**
 * Set the projector indices and bits for each term
 *
 * @param term OperatorTerm to add projector information to
 */
void set_proj_indices(OperatorTerm_t& term)
{
	std::size_t kk;
	unsigned int val;
	term.proj_indices.resize(0);
	term.proj_bits.resize(0);
	for(kk = 0; kk < term.values.size(); kk++)
	{
		val = term.values[kk];
		if(val == 1 || val == 2)
		{
			term.proj_indices.push_back(term.indices[kk]);
			term.proj_bits.push_back(val - 1);
		}
	}
}

/**
 * Comparitor for off-diagonal weight grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparitor value
 */
int offweight_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
	return term1.offdiag_weight < term2.offdiag_weight;
}

/**
 * Comparitor for weight grouping
 *
 * @param term1 The first term
 * @param term2 The second term
 *
 * @return comparitor value
 */
int weight_comp(OperatorTerm_t& term1, OperatorTerm_t& term2)
{
	return term1.indices.size() < term2.indices.size();
}

/**
 * Set the pointers for the off-diagonal weights
 *
 * @param terms Operator terms
 * @param vec Vector to add pointers to
 *
 */
void set_offdiag_weight_ptrs(std::vector<OperatorTerm_t>& __restrict terms,
							 std::vector<std::size_t>& vec)
{
	vec.resize(0);
	std::size_t kk;
	unsigned int val = terms[0].offdiag_weight;
	if(val > 0) // Only start pointers where non-diagonal terms start
	{
		vec.push_back(0);
	}
	for(kk = 1; kk < terms.size(); kk++)
	{
		if(terms[kk].offdiag_weight > val)
		{
			vec.push_back(kk);
			val = terms[kk].offdiag_weight;
		}
	}
	if(vec.size() != 0)
	{
		vec.push_back(terms.size());
	}
}

/**
 * Find max. number of elements with same off-diag weight
 *
 * Used for offseting the group counter for parallel execution
 *
 * @param vec Vector of off-diagonal pointers
 *
 * @returns Unsigned int for max. number of terms
 *
 */
unsigned int max_offdiag_ptr_size(std::size_t* vec, std::size_t size)
{
	std::size_t kk;
	unsigned int temp, max = 0;
	for(kk = 0; kk < size - 1; kk++)
	{
		temp = vec[kk + 1] - vec[kk];
		if(temp > max)
		{
			max = temp;
		}
	}
	return max;
}

/**
 * In-place marks a term as extended or not
 *
 * @param term Hamiltonian term
 *
 */
void set_extended_flag(OperatorTerm_t& term)
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
void set_offdiag_weight(OperatorTerm_t& term)
{
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

void set_weight_ptrs(std::vector<OperatorTerm_t>& __restrict terms, std::vector<std::size_t>& vec)
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

/**
 * Combine repeated terms that represent same
 * operators, dropping terms smaller than requested tolerance.
 *
 * Input terms must be sorted by weight before calling this routine
 *
 * @param terms Terms for input operator
 * @param out_terms Terms for ouput operator (to push_back to)
 * @param touched pointer array indicating if term has been touched
 * @param num_terms Number of terms in input operator
 * @param atol Absolute tolerance for term truncation
 *
 */
void combine_qubit_terms(std::vector<OperatorTerm_t>& __restrict terms,
						 std::vector<OperatorTerm_t>& __restrict out_terms,
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

inline unsigned int term_ladder_int(const OperatorTerm_t& term, unsigned int ladder_width)
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

void offdiag_weight_sort(QubitOperator_t& oper)
{
	// sort by group index
	std::sort(oper.terms.begin(), oper.terms.end(), offweight_comp);
	oper.off_weight_sorted = 1;
	oper.weight_sorted = 0;
}

void weight_sort(QubitOperator_t& oper)
{
	// sort by group index
	std::sort(oper.terms.begin(), oper.terms.end(), weight_comp);
	oper.off_weight_sorted = 0;
	oper.weight_sorted = 1;
}
