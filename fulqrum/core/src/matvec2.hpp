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
#include <mutex>
#include <vector>

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include "operators.hpp"
#include <boost/dynamic_bitset.hpp>

template <typename T>
void omp_matvec2(const std::vector<OperatorTerm_t>& terms,
				 const bitset_map_namespace::BitsetHashMapWrapper& subspace,
				 const T* __restrict diag_vec,
				 const std::size_t width,
				 const std::size_t subspace_dim,
				 const int has_nonzero_diag,
				 const std::size_t* __restrict group_ptrs,
				 const std::size_t* __restrict group_ladder_ptrs,
				 const unsigned int* __restrict group_rowint_length,
				 const std::vector<std::vector<unsigned int>>& group_offdiag_inds,
				 const unsigned int num_groups,
				 const unsigned int ladder_offset,
				 const T* __restrict in_vec,
				 T* __restrict out_vec)
{
	std::size_t kk;
	const auto* bitsets = subspace.get_bitsets();

	std::vector<std::mutex> mutex1(subspace_dim);

	std::vector<uint16_t> grp_max_inds(num_groups, width);
	get_group_max_inds(grp_max_inds, group_offdiag_inds, num_groups);

	// Take care of diagonal term first, if any (usually there is)
	if(has_nonzero_diag)
	{
#pragma omp for
		for(kk = 0; kk < subspace_dim; kk++)
		{
			out_vec[kk] = diag_vec[kk] * in_vec[kk];
		}
	}

#pragma omp parallel if(subspace_dim > 4096)
	{
		std::size_t num_terms = terms.size();
		// Take care of off-diagonal terms
		if(num_terms)
		{
#pragma omp for schedule(dynamic)
			for(kk = 0; kk < subspace_dim; kk++)
			{
				const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;

				boost::dynamic_bitset<std::size_t> col_vec;
				T temp_val;
				const OperatorTerm_t* term;
				unsigned int group;
				std::size_t group_int_start, group_int_stop;
				std::size_t idx;
				std::size_t* col_ptr;
				std::size_t col_idx;
				unsigned int row_int;
				const std::vector<unsigned int>* group_inds;

				std::vector<uint8_t> row_set_bits(row.size(), 0);
				bitset_to_bitvec(row, row_set_bits);

				// Loop over all off-diagonal terms in operator
				for(group = 0; group < num_groups; group++)
				{
					// Detects a lower or an upper
					// triangle matrix element.
					// See details in ``get_group_max_inds()``
					// in fulqrum/core/src/offdiag_grouping.hpp
					if(!row_set_bits[grp_max_inds[group]])
					{
						continue;
					}

					group_inds = &group_offdiag_inds[group];
					row_int = bitset_ladder_int(
						row_set_bits.data(), group_inds->data(), group_rowint_length[group]);
					group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
					group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];
					temp_val = 0;
					if(group_int_start < group_int_stop)
					{
						col_vec = row;
						flip_bits(col_vec, group_inds->data(), group_inds->size());

						col_ptr = subspace.get_ptr(col_vec);
						if(col_ptr == nullptr)
						{
							continue;
						} // column is NOT in the subspace so break group
						col_idx = *col_ptr;
					}
					for(idx = group_int_start; idx < group_int_stop; idx++)
					{
						term = &terms[idx];
						if(passes_proj_validation(term, row))
						{
							accum_element(row,
										  col_vec,
										  &term->indices[0],
										  &term->values[0],
										  term->coeff,
										  term->real_phase,
										  term->indices.size(),
										  temp_val);
						}
					} // end loop over this group
					if(std::abs(temp_val) > ATOL) // if at least one element was found
					{
						{
							std::lock_guard<std::mutex> lock1(mutex1[kk]);
							out_vec[kk] += (temp_val * in_vec[col_idx]);
						}

						{
							std::lock_guard<std::mutex> lock2(mutex1[col_idx]);
							if constexpr(std::is_same_v<T, double>)
							{
								out_vec[col_idx] += (temp_val * in_vec[kk]);
							}
							else
							{
								out_vec[col_idx] += (std::conj(temp_val) * in_vec[kk]);
							}
						}
					}
				} // end loop over all groups
			} // end for-loop over rows
		} // end if num_terms
	} // end parallel region
} // end matvec
