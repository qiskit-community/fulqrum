/**
 * This code is a part of Fulqrum.
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
#include <boost/dynamic_bitset.hpp>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../core/src/base.hpp"
#include "../../core/src/bitset_hashmap.hpp"
#include "../../core/src/bitset_utils.hpp"
#include "../../core/src/constants.hpp"
#include "../../core/src/diag.hpp"
#include "../../core/src/elements.hpp"

double simple_restricted(const QubitOperator& oper,
                         const bitset_map_namespace::BitsetHashMapWrapper& restricted_subspace,
                         bitset_map_namespace::BitsetHashMapWrapper& out_subspace,
                         QubitOperator& diag_oper,
                         const width_t width,
                         const std::size_t subspace_dim,
                         const int has_nonzero_diag,
                         const std::size_t* __restrict group_ptrs,
                         const std::size_t* __restrict group_ladder_ptrs,
                         const width_t* __restrict group_rowint_length,
                         const std::vector<std::vector<width_t>>& group_offdiag_inds,
                         const std::size_t num_groups,
                         const unsigned int ladder_offset,
                         const double target_energy,
                         const unsigned int max_recursion,
                         const double tol)
{

    // do stuff if the diagonal can be evaluated quickly
    bool do_fast_diag = fast_diag_compatible(oper);
    std::pair<std::vector<std::pair<std::size_t, std::size_t>>, std::size_t> ptrs_and_offset;
    if(do_fast_diag)
    {
        diag_proj_index_sort(diag_oper);
        ptrs_and_offset = projector_ptrs_and_offset(diag_oper);
    }

    std::size_t recur, kk;
    const auto* input_bitsets = restricted_subspace.get_bitsets();
    auto* output_bitsets = out_subspace.get_bitsets();

    std::vector<std::size_t> current_rows;
    std::vector<std::size_t> next_rows = {*restricted_subspace.get_ptr(output_bitsets[0].first)};

    std::vector<double> current_prefactors;
    std::vector<double> next_prefactors = {1.0 / target_energy};

    double est_energy = target_energy;
    double col_energy = 0;
    double energy_amp = 0;

    const OperatorTerm_t* term;
    boost::dynamic_bitset<std::size_t> col_vec;
    const std::vector<width_t>* group_inds;
    std::size_t* col_ptr;
    std::size_t* out_col_ptr;
    std::size_t idx;
    std::size_t group;
    std::size_t group_int_start, group_int_stop;
    std::size_t num_inserted_bitsets = 1;
    std::complex<double> val = 0;
    unsigned int row_int;
    int do_col_search;

    for(recur = 0; recur < max_recursion; recur++)
    {
        // Set current terms from previous recursion results
        current_rows = next_rows;
        current_prefactors = next_prefactors;
        next_rows.clear();
        next_prefactors.clear();
        // Loop over all rows in the current set
        for(kk = 0; kk < current_rows.size(); kk++)
        {
            const boost::dynamic_bitset<size_t>& row = input_bitsets[current_rows[kk]].first;
            std::vector<uint8_t> row_set_bits(row.size(), 0);
            bitset_to_bitvec(row, row_set_bits);

            for(group = 0; group < num_groups; group++)
            {
                group_inds = &group_offdiag_inds[group];
                row_int = bitset_ladder_int(
                    row_set_bits.data(), group_inds->data(), group_rowint_length[group]);
                group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
                group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];
                do_col_search = 1;
                val = 0;
                // begin loop over terms in this group
                for(idx = group_int_start; idx < group_int_stop; idx++)
                {
                    if(do_col_search)
                    {
                        col_vec = row;
                        flip_bits(col_vec, group_inds->data(), group_inds->size());
                        col_ptr = restricted_subspace.get_ptr(col_vec);
                        if(col_ptr == nullptr)
                        {
                            break; // column is NOT in the subspace so break group
                        }
                        do_col_search = 0;
                    }
                    term = &oper.terms[idx];
                    if(passes_proj_validation(term, row))
                    {
                        accum_element(row,
                                      col_vec,
                                      term->indices,
                                      term->values,
                                      term->coeff,
                                      term->real_phase,
                                      term->indices.size(),
                                      val);
                    }
                } // end loop over terms in this group

                if(!do_col_search)
                {
                    // If this column is in the subspace we need to compute the columns diagonal energy
                    if(do_fast_diag)
                    {
                        single_bitstring_diagonal_fast(input_bitsets[*col_ptr].first,
                                                       diag_oper.terms,
                                                       ptrs_and_offset.first,
                                                       ptrs_and_offset.second,
                                                       col_energy);
                    }
                    else
                    {
                        single_bitstring_diagonal(
                            input_bitsets[*col_ptr].first, diag_oper.terms, col_energy);
                    }
                    energy_amp = current_prefactors[kk] * std::pow(std::abs(val), 2) /
                                 (target_energy - col_energy + 1e-15);
                    // If the amplitude is larger than tol
                    if(std::abs(energy_amp) > tol)
                    {
                        //if col not in out_subspace then add the column to the output subspace
                        // and add the col_ptr to the next rows array and add a new prefactor
                        // to the next_prefactors
                        out_col_ptr = out_subspace.get_ptr2(col_vec);
                        if(out_col_ptr == nullptr)
                        {
                            est_energy += target_energy * energy_amp;
                            next_rows.push_back(*col_ptr);
                            next_prefactors.push_back(energy_amp /
                                                      (target_energy - col_energy + 1e-15));
                            out_subspace.emplace(col_vec, num_inserted_bitsets);
                            num_inserted_bitsets += 1;
                        }
                    }
                }
            }
        }
    }
    out_subspace.set_bucket_occupancy();
    return est_energy;
}
