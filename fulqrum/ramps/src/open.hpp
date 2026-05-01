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

double open_ramps(const QubitOperator& oper,
                         bitset_map_namespace::BitsetHashMapWrapper& out_subspace,
                         QubitOperator& diag_oper,
                         const width_t width,
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
    bool do_fast_diag = fast_diag_compatible(diag_oper);
    std::pair<std::vector<std::pair<std::size_t, std::size_t>>, std::size_t> ptrs_and_offset;
    if(do_fast_diag)
    {
        diag_proj_index_sort(diag_oper);
        ptrs_and_offset = projector_ptrs_and_offset(diag_oper);
        std::cout << "Fast diagonal mode" << std::endl;
    }

    std::size_t recur, kk;
    auto* output_bitsets = out_subspace.get_bitsets();

    std::vector<boost::dynamic_bitset<std::size_t>> current_rows;
    std::vector<boost::dynamic_bitset<std::size_t>> next_rows;
    for(kk=0; kk < out_subspace.size(); kk++)
    {
        next_rows.push_back(output_bitsets[kk].first);
    }

    std::vector<double> current_prefactors;
    std::vector<double> next_prefactors(next_rows.size(), 1.0 / target_energy);

    double est_energy = target_energy;
    double col_energy = 0;
    double energy_amp = 0;

    const OperatorTerm_t* term;
    boost::dynamic_bitset<std::size_t> col_vec;
    const std::vector<width_t>* group_inds;
    std::size_t* out_col_ptr;
    std::size_t idx;
    std::size_t group;
    std::size_t group_int_start, group_int_stop;
    std::size_t num_inserted_bitsets = 1;
    std::complex<double> val = 0;
    unsigned int row_int;
    int do_col_search;

    struct Candidate
    {
        boost::dynamic_bitset<std::size_t> col_vec;
        double est_delta;
        double next_prefactor;
    };

    for(recur = 0; recur < max_recursion; recur++)
    {
        // Set current terms from previous recursion results
        current_rows = next_rows;
        current_prefactors = next_prefactors;
        next_rows.clear();
        next_prefactors.clear();
        std::vector<std::vector<Candidate>> pending_candidates(current_rows.size());
        // Loop over all rows in the current set
        #pragma omp parallel for private(col_vec,                                                  \
                                 group_inds,                                                       \
                                 out_col_ptr,                                                      \
                                 idx,                                                              \
                                 group,                                                            \
                                 group_int_start,                                                  \
                                 group_int_stop,                                                   \
                                 val,                                                              \
                                 row_int,                                                          \
                                 do_col_search,                                                    \
                                 col_energy,                                                       \
                                 energy_amp,                                                       \
                                 term) schedule(guided)
        for(kk = 0; kk < current_rows.size(); kk++)
        {
            const boost::dynamic_bitset<size_t>& row = current_rows.at(kk);
            std::vector<uint8_t> row_set_bits(row.size(), 0);
            std::vector<Candidate> row_candidates;
            bitset_to_bitvec(row, row_set_bits);

            for(group = 0; group < num_groups; group++)
            {
                group_inds = &group_offdiag_inds[group];
                row_int = bitset_ladder_int(row_set_bits.data(), group_inds->data(), group_rowint_length[group]);
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


                // Wwe need to compute the columns diagonal energy
                col_energy = 0;
                if(do_fast_diag)
                {
                    single_bitstring_diagonal_fast(col_vec, diag_oper.terms,
                                                   ptrs_and_offset.first,
                                                   ptrs_and_offset.second,
                                                   col_energy);
                }
                else
                {
                    single_bitstring_diagonal(col_vec, diag_oper.terms, col_energy);
                }
                energy_amp = current_prefactors.at(kk) * std::pow(std::abs(val), 2) /
                                (target_energy - col_energy + 1e-15);
                
                // If the amplitude is larger than tol then need to add column bit-string to output set
                if(std::abs(energy_amp) > tol)
                {
                    //if col not in out_subspace then add the column to the output subspace
                    // and add the col_ptr to the next rows array and add a new prefactor
                    // to the next_prefactors
                    out_col_ptr = out_subspace.get_ptr2(col_vec);
                    if(out_col_ptr == nullptr)
                    {
                        row_candidates.push_back(
                            {col_vec,
                             target_energy * energy_amp,
                             energy_amp / (target_energy - col_energy + 1e-15)});
                    }
                }
            }
            // add all the candidate row bit-strings for the given row to the pending candidates
            pending_candidates[kk] = std::move(row_candidates);
        }

        std::size_t total_candidates = 0;
        for(kk = 0; kk < pending_candidates.size(); kk++)
        {
            total_candidates += pending_candidates.at(kk).size();
            for(const Candidate& candidate : pending_candidates.at(kk))
            {
                out_col_ptr = out_subspace.get_ptr2(candidate.col_vec);
                if(out_col_ptr == nullptr)
                {
                    est_energy += candidate.est_delta;
                    next_rows.push_back(candidate.col_vec);
                    next_prefactors.push_back(candidate.next_prefactor);
                    out_subspace.emplace(candidate.col_vec, num_inserted_bitsets);
                    num_inserted_bitsets += 1;
                }
            }
        }
        std::cout << "recursion " << recur << " found " << total_candidates << " possible bit-strings " << std::endl;
    }
    out_subspace.set_bucket_occupancy();
    return est_energy;
}
