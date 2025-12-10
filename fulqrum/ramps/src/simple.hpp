/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <complex>
#include <boost/dynamic_bitset.hpp>

#include "../../core/src/base.hpp"
#include "../../core/src/bitset_utils.hpp"
#include "../../core/src/constants.hpp"
#include "../../core/src/bitset_hashmap.hpp"
#include "../../core/src/elements.hpp"
#include "../../core/src/operators.hpp"
#include "../../core/src/diag.hpp"



double simple_refinement(const std::vector<OperatorTerm_t>& diag_terms,
                         const std::vector<OperatorTerm_t>& off_terms,
                         const boost::dynamic_bitset<std::size_t> &start,
                         const bitset_map_namespace::BitsetHashMapWrapper &subspace,
                         bitset_map_namespace::BitsetHashMapWrapper &out_subspace,
                         unsigned int max_recursion,
                         const std::size_t *__restrict group_ptrs,
                         const std::size_t *__restrict group_ladder_ptrs,
                         const unsigned int *__restrict group_rowint_length,
                         const std::vector<std::vector<unsigned int>> &group_offdiag_inds,
                         const std::size_t num_groups,
                         const unsigned int ladder_offset)
{
    std::size_t recur, kk;
    const auto * input_bitsets = subspace.get_bitsets();
    auto * output_bitsets = out_subspace.get_bitsets();

    unsigned int current_idx = 0;
    unsigned int previous_idx = 0;

    double min_energy = 0;
    single_bitstring_diagonal(output_bitsets[current_idx].first, diag_terms, min_energy);

    std::vector<std::size_t> current_rows;
    std::vector<std::size_t> next_rows = {output_bitsets[current_idx].second};

    std::vector<double> current_prefactors;
    std::vector<double> next_prefactors = {1.0/min_energy};
    
    double est_energy = min_energy;
    
    const OperatorTerm_t *term;
    boost::dynamic_bitset<std::size_t> col_vec;
    const std::vector<unsigned int> *group_inds;
    std::size_t *col_ptr;
    std::size_t idx;
    std::size_t group;
    std::size_t group_int_start, group_int_stop;
    double val;
    unsigned int row_int;
    int do_col_search;

    for (recur=0; recur < max_recursion; recur++)
    {
        current_rows = next_rows;
        current_prefactors = next_prefactors;
        next_rows.clear();
        next_prefactors.clear();
        for (kk=0; kk < current_rows.size(); kk++)
        {
            const boost::dynamic_bitset<size_t> &row = input_bitsets[current_rows[kk]].first;
            std::vector<uint8_t> row_set_bits(row.size(), 0);
            for (size_t block = 0; block < row.num_blocks(); block++)
            {
                auto bitset = row.m_bits[block];
                while (bitset != 0)
                {
                    uint64_t t = bitset & -bitset;
                    int r = __builtin_ctzll(bitset);
                    row_set_bits[block * BITS_PER_BLOCK + r] = 1;
                    bitset ^= t;
                }
            }
            
            for (group = 0; group < num_groups; group++)
            {
                group_inds = &group_offdiag_inds[group];
                row_int = bitset_ladder_int(row_set_bits.data(), group_inds->data(), group_rowint_length[group]);
                group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
                group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];
                do_col_search = 1;
                val = 0;
            }
        }
    }
    return est_energy;
}