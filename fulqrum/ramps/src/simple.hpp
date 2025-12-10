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
                      bitset_map_namespace::BitsetHashMapWrapper &out_subspace)
{
    const auto * input_bitsets = subspace.get_bitsets();
    auto * output_bitsets = out_subspace.get_bitsets();

    unsigned int current_idx = 0;
    unsigned int previous_idx = 0;

    double energy = 0;
    single_bitstring_diagonal(output_bitsets[current_idx].first, diag_terms, energy);
    return energy;
}