/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <complex>

#include "../../core/src/base.hpp"
#include "../../core/src/bitset_utils.hpp"
#include "../../core/src/constants.hpp"
#include "../../core/src/bitset_hashmap.hpp"
#include "../../core/src/elements.hpp"
#include "../../core/src/operators.hpp"
#include <boost/dynamic_bitset.hpp>


int simple_refinement(const OperatorTerm_t * diag_terms,
                      const OperatorTerm_t * off_terms,
                      const bitset_map_namespace::BitsetHashMapWrapper &subspace,
                      bitset_map_namespace::BitsetHashMapWrapper &out_subspace)
{
    return 0;
}