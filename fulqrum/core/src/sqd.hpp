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

#include "subspace.hpp"
#include <boost/dynamic_bitset.hpp>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "qiskit/addon/sqd/configuration_recovery.hpp"
#include "qiskit/addon/sqd/postselection.hpp"
#include "qiskit/addon/sqd/subsampling.hpp"

using bitset_t = boost::dynamic_bitset<std::size_t>;
using bitvec_t = std::vector<bitset_t>;
using wvec_t = std::vector<double>;
using matcher_t = Qiskit::addon::sqd::MatchesRightLeftHamming<uint32_t>;

std::pair<bitvec_t, wvec_t> postselect_bitstrings_cpp(const bitvec_t& bitstrings,
                                                      const wvec_t& weights,
                                                      const uint32_t& right,
                                                      const uint32_t& left)
{
    auto matcher = matcher_t(right, left);
    return Qiskit::addon::sqd::postselect_bitstrings<bitvec_t, wvec_t, matcher_t>(
        bitstrings, weights, matcher);
}

bitvec_t subsample_cpp(const bitvec_t& bitstrings,
                       const wvec_t& weights,
                       const unsigned int& samples_per_batch,
                       const uint32_t& seed)
{
    std::mt19937 rng(seed);
    return Qiskit::addon::sqd::subsample(bitstrings, weights, samples_per_batch, rng);
}

std::pair<bitvec_t, wvec_t> recover_configurations_cpp(const bitvec_t& bitstrings,
                                                       const wvec_t& weights,
                                                       const std::vector<double>& avg_occupancies_a,
                                                       const std::vector<double>& avg_occupancies_b,
                                                       const std::uint64_t num_elec_a,
                                                       const std::uint64_t num_elec_b,
                                                       const uint32_t seed)
{
    std::mt19937 rng(seed);
    std::array<std::vector<double>, 2> avg_occupancies = {avg_occupancies_a, avg_occupancies_b};
    std::array<uint64_t, 2> num_elec = {num_elec_a, num_elec_b};
    return Qiskit::addon::sqd::recover_configurations(
        bitstrings, weights, avg_occupancies, num_elec, rng);
}
