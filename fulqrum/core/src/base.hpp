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
#include "bitset_hashmap.hpp"
#include "constants.hpp"
#include <boost/dynamic_bitset.hpp>
#include <complex>
#include <cstdlib>
#include <vector>

/** @brief Data structure for each operator term, i.e. 'word' in the operator
 *
 * @var indices the qubits (locations) where non-identity term operators are
 * @var values are the char representations of the operators
 * @var coeff is the complex coefficient multiplying the term
 * @var offdiag_weight is the number of non-diagonal operators in the term
 */
typedef struct OperatorTerm
{
	std::complex<double> coeff;
	std::vector<unsigned int> indices;
	std::vector<unsigned char> values;
	std::vector<unsigned int> proj_indices;
	std::vector<unsigned int> proj_bits;
	unsigned int offdiag_weight{0};
	int extended{0};
	int real_phase{1}; // 'phase' of real part (+/- 1), 0 means operator is complex-valued
	int group{-1}; // -1 means unset here
} OperatorTerm_t;

/** @struct QubitOperator
 * @brief Data structure for each a qubit operator, i.e. a collection of 'words'
 *
 * @var width is the number of qubits
 * @var terms is a vector of OperatorTerms that make up the operator
 * @var sorted is a flag that indicates the term is sorted (NOT USED AT PRESENT)
 */
typedef struct QubitOperator
{
	unsigned int width;
	std::vector<OperatorTerm_t> terms;
	int type{1};
	unsigned int ladder_width{DEFAULT_LADDER_WIDTH};
	int sorted{0};
	int weight_sorted{0};
	int off_weight_sorted{0};
	int ladder_sorted{0};
} QubitOperator_t;

/** @brief Data structure for each Fermionic operator term
 *
 * @var indices the modes (locations) where non-identity term operators are
 * @var values are the char representations of the operators
 * @var coeff is the complex coefficient multiplying the term
 */
typedef struct FermionicTerm
{
	std::complex<double> coeff;
	std::vector<unsigned int> indices;
	std::vector<unsigned char> values;
} FermionicTerm_t;

/** @struct FermionicOperator
 * @brief Data structure for each a qubit operator, i.e. a collection of 'words'
 *
 * @var width is the number of qubits
 * @var terms is a vector of OperatorTerms that make up the operator
 * @var sorted is a flag that indicates the term is sorted (NOT USED AT PRESENT)
 */
typedef struct FermionicOperator
{
	unsigned int width;
	std::vector<FermionicTerm_t> terms;
} FermionicOperator_t;

/** @struct subspace
 * @brief Data structure for subspace defined by counts
 *
 * @var bitstrings The subspace bit-strings stored in a hash table
 * @var num_qubits The number of qubits, i.e length of bitstrings
 * @var size Dimension / number of bit-strings in the subspace
 */
typedef struct Subspace
{
	bitset_map_namespace::BitsetHashMapWrapper bitstrings;
	unsigned int num_qubits;
	std::size_t size;
} Subspace_t;
