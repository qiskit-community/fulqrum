/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <complex>
#include <boost/dynamic_bitset.hpp>
#include "constants.hpp"
#include "bitset_hashmap.hpp"

/** @brief Data structure for each operator term, i.e. 'word' in the operator
 *
 * @var indices the qubits (locations) where non-idenity term operators are
 * @var values are the char representations of the operators
 * @var coeff is the complex coeffcient multiplying the term
 * @var offdiag_weight is the number of non-diagonal operators in the term
 */
typedef struct OperatorTerm{
    std::complex<double> coeff;
    std::vector<unsigned int> indices;
    std::vector<unsigned char> values;
    std::vector<unsigned int> proj_indices;
    std::vector<unsigned int> proj_bits;
    unsigned int offdiag_weight {0};
    int extended {0};
    int real_phase {1}; // 'phase' of real part (+/- 1), 0 means operator is complex-valued
    int group {-1}; // -1 means unset here
} OperatorTerm_t;


/** @struct QubitOperator
 * @brief Data structure for each a qubit operator, i.e. a collection of 'words'
 *
 * @var width is the number of qubits
 * @var terms is a vector of OperatorTerms that make up the operator
 * @var sorted is a flag that indicates the term is sorted (NOT USED AT PRESENT)
 */
typedef struct QubitOperator{
    unsigned int width;
    std::vector<OperatorTerm_t> terms;
    int type {1};
    unsigned int ladder_width {DEFAULT_LADDER_WIDTH};
    int sorted {0};
    int weight_sorted {0};
    int off_weight_sorted {0};
    int ladder_sorted {0};
} QubitOperator_t;


/** @brief Data structure for each Fermionic operator term
 *
 * @var indices the modes (locations) where non-idenity term operators are
 * @var values are the char representations of the operators
 * @var coeff is the complex coeffcient multiplying the term
 */
 typedef struct FermionicTerm{
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
 typedef struct FermionicOperator{
    unsigned int width;
    std::vector<FermionicTerm_t> terms;
} FermionicOperator_t;


/** @struct subspace
 * @brief Data structure for subspace defined by counts
 *
 * @var bitstrings The subspace bit-strings stored in a hash table
 * @var num_qubits The number of qubits, i.e length of bitstrings
 * @var size Dimenion / number of bit-strings in the subpsace
 */
 typedef struct Subspace{
    bitset_map_namespace::BitsetHashMapWrapper bitstrings;
    unsigned int num_qubits;
    std::size_t size;
} Subspace_t;

