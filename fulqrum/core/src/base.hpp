/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <complex>
#include <boost/dynamic_bitset.hpp>


const std::size_t MAX_SIZE_T = (std::size_t)-1;
const unsigned int MAX_UINT = (unsigned int)-1;
const unsigned int BITS_PER_BLOCK = 8 * sizeof(std::size_t);


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
    int sorted {0};
    int weight_sorted {0};
    int off_weight_sorted {0};
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
 * @brief Datastructure for subspace defined by counts
 *
 * @var bitstrings The subspace bit-strings sorted by bin_width 
 * @var bin_counts number of bit-strings in each bin
 * @var bin_ranges The range (indices) over which each bin is defined
 * @var num_qubits The number of qubits, i.e length of bitstrings
 * @var num_bins The number of bins
 * @var bin_width The bin_width used in the partial sorting
 * @var size Dimenion / number of bit-strings in the subpsace
 */
 typedef struct Subspace{
    std::vector<boost::dynamic_bitset<std::size_t> > bitstrings;
    std::vector<std::size_t> bin_counts;
    std::vector<std::size_t> bin_ranges;
    unsigned int num_qubits;
    std::size_t num_bins;
    std::size_t bin_width;
    std::size_t size;
} Subspace_t;
