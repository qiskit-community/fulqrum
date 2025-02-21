/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <vector>
#include <complex>


/**
 * Data structure for each operator term, i.e. 'word' in the operator
 *
 * Indices are the qubits (locations) where non-idenity term operators are
 * Values are the char representations of the operators
 * coeff is the complex coeffcient multiplying the term
 * offdiag_weight is the number of non-diagonal operators in the term
 */
typedef struct OperatorTerm{
    std::complex<double> coeff;
    std::vector<std::size_t> indices;
    std::vector<char> values;
    std::size_t offdiag_weight {0};
} OperatorTerm_t;


/**
 * Data structure for each a qubit operator, i.e. a collection of 'words'
 *
 * width is the number of qubits
 * terms is a vector of OperatorTerms that make up the operator
 * sorted is a flag that indicates the term is sorted (NOT USED AT PRESENT)
 */
typedef struct QubitOperator{
    std::size_t width;
    std::vector<OperatorTerm_t> terms;
    int sorted {0};
} QubitOperator_t;

