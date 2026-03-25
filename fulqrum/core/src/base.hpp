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
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <algorithm>


typedef std::tuple<std::string, std::vector<unsigned int>, std::complex<double>> TermData;

// Map converting standard char values into continuous values
std::unordered_map<unsigned char, unsigned char> oper_map =
    {
        {90, 0}, {48, 1}, {49, 2}, {88, 3}, {89, 4}, {45, 5}, {43, 6}
    };



/** @brief Data structure for each operator term, i.e. 'word' in the operator
 *
 * @var indices the qubits (locations) where non-identity term operators are
 * @var values are the char representations of the operators
 * @var coeff is the complex coefficient multiplying the term
 * @var offdiag_weight is the number of non-diagonal operators in the term
 */
typedef struct OperatorTerm
{
    std::vector<unsigned char> values;
    std::vector<unsigned int> indices;
    std::complex<double> coeff;
    std::vector<unsigned int> proj_indices;
    std::vector<unsigned int> proj_bits;
    unsigned int offdiag_weight{0};
    int extended{0};
    int real_phase{1}; // 'phase' of real part (+/- 1), 0 means operator is complex-valued
    int group{-1}; // -1 means unset here

    OperatorTerm() {}
    OperatorTerm(std::string vals, std::vector<unsigned int> inds, std::complex<double> c): indices(inds), coeff(c)
    {
        // Iterate over string of values, mapping to new values and adding to term
        for(std::string::iterator it = vals.begin(); it != vals.end(); ++it)
        {
            if(*it == 73)
            {
                throw std::runtime_error("Cannot use identity operators in sparse format.");
            }
            else{
                values.push_back(oper_map[*it]);
            }
        }
        //check that length of values == length of indices
        if(values.size() != indices.size())
        {
            throw std::runtime_error("Size of values vector does not equal that of indices.");
        }
        sort_term_data(); // sort term data from low -> high indices
        set_proj_indices(); // set projection operator indices, if any
    }
    /**
     * Sorting of indices and values for Operator term data
     */
    void sort_term_data()
    {
        std::size_t n = indices.size();
        for(std::size_t i = 1; i < n; i++)
        {
            unsigned int key = indices[i];
            char val = values[i];
            std::size_t j = std::lower_bound(indices.begin(), indices.begin() + i, key) - indices.begin();

            for(std::size_t k = i; k > j; k--)
            {
                indices[k] = indices[k - 1];
                values[k] = values[k - 1];
            }
            indices[j] = key;
            values[j] = val;
        }
    }
    /**
     * Set the projector indices and bits for term
     */
    void set_proj_indices()
    {
        std::size_t kk;
        unsigned int val;
        proj_indices.resize(0);
        proj_bits.resize(0);
        for(kk = 0; kk < values.size(); kk++)
        {
            val = values[kk];
            if(val == 1 || val == 2)
            {
                proj_indices.push_back(indices[kk]);
                proj_bits.push_back(val - 1);
            }
        }
    }
} OperatorTerm_t;


/**
 * Validate that term indices are less than operator width
 *
 * @param[in] indices Indices for the given term
 * @param[in] width The operator width
 */
void _validate_indices(std::vector<unsigned int>& inds, unsigned int width){
    std::size_t size = inds.size();
    for(std::size_t kk=0; kk < size; kk++)
    {
        if(inds[kk] >= width)
        {
            throw std::runtime_error("Index is larger than the operator width.");
        }
    }
}


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

    QubitOperator() {}
    /**
     * Constructor building an empty operator with a given width
     *
     * @param[in] width The width (number of qubits) of the operator
     */
    QubitOperator(unsigned int x){width = x;}

    QubitOperator(unsigned int x, std::vector<TermData> data): width(x) {
       unsigned int num_terms = data.size();
       std::size_t kk;
       TermData tdata;
       for(kk =0; kk < num_terms; kk++)
       {
        tdata = data[kk];
        _validate_indices(std::get<1>(tdata), width); // validate that all indices are less than operator width
        terms.push_back(OperatorTerm(std::get<0>(tdata), std::get<1>(tdata), std::get<2>(tdata)));
       }
    }
    /**
     * The number of terms in the operator
     *
     * @param[out] size The number of terms in the operator
     */
    std::size_t size(){return terms.size();}
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
