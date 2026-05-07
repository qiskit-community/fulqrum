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
#include "constants.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <string>
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
    std::vector<unsigned char> values;
    std::vector<width_t> indices;
    std::complex<double> coeff;
    std::vector<width_t> proj_indices;
    std::vector<width_t> proj_bits;
    width_t offdiag_weight{0};
    int extended{0};
    int real_phase{1}; // 'phase' of real part (+/- 1), 0 means operator is complex-valued
    int group{-1}; // -1 means unset here

    OperatorTerm() {}
    OperatorTerm(std::complex<double> c)
        : coeff(c)
    {} // Init empty term with given coefficient
    OperatorTerm(std::string vals, std::vector<width_t> inds, std::complex<double> c)
        : coeff(c)
    {
        //check that length of values == length of indices
        if(vals.size() != inds.size())
        {
            throw std::runtime_error("Size of input string does not equal that of indices");
        }
        unsigned char val;
        unsigned int counter = 0;
        // Iterate over string of values, mapping to new values and adding to term
        for(std::string::iterator it = vals.begin(); it != vals.end(); ++it)
        {
            counter += 1;
            if(*it == 73) // identity operator
            {
                continue;
            }
            else
            {
                val = oper_map[*it];
                values.push_back(val);
                indices.push_back(inds[counter - 1]);
                offdiag_weight += static_cast<width_t>(val > 2);
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
    // destructor
    ~OperatorTerm()
    {
        std::vector<unsigned char>().swap(values);
        std::vector<width_t>().swap(indices);
        std::vector<width_t>().swap(proj_indices);
        std::vector<width_t>().swap(proj_bits);
    }
    /**
     * Inplace multiplication by a complex value
     */
    OperatorTerm& operator*=(std::complex<double> c)
    {
        coeff *= c;
        return *this;
    }
    OperatorTerm copy() const
    {
        OperatorTerm out = OperatorTerm(this->coeff);
        out.values = this->values;
        out.indices = this->indices;
        out.proj_indices = this->proj_indices;
        out.proj_bits = this->proj_bits;
        out.offdiag_weight = this->offdiag_weight;
        return out;
    }
    /**
     * Term multiplication by a complex number
     */
    friend OperatorTerm operator*(OperatorTerm& op, std::complex<double> c)
    {
        OperatorTerm out = op.copy();
        out.coeff *= c;
        return out;
    }
    /**
     * Term multiplication by a complex number
     */
    friend OperatorTerm operator*(std::complex<double> c, OperatorTerm& op)
    {
        OperatorTerm out = op.copy();
        out.coeff *= c;
        return out;
    }
    /**
     * Return the size of the term
     */
    std::size_t size() const
    {
        return indices.size();
    }
    /**
     * Return the weight (num. non-identity) operators
     * 
     * @param[out] weight The weight of the term
     */
    width_t weight() const
    {
        return static_cast<width_t>(indices.size());
    }
    /**
     * Sorting of indices and values for Operator term data
     */
    OperatorTerm& sort_term_data()
    {
        std::size_t n = indices.size();
        for(std::size_t i = 1; i < n; i++)
        {
            width_t key = indices[i];
            char val = values[i];
            std::size_t j =
                std::lower_bound(indices.begin(), indices.begin() + i, key) - indices.begin();

            for(std::size_t k = i; k > j; k--)
            {
                indices[k] = indices[k - 1];
                values[k] = values[k - 1];
            }
            indices[j] = key;
            values[j] = val;
        }
        return *this;
    }
    /**
     * Set the projector indices and bits for term
     */
    OperatorTerm& set_proj_indices()
    {
        std::size_t kk;
        width_t val;
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
        return *this;
    }
    std::vector<OpData> operators() const
    {
        std::vector<OpData> out;
        for(std::size_t kk = 0; kk < indices.size(); kk++)
        {
            OpData item{std::string(1, static_cast<char>(rev_oper_map[values[kk]])), indices[kk]};
            out.push_back(item);
        }
        return out;
    }
    /**
     * Is the term diagonal
     */
    bool is_diagonal() const
    {
        std::size_t kk;
        bool diag = 1;
        for(kk = 0; kk < values.size(); kk++)
        {
            if(values[kk] > 2)
            {
                diag = 0;
                break;
            }
        }
        return diag;
    }
} OperatorTerm_t;

/**
 * In-pace set the projector indices and bits for term in a Hamiltonian
 */
inline OperatorTerm& set_proj_indices(OperatorTerm& term)
{
    std::size_t kk;
    width_t val;
    term.proj_indices.resize(0);
    term.proj_bits.resize(0);
    for(kk = 0; kk < term.values.size(); kk++)
    {
        val = term.values[kk];
        if(val == 1 || val == 2)
        {
            term.proj_indices.push_back(term.indices[kk]);
            term.proj_bits.push_back(val - 1);
        }
    }
    return term;
}
