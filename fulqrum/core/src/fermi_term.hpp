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
#include <unordered_map>
#include <vector>

// Fermionic components ---------------------------------------------------------------------------

/** @brief Data structure for each Fermionic operator term
 *
 * @var indices the modes (locations) where non-identity term operators are
 * @var values are the char representations of the operators
 * @var coeff is the complex coefficient multiplying the term
 */
typedef struct FermionicTerm
{
    std::vector<unsigned char> values;
    std::vector<unsigned int> indices;
    std::complex<double> coeff;

    FermionicTerm() {}
    FermionicTerm(std::complex<double> c)
        : coeff(c)
    {} // Init empty term with given coefficient
    FermionicTerm(std::string vals, std::vector<unsigned int> inds, std::complex<double> c)
        : indices(inds)
        , coeff(c)
    {
        // Iterate over string of values, mapping to new values and adding to term
        for(std::string::iterator it = vals.begin(); it != vals.end(); ++it)
        {
            if(*it == 73)
            {
                throw std::runtime_error("Cannot use identity operators in sparse format.");
            }
            else
            {
                values.push_back(oper_map[*it]);
            }
        }
        //check that length of values == length of indices
        if(values.size() != indices.size())
        {
            throw std::runtime_error("Size of values vector does not equal that of indices.");
        }
        insertion_sort();
    }
    // destructor
    ~FermionicTerm()
    {
        std::vector<unsigned char>().swap(values);
        std::vector<unsigned int>().swap(indices);
    }
    /**
     * Return the size of the term
     */
    std::size_t size() const
    {
        return indices.size();
    }
    /**
     * Insertion sort indices (and values) in the term
     */
    void insertion_sort()
    {
        std::size_t kk;
        int ll;
        std::size_t num_elems = indices.size();
        unsigned int temp_index;
        unsigned char temp_value;
        int prefactor = 1;
        for(kk = 1; kk < num_elems; kk++)
        {
            temp_index = indices[kk];
            temp_value = values[kk];
            ll = kk - 1;
            // Only switch elements if they are of different indices
            // In this case we always pick up a minus sign that
            // we need to keep track of with the 'prefactor'
            while(ll >= 0 && temp_index < indices[ll])
            {
                indices[ll + 1] = indices[ll];
                values[ll + 1] = values[ll];
                // Only add a minus sign if both operators (values)
                // are not projectors (ie. > 4 since '-'=5 and '+'=6)
                if((temp_value > 4) and (values[ll] > 4))
                {
                    prefactor *= -1;
                }
                ll -= 1;
            }
            indices[ll + 1] = temp_index;
            values[ll + 1] = temp_value;
        }
        coeff *= prefactor;
    }
} FermionicTerm_t;
