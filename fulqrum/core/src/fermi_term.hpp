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
#include "qubit_term.hpp"
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


/**
 * Compute the JW phase for a given operator
 *
 * @param[in] op The operator in question
 * 
 * @return Integer phase value
 */
inline int jw_phase(const unsigned char op)
{
    int out;
    switch (op)
    {
    case 5: //minus sign if op = '-'
        out = -1;
        break;
    case 2: //minus sign if op = '1'
        out = -1;
        break;
    default:
        out = 1;
        break;
    }
    return out;
}


/**
 * Compute the extended JW transformation for a single Fermionic term
 *
 * @param[in] fermi_term Input Fermionic term
 * @param[in,out] qubit_term Output qubit term
 */
void jw_term(const FermionicTerm_t& fermi_term, OperatorTerm_t& qubit_term)
{
    int num_elems = fermi_term.indices.size();
    int kk, mm;
    unsigned int jj;
    int phase = 1;
    unsigned int current_ind;
    unsigned char current_val;
    qubit_term.coeff = fermi_term.coeff;
    qubit_term.extended = (num_elems > 0);
    //Start with do_z = 0 since nothing has been done yet
    int do_z = 0;
    for(kk = num_elems - 1; kk > -1; kk--)
    {
        current_ind = fermi_term.indices[kk];
        current_val = fermi_term.values[kk];
        // Add start element to qubit operator
        qubit_term.indices.push_back(current_ind);
        qubit_term.values.push_back(current_val);
        // If a Z term acts on the current value then need to account
        // for the phase factor in the coefficient
        if(do_z)
        {
            phase *= jw_phase(current_val);
        }
        // update do_z with this operator
        do_z ^= (current_val > 4);
        // if not at last element in num_elems and do_z
        // make every id element between start and the next elem a Z operator
        if(kk && do_z)
        {
            for(jj = current_ind - 1; jj > fermi_term.indices[kk - 1]; jj--)
            {
                qubit_term.indices.push_back(jj);
                qubit_term.values.push_back(0);
            }
        }
        // If only one element exists then kk=0 but I still need to
        // add Z operators down to zero
        else if(num_elems == 1 && do_z)
        {
            for(mm = current_ind - 1; mm > -1; mm--)
            {
                qubit_term.indices.push_back(mm);
                qubit_term.values.push_back(0);
            }
        }
    } // end kk loop
    qubit_term.coeff *= phase; // multiple coefficient by phase factor
}



// Converts a regular value index into a deflated one
inline int collapse_value(unsigned char x)
{
    int out;
    switch (x)
    {
    case 1:
        out = 0;
        break;
    case 2:
        out = 1;
        break;
    case 5:
        out = 2;
        break;
    default: //  x=6
        out = 3;
        break;
    }
    return out;
}

inline void deflate_term_indices(const FermionicTerm& term, std::vector<FermionicTerm>& out_terms, 
                                 const std::vector<int>& collapsed_values)
{
    unsigned int num_elems = term.indices.size();
    std::size_t kk, num_touched;
    FermionicTerm_t new_term = FermionicTerm();
    unsigned int current_index;
    int temp_int;
    unsigned char current_value;

    num_touched = 0;
    while(num_touched < num_elems)
    {
        current_index = term.indices[num_touched];
        current_value = term.values[num_touched];
        num_touched += 1;
        for(kk=num_touched; kk < num_elems; kk++)
        {
            // next term has a matching index with the current one
            if(term.indices[kk] == current_index)
            {
                temp_int = collapsed_values[4*collapse_value(current_value) + collapse_value(term.values[kk])];
                // This operator becomes a null operator return
                if(temp_int < 0)
                {
                    return;
                }
                else
                {
                    current_value = static_cast<unsigned char>(temp_int);
                }
                num_touched += 1;
            }
            else
            {   // Move on to next index since not matching and we assume we index sorted already
                break;
            }
        }
        new_term.indices.push_back(current_index);
        new_term.values.push_back(current_value);
    }
    new_term.coeff = term.coeff;
    out_terms.push_back(new_term);
}