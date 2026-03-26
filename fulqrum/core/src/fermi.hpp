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
#include <cstddef>
#include <vector>

#include "base.hpp"
#include "operators.hpp"

inline int jw_phase(unsigned char op)
{
    int out = 1;
    if(op == 5) //minus sign if op = -
    {
        out = -1;
    }
    else if(op == 2) //minus sign if op = 1
    {
        out = -1;
    }
    return out;
}

void jw_term(const FermionicTerm_t& fermi_term, OperatorTerm_t& qubit_term)
{
    int num_elems = fermi_term.indices.size();
    int kk;
    unsigned int jj;
    int mm;
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
            qubit_term.coeff *= jw_phase(current_val);
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
}

void extended_jw_transform(const FermionicOperator_t& fermi,
                           QubitOperator_t& out,
                           std::size_t num_terms)
{
    std::size_t kk;
#pragma omp parallel for if(num_terms > 128)
    for(kk = 0; kk < num_terms; kk++)
    {
        jw_term(fermi.terms[kk], out.terms[kk]);
        out.terms[kk].sort_term_data();
        set_offdiag_weight(out.terms[kk]);
        out.terms[kk].set_proj_indices();
    }
}


void insertion_sort_term(FermionicTerm_t& term)
{
    std::size_t kk;
    int ll;
    std::size_t num_elems = term.indices.size();
    unsigned int temp_index;
    unsigned char temp_value;
    int prefactor = 1;
    for(kk=1; kk < num_elems; kk++)
    {
        temp_index = term.indices[kk];
        temp_value = term.values[kk];
        ll = kk - 1;
        // Only switch elements if they are of different indices
        // In this case we always pick up a minus sign that
        // we need to keep track of with the 'prefactor'
        while(ll >= 0 && temp_index < term.indices[ll])
        {
            term.indices[ll + 1] = term.indices[ll];
            term.values[ll + 1] = term.values[ll];
            // Only add a minus sign if both operators (values)
            // are not projectors (ie. > 4 since '-'=5 and '+'=6)
            if((temp_value > 4) and (term.values[ll] > 4))
            {
                prefactor *= -1;
            }
            ll -= 1;
        }
        term.indices[ll + 1] = temp_index;
        term.values[ll + 1] = temp_value;
    }
    term.coeff *= prefactor;
}