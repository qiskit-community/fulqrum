/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <complex>

#include "base.hpp"


void jw_term(FermionicTerm_t& fermi_term, OperatorTerm_t& qubit_term)
{
    std::size_t num_elems = fermi_term.indices.size();
    std::size_t kk, jj, current_ind;
    unsigned char current_val;
    qubit_term.coeff = fermi_term.coeff;
    qubit_term.extended = (num_elems > 0); // Because any Fermi elems are extended
    // Start with do_z = 0 since nothing has been done yet
    int do_z = 0;
    for(kk=num_elems-1; kk > -1; kk--)
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
            switch (current_val)
            {
                case 5: // minus sign if op = -
                    qubit_term.coeff *= -1;
                    break;
                case 2: // minus sign if op = 1
                    qubit_term.coeff *= -1;
                    break;
            }
        }
        // update do_z with this operator
        do_z ^= (current_val > 4);
        // if not at last element in num_elems and do_z
        // make every id element between start and the next elem a Z operator
        if(kk && do_z)
        {
            for(jj=current_ind-1; jj>fermi_term.indices[kk-1]; jj--)
            {
                qubit_term.indices.push_back(jj);
                qubit_term.values.push_back(0);
            }
        }
        // If only one element exists then kk=0 but I still need to
        // add Z operators down to zero
        else if(num_elems==1 && do_z)
        {
            for(jj=current_ind-1; jj>-1; jj--)
            {
                qubit_term.indices.push_back(jj);
                qubit_term.values.push_back(0);
            }
        }
    }
}

