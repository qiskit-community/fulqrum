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
#include "fermi_term.hpp"
#include "qubit_oper.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>



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
    FermionicOperator() {}
    /**
     * Constructor building an empty operator with a given width
     *
     * @param[in] width The width (number of qubits) of the operator
     */
    FermionicOperator(unsigned int x)
    {
        width = x;
    }
    FermionicOperator(unsigned int x, std::vector<TermData> data)
        : width(x)
    {
        unsigned int num_terms = data.size();
        std::size_t kk;
        TermData tdata;
        for(kk = 0; kk < num_terms; kk++)
        {
            tdata = data[kk];
            _validate_indices(std::get<1>(tdata),
                              width); // validate that all indices are less than operator width
            terms.push_back(
                FermionicTerm(std::get<0>(tdata), std::get<1>(tdata), std::get<2>(tdata)));
        }
    }
    // deallocation
    ~FermionicOperator()
    {
        std::vector<FermionicTerm_t>().swap(terms);
    }
    /**
     * Print object to standard output stream
     */
    friend auto operator<<(std::ostream& os, const FermionicOperator& self) -> std::ostream&
    {
        std::size_t num_terms = self.size();
        std::size_t total_terms = num_terms;
        FermionicTerm_t term;
        int too_many_terms = 0;
        std::size_t kk, jj;

        // restrict to outputting at most 100 terms
        if(num_terms > 100)
        {
            too_many_terms = 1;
            num_terms = 100;
        }
        os << "<FermionicOperator["; // start output here
        for(kk = 0; kk < num_terms; kk++)
        {
            term = self.terms[kk];
            os << "{";
            for(jj = 0; jj < term.indices.size(); jj++)
            {
                os << rev_oper_map[term.values[jj]] << ":" << term.indices[jj];
                if(jj != term.indices.size() - 1)
                {
                    os << " ";
                }
            }
            os << ", " << term.coeff;
            os << "}";
            if(kk != num_terms - 1)
            {
                os << ", ";
            }
        }
        if(too_many_terms)
        {
            os << " + " << (total_terms - 100) << "terms";
        }
        return os << ", width=" << self.width << "]>";
    }
    /**
     * Return the size of the operator
     */
    std::size_t size() const
    {
        return terms.size();
    }
} FermionicOperator_t;


/**
 * Extended JW transformation
 *
 * @param[in] fermi Input FermionicOperator
 * @param[in,out] out Output QubitOperator
 */
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
        set_offdiag_weight_and_phase(out.terms[kk]);
        set_extended_flag(out.terms[kk]);
        out.terms[kk].set_proj_indices();
    }
}
