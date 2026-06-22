/**
 * This code is a part of Fulqrum.
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
#include <cmath>
#include <string>
#include <vector>
#include <complex>
#include <stdexcept>

#include "../../core/src/fermi_oper.hpp"


inline std::size_t _flat_index2d(width_t i, width_t j, width_t dim)
{
    return i * dim + j;
}


inline std::size_t _flat_index4d(width_t i, width_t j, width_t k, width_t l, width_t dim)
{
    return i + j * dim + k * dim * dim + l * dim * dim * dim;
}

template <typename T>
inline FermionicOperator pyscf_integrals_to_fermionic(T * __restrict flat_one_body_integrals,
                                                      T * __restrict flat_two_body_integrals,
                                                      unsigned int ob_arr_len, unsigned int tb_arr_len,
                                                      std::complex<double> constant=0, double EQ_TOLERANCE=1e-12)
{
    width_t half_num_qubits = std::sqrt(ob_arr_len);
    width_t num_qubits = 2 * half_num_qubits;

    if(tb_arr_len != half_num_qubits * half_num_qubits * half_num_qubits * half_num_qubits)
    {
        throw std::runtime_error("Input flat_two_body_integral array does not match expected length");
    }

    std::string ob_str = "+-";   // One-body operator string (normal ordered)
    std::string tb_str = "++--"; // Two-body operator string (normal ordered)

    std::vector<width_t> qubit_mapping(num_qubits);
    width_t p, q, r, s, ii, jj, kk, ll;
    T temp_one_body, temp_two_body;

    for(kk=0; kk < num_qubits; kk++)
    {
        qubit_mapping[kk] = ((!(kk % 2)) * kk / 2) + ((kk % 2) * (kk / 2 + half_num_qubits));
    }

    FermionicOperator fop = FermionicOperator(num_qubits);
    if(std::abs(constant) > EQ_TOLERANCE)
    {
        fop += FermionicOperator(num_qubits, {{{}, {}, constant}});
    }

    for(p=0; p < half_num_qubits; p++)
    {
        for(q=0; q < half_num_qubits; q++)
        {
            temp_one_body = flat_one_body_integrals[_flat_index2d(p, q, half_num_qubits)];
            if(std::abs(temp_one_body) > EQ_TOLERANCE)
            {
                // Populate 1-body coefficients. Require p and q have same spin.
                ii = 2 * p;
                jj = 2 * q;
                fop += FermionicOperator(num_qubits, {{ob_str, {qubit_mapping[ii], qubit_mapping[jj]}, temp_one_body}});

                ii = 2 * p + 1;
                jj = 2 * q + 1;
                fop += FermionicOperator(num_qubits, {{ob_str, {qubit_mapping[ii], qubit_mapping[jj]}, temp_one_body}});
            }
            // Continue looping to prepare 2-body coefficients.
            for(r=0; r < half_num_qubits; r++)
            {
                for(s=0; s < half_num_qubits; s++)
                {
                    temp_two_body = flat_two_body_integrals[_flat_index4d(p, q, r, s, half_num_qubits)] / 2.0; 
                    if(std::abs(temp_two_body) > EQ_TOLERANCE)
                    {
                        // Mixed spin
                        ii = 2 * p;
                        jj = 2 * q + 1;
                        kk = 2 * r + 1;
                        ll = 2 * s;
                        fop += FermionicOperator(num_qubits, {{tb_str, {qubit_mapping[ii], qubit_mapping[jj], qubit_mapping[kk], qubit_mapping[ll]}, temp_two_body}});

                        ii = 2 * p + 1;
                        jj = 2 * q;
                        kk = 2 * r;
                        ll = 2 * s + 1;
                        fop += FermionicOperator(num_qubits, {{tb_str, {qubit_mapping[ii], qubit_mapping[jj], qubit_mapping[kk], qubit_mapping[ll]}, temp_two_body}});

                        // Same spin
                        ii = 2 * p;
                        jj = 2 * q;
                        kk = 2 * r;
                        ll = 2 * s;
                        fop += FermionicOperator(num_qubits, {{tb_str, {qubit_mapping[ii], qubit_mapping[jj], qubit_mapping[kk], qubit_mapping[ll]}, temp_two_body}});

                        ii = 2 * p + 1;
                        jj = 2 * q + 1;
                        kk = 2 * r + 1;
                        ll = 2 * s + 1;
                        fop += FermionicOperator(num_qubits, {{tb_str, {qubit_mapping[ii], qubit_mapping[jj], qubit_mapping[kk], qubit_mapping[ll]}, temp_two_body}});
                    }
                }
            }
        }
    }
    return fop;
}
