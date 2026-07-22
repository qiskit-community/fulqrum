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
#include <vector>

/**
 * In-pace set the projector indices and bits for term in a Hamiltonian
 */
template <typename T>
inline T& set_term_proj_indices(T& term)
{
    std::size_t kk;
    width_t val;
    term.proj_indices.resize(0);
    term.proj_bits.resize(0);
    term.proj_structure = 0;
    for(kk = 0; kk < term.values.size(); kk++)
    {
        val = term.values[kk];
        if(val == 1 || val == 2)
        {
            term.proj_indices.push_back(term.indices[kk]);
            term.proj_structure += term.indices[kk];
            term.proj_bits.push_back(val - 1);
        }
    }
    return term;
}
