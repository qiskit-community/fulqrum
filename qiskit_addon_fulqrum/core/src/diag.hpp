/**
 * This code is a Qiskit project.
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
#include <cstdlib>
#include <cstddef>
#include <complex>
#include <vector>

#include "base.hpp"
#include "elements.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include <boost/dynamic_bitset.hpp>


/**
 * Populate the diagonal vector for a given diagonal operator
 *
 *
 * @param data The subspace data
 * @param diag_vec The diagonal vector to store information to
 * @param val Variable storing the element value
 * @param diag_oper The diagonal operator
 * @param width The width of the operator
 * @param subspace_dim The dimension of the subspace
 */
template <typename T>
void compute_diag_vector(const bitset_map_namespace::BitsetHashMapWrapper &data,
                         T *__restrict diag_vec,
                         const QubitOperator_t &diag_oper,
                         const unsigned int width,
                         const std::size_t subspace_dim)
{
    std::size_t kk;
    const auto *bitsets = data.get_bitsets();

    #pragma omp parallel for if (subspace_dim > 4096)
    for (kk = 0; kk < subspace_dim; kk++)
    {
        T val = 0;
        const boost::dynamic_bitset<size_t>& row = bitsets[kk].first;
        single_bitstring_diagonal(row, diag_oper.terms, val);
        diag_vec[kk] = val;
    }
}


/**
 * Compute the diagonal matrix-element for a single bit-string
 *
 *
 * @param row The row bit-string
 * @param diag_terms The diagonal operator
 * @param val Variable storing the element value
 */
template <typename T>
inline void single_bitstring_diagonal(const boost::dynamic_bitset<size_t>& row,
                                      const std::vector<OperatorTerm_t>& diag_terms,
                                      T& val)
{
    val = 0;
    const std::size_t num_terms = diag_terms.size();
    const OperatorTerm_t *term;
    unsigned int weight;
    std::size_t ll;
    for (ll = 0; ll < num_terms; ll++)
    {
        term = &diag_terms[ll];
        weight = term->indices.size();
        if (passes_proj_validation(term, row))
        {
            accum_element(row, row,
                          &term->indices[0], &term->values[0], term->coeff, term->real_phase,
                          weight, val);
        }
    }
}
