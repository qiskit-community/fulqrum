/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <cstddef>
#include <vector>
#include <complex>

#include "base.hpp"
#include "bitset_utils.hpp"
#include "bitset_hashmap.hpp"
#include "constants.hpp"
#include "elements.hpp"
#include "operators.hpp"
#include <boost/dynamic_bitset.hpp>

template <typename T>
void omp_matvec2(const std::vector<OperatorTerm_t> &terms,
                 const bitset_map_namespace::BitsetHashMapWrapper &subspace,
                 const T *__restrict diag_vec,
                 const std::size_t width,
                 const std::size_t subspace_dim,
                 const int has_nonzero_diag,
                 const std::size_t *__restrict group_ptrs,
                 const std::size_t *__restrict group_ladder_ptrs,
                 const unsigned int *__restrict group_rowint_length,
                 const std::vector<std::vector<unsigned int>> &group_offdiag_inds,
                 const unsigned int num_groups,
                 const unsigned int ladder_offset,
                 const T *__restrict in_vec,
                 T *__restrict out_vec)
{
    std::size_t kk;
    const auto *bitsets = subspace.get_bitsets();

    #pragma omp parallel if (subspace_dim > 4096)
    {
        // Take care of diagonal term first, if any (usually there is)
        if (has_nonzero_diag)
        {
            #pragma omp for
            for (kk = 0; kk < subspace_dim; kk++)
            {
                out_vec[kk] = diag_vec[kk] * in_vec[kk];
            }
        }

        std::size_t num_terms = terms.size();
        // Take care of off-diagonal terms
        if (num_terms)
        {
            #pragma omp for schedule(dynamic)
            for (kk = 0; kk < subspace_dim; kk++)
            {
                const boost::dynamic_bitset<size_t> &row = bitsets[kk].first;
                // TODO: Move it to a function as it is re-used in csr2.hpp
                // see csr2.hpp for background of this block
                std::vector<uint8_t> row_set_bits(row.size(), 0);
                for (size_t block = 0; block < row.num_blocks(); block++)
                {
                    auto bitset = row.m_bits[block];
                    while (bitset != 0)
                    {
                        uint64_t t = bitset & -bitset;
                        int r = __builtin_ctzll(bitset);
                        row_set_bits[block * BITS_PER_BLOCK + r] = 1;
                        bitset ^= t;
                    }
                }

                boost::dynamic_bitset<std::size_t> col_vec;
                T temp_val, val = 0;
                const OperatorTerm_t *term;
                unsigned int group;
                std::size_t group_int_start, group_int_stop;
                std::size_t idx;
                std::size_t *col_ptr;
                unsigned int row_int;
                const std::vector<unsigned int> *group_inds;
                int do_col_search;
                // Loop over all off-diagonal terms in operator
                for (group = 0; group < num_groups; group++)
                {
                    group_inds = &group_offdiag_inds[group];
                    row_int = bitset_ladder_int(row_set_bits.data(), group_inds->data(), group_rowint_length[group]);
                    do_col_search = 1;
                    group_int_start = group_ladder_ptrs[group * ladder_offset + row_int];
                    group_int_stop = group_ladder_ptrs[group * ladder_offset + row_int + 1];
                    temp_val = 0;
                    for (idx = group_int_start; idx < group_int_stop; idx++)
                    {
                        if (do_col_search)
                        {
                            col_vec = row;
                            flip_bits(col_vec, group_inds->data(), group_inds->size());
                            col_ptr = subspace.get_ptr(col_vec);
                            if (col_ptr == nullptr)
                            {
                                break;
                            } // column is NOT in the subspace so break group
                            do_col_search = 0;
                        }
                        term = &terms[idx];
                        if (passes_proj_validation(term, row))
                        {
                            accum_element(row, col_vec, &term->indices[0], &term->values[0],
                                          term->coeff, term->real_phase, term->indices.size(), temp_val);
                        }
                    } // end loop over this group
                    if (std::abs(temp_val) > ATOL) // if at least one element was found
                    {
                        val += temp_val * in_vec[*col_ptr];
                    }
                } // end loop over all groups
                out_vec[kk] += val;
            } // end for-loop over rows
        } // end if num_terms
    } // end parallel region
} // end matvec
