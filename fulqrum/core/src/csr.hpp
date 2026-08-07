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
#include <complex>
#include <cstdlib>
#include <vector>

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include <boost/dynamic_bitset.hpp>

template <typename T, typename U>
void csr_matrix_builder(const std::vector<OperatorTerm_t>& terms,
                        const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                        const U* __restrict diag_vec,
                        const width_t width,
                        const std::size_t subspace_dim,
                        const int has_nonzero_diag,
                        const std::size_t* __restrict group_ptrs,
                        const std::vector<std::vector<width_t>>& group_offdiag_inds,
                        const std::size_t num_groups,
                        T* __restrict indptr,
                        T* __restrict indices,
                        U* __restrict data,
                        const int compute_values)
{
    std::size_t kk;
    T temp, _sum;

    const auto* bitsets = subspace.get_bitsets();

    // Flatten group_offdiag_inds from a vector-of-vectors (scattered heap
    // allocations with pointer chasing per group visit) into a single
    // contiguous CSR-like buffer.  The indirection cost was measurable
    // in profiles, especially for large num_groups.
    std::vector<width_t> _flat_inds;
    std::vector<std::size_t> _inds_offsets;
    flatten_offdiag_inds(group_offdiag_inds, _flat_inds, _inds_offsets);
    const width_t* __restrict flat_inds = _flat_inds.data();
    const std::size_t* __restrict inds_offsets = _inds_offsets.data();
    auto gview = [&](std::size_t g) -> GroupIndsView {
        const std::size_t off = inds_offsets[g];
        return GroupIndsView{flat_inds + off, inds_offsets[g + 1] - off};
    };

    // grp_max_inds[g] = the highest bit-flip index for group g.
    // Used to detect lower-triangle elements without building col_vec.
    // See get_group_max_inds() in offdiag_grouping.hpp for the rationale.
    std::vector<uint16_t> grp_max_inds(num_groups, width);
    get_group_max_inds(grp_max_inds, group_offdiag_inds, num_groups);

    // Block size for the tiled outer loop.
    std::size_t BLK = 128;
    if(const char* _blk_env = std::getenv("FQ_BLK"))
    {
        long _blk = std::atol(_blk_env);
        if(_blk > 0)
            BLK = static_cast<std::size_t>(_blk);
    }
    const std::size_t rsb_w = width; // one uint8 per qubit
    const std::size_t num_blocks = (subspace_dim + BLK - 1) / BLK;

    // a vector containing the NNZ per row of the matrix
    std::vector<T> row_nnz_s(subspace_dim, 0);
#pragma omp parallel if(subspace_dim > 4096)
    {
        // do diagonal first, if any
        if(has_nonzero_diag)
        {
#pragma omp for
            for(kk = 0; kk < subspace_dim; kk++)
            {
                T& row_nnz = row_nnz_s[kk]; // reference T& is critical
                T& elem_start = indptr[kk];
                if(diag_vec[kk] != 0.0)
                {
                    if(compute_values)
                    {
                        indices[elem_start + row_nnz] = kk;
                        data[elem_start + row_nnz] = diag_vec[kk];
                    }
                    row_nnz += 1;
                }
            }
        }

        // Per-thread scratch buffers, allocated once and reused for all
        // blocks assigned to this thread (avoids per-row heap allocation).
        std::vector<uint8_t> rsb_buf; // row_set_bits for BLK rows
        boost::dynamic_bitset<std::size_t> col_vec(width);
        const std::size_t num_col_blocks = col_vec.num_blocks();

#pragma omp for
        for(std::size_t blk = 0; blk < num_blocks; ++blk)
        {
            const std::size_t r0 = blk * BLK;
            const std::size_t r1 = std::min(r0 + BLK, subspace_dim);
            const std::size_t bn = r1 - r0;

            // Build the bit-vector for every row in the block.
            // Stored contiguously as bn * rsb_w bytes so the
            // group loop can stride over rows cheaply.
            rsb_buf.assign(bn * rsb_w, 0);
            for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
            {
                const boost::dynamic_bitset<std::size_t>& row = bitsets[r0 + row_in_block].first;
                uint8_t* dst = rsb_buf.data() + row_in_block * rsb_w;
                for(std::size_t b = 0; b < row.num_blocks(); ++b)
                {
                    std::size_t bits = row.m_bits[b];
                    while(bits != 0)
                    {
                        int r = __builtin_ctzll(bits);
                        dst[b * BITS_PER_BLOCK + r] = 1;
                        bits &= bits - 1;
                    }
                }
            }

            for(std::size_t group = 0; group < num_groups; group++)
            { // begin loop over groups
                const GroupIndsView group_inds = gview(group);
                const std::size_t group_start = group_ptrs[group];
                const std::size_t group_stop = group_ptrs[group + 1];
                if(group_start >= group_stop)
                    continue;

                const uint16_t max_ind = grp_max_inds[group];

                for(std::size_t row_in_block = 0; row_in_block < bn; ++row_in_block)
                {
                    const uint8_t* row_set_bits = rsb_buf.data() + row_in_block * rsb_w;
                    // Lower-triangle check: skip upper-triangle elements.
                    // See get_group_max_inds() in offdiag_grouping.hpp.
                    if(!row_set_bits[max_ind])
                        continue; // continue onto new row

                    const std::size_t kk = r0 + row_in_block;
                    const boost::dynamic_bitset<std::size_t>& row = bitsets[kk].first;
                    std::memcpy(col_vec.m_bits.data(),
                                row.m_bits.data(),
                                num_col_blocks * sizeof(std::size_t));
                    flip_bits(col_vec, group_inds.data(), group_inds.size());

                    std::size_t* col_ptr = subspace.get_ptr(col_vec);
                    if(col_ptr == nullptr)
                        continue;
                    const std::size_t col_idx = *col_ptr;
                    T row_nnz_col_idx, elem_start_col_idx;
                    T& row_nnz = row_nnz_s[kk];
                    T& elem_start = indptr[kk];
                    U val = 0;
                    for(std::size_t idx = group_start; idx < group_stop; idx++)
                    { // begin loop over terms in this group
                        const OperatorTerm_t* term = &terms[idx];
                        if(passes_proj_validation(term, row))
                        {
                            accum_element(row,
                                          col_vec,
                                          term->indices,
                                          term->values,
                                          term->coeff,
                                          term->real_phase,
                                          term->indices.size(),
                                          val);
                        }
                    } // end loop over terms in this group
                    if(std::abs(val) > ATOL)
                    {
                        if(compute_values)
                        {
#pragma omp atomic write
                            indices[elem_start + row_nnz] = col_idx;
                            if constexpr(std::is_same_v<U, double>) // real case
                            {
#pragma omp atomic write
                                data[elem_start + row_nnz] = val;
                            }
                            else // imaginary case
                            {
                                double* __restrict p =
                                    reinterpret_cast<double*>(&data[elem_start + row_nnz]);
                                const double* __restrict q = reinterpret_cast<const double*>(&val);
#pragma omp atomic
                                p[0] += q[0]; // real part
#pragma omp atomic
                                p[1] += q[1]; // imag part
                            }
#pragma omp atomic
                            row_nnz += 1;

                            row_nnz_col_idx = row_nnz_s[col_idx];
                            elem_start_col_idx = indptr[col_idx];
#pragma omp atomic write
                            indices[elem_start_col_idx + row_nnz_col_idx] = kk;
#pragma omp atomic
                            row_nnz_s[col_idx] += 1;
                            // process col_idx
                            if constexpr(std::is_same_v<U, double>)
                            {
#pragma omp atomic write
                                data[elem_start_col_idx + row_nnz_col_idx] = val;
                            }
                            else
                            {
                                // for complex-valued matrix, the upper triangle
                                // element will be complex conjugate of the lower
                                // triangle element
                                const U update_val = std::conj(val);
                                double* __restrict p2 = reinterpret_cast<double*>(
                                    &data[elem_start_col_idx + row_nnz_col_idx]);
                                const double* __restrict q2 =
                                    reinterpret_cast<const double*>(&update_val);
#pragma omp atomic write
                                p2[0] = q2[0]; // real part
#pragma omp atomic write
                                p2[1] = q2[1]; // imag part
                            }
                        }
                        else
                        {
#pragma omp atomic
                            row_nnz += 1;

#pragma omp atomic
                            row_nnz_s[col_idx] += 1;
                        }
                    }
                } // end loop over row block
            } // end loop over groups
        } // end loop over all row blocks
    } // end omp parallel region

    if(!compute_values) // Done with all rows so accumulate for correct indptr structure
    {
        _sum = 0;
        for(kk = 0; kk < (subspace_dim); kk++)
        {
            temp = _sum + row_nnz_s[kk];
            indptr[kk] = _sum;
            _sum = temp;
        }
        indptr[subspace_dim] = _sum;
    }
}

template <typename T, typename U>
void csr_spmv(const T* __restrict indptr,
              const T* __restrict indices,
              const U* __restrict data,
              const U* __restrict vec,
              U* __restrict out,
              std::size_t dim)
{
    std::size_t row;
#pragma omp parallel for if(dim > 128)
    for(row = 0; row < dim; row++)
    {
        T jj;
        T row_start, row_end;
        U dot = 0.0;
        row_start = indptr[row];
        row_end = indptr[row + 1];

        for(jj = row_start; jj < row_end; jj++)
        {
            dot += data[jj] * vec[indices[jj]];
        }
        out[row] += dot;
    }
}
