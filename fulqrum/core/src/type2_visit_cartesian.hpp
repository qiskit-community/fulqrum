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
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "external/hash_table8.hpp"

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "elements.hpp"
#include "halfstr_tables.hpp"
#include "offdiag_grouping.hpp"
#include "type2_common.hpp"
#include <boost/dynamic_bitset.hpp>
#ifdef _OPENMP
#    include <omp.h>
#endif

// ============================================================================
// type2_visit_cartesian -- the fast visitor for a CARTESIAN subspace (tables.cartesian).
//
// The determinant set is a clean beta x alpha product, so a column index is arithmetic
// and every same-sector partner is precomputed in the tables' connection lists.
//
// NOTE: this path is selected by detection, not by how the Subspace was built.
// A full-strs-mode subspace that happens to be a clean product lands here too.
// For the non-Cartesian case see type2_visit_non_cartesian in type2_visit_non_cartesian.hpp.
//
// The sink is the ONLY thing that differs between matvec2.hpp, csr2.hpp and
// csrlike_builder2.hpp: a matvec accumulates out[row] += val * in[col], a CSR builder
// appends (col, val) to that row.
//
// sink(row, col, val) is called parallely, but each row is touched by exactly one
// thread for the duration of the call, so no write needs locking.
// ============================================================================
template <typename T, typename Sink>
void type2_visit_cartesian(const HalfStrTables<T>& tables,
                           const std::vector<OperatorTerm_t>& terms,
                           const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                           const std::size_t subspace_dim,
                           const width_t* __restrict group_rowint_length,
                           const unsigned int ladder_offset,
                           Sink sink)
{
    const auto* bitsets = subspace.get_bitsets();
    const std::size_t num_alpha = tables.num_alpha;
    const std::size_t num_b_pairs = tables.num_b_pairs;
    const std::size_t rsb_w = tables.rsb_w;
    const width_t* __restrict flat_inds = tables.flat_inds.data();
    const std::size_t* __restrict inds_offsets = tables.inds_offsets.data();
    const std::uint32_t* __restrict ladder32 = tables.ladder32.data();
    const T* __restrict aabb_val_2d = tables.aabb_val_2d.data();

    auto gview = [&](std::size_t g) -> GroupIndsView {
        const std::size_t off = inds_offsets[g];
        return GroupIndsView{flat_inds + off, inds_offsets[g + 1] - off};
    };

    const std::size_t num_beta = tables.num_beta;
    const bool sym = tables.sym;

#pragma omp parallel if(subspace_dim > 4096)
    {
        std::vector<uint8_t> rsb(rsb_w, 0);
        boost::dynamic_bitset<std::size_t> col_vec;

        auto set_rsb = [&](const boost::dynamic_bitset<std::size_t>& row) {
            std::fill(rsb.begin(), rsb.end(), 0);
            for(std::size_t b = 0; b < row.num_blocks(); b++)
            {
                std::size_t bits = row.m_bits[b];
                while(bits != 0)
                {
                    const int r = __builtin_ctzll(bits);
                    rsb[b * BITS_PER_BLOCK + r] = 1;
                    bits &= bits - 1;
                }
            }
        };

        auto eval = [&](std::size_t g, const boost::dynamic_bitset<std::size_t>& row) -> T {
            const GroupIndsView group_inds = gview(g);
            const unsigned int row_int =
                bitset_ladder_int(rsb.data(), group_inds.data(), group_rowint_length[g]);
            const std::size_t s = static_cast<std::size_t>(ladder32[g * ladder_offset + row_int]);
            const std::size_t e =
                static_cast<std::size_t>(ladder32[g * ladder_offset + row_int + 1]);
            if(s >= e)
                return T(0);
            col_vec = row;
            flip_bits(col_vec, group_inds.data(), group_inds.size());
            T val = 0;
            for(std::size_t idx = s; idx < e; idx++)
            {
                const OperatorTerm_t* term = &terms[idx];
                if(passes_proj_validation(term, row))
                    accum_element(row,
                                  col_vec,
                                  term->indices,
                                  term->values,
                                  term->coeff,
                                  term->real_phase,
                                  term->indices.size(),
                                  val);
            }
            return val;
        };

        // One same-sector list (aa/aaaa on a beta row, bb/bbbb on an alpha column).
        // `tri` => this task owns the whole line, so only partners past `self` are
        // evaluated and each unordered pair costs ONE accum_element; both writes land on
        // this line, which the task owns exclusively.
        auto walk_list = [&](const auto& conns,
                             const boost::dynamic_bitset<std::size_t>& row,
                             std::size_t self,
                             std::size_t idx,
                             std::size_t base,
                             std::size_t stride,
                             bool tri) {
            for(const auto& c : conns)
            {
                if(c.g == NO_HALF_G)
                    continue;
                if(tri && c.col < self)
                    continue;
                const T val = eval(c.g, row);
                if(std::abs(val) <= ATOL)
                    continue;
                const std::size_t j = base + c.col * stride;
                sink(idx, j, val);
                if(tri)
                    sink(j, idx, val); // H real symmetric; j is on this line
            }
        };

        // pass A: alpha same-sector + aabb, over beta rows
#pragma omp for schedule(dynamic, 1)
        for(std::size_t t = 0; t < tables.task_a.size(); t++)
        {
            const FsTask tk = tables.task_a[t];
            const std::size_t b = tk.grp;
            const std::size_t beta_base = b * num_alpha;
            const bool tri = sym && (tk.beg == 0 && tk.end == num_alpha);
            const auto& b_singles = tables.beta_single[b];

            for(std::size_t a = tk.beg; a < tk.end; a++)
            {
                const std::size_t idx = beta_base + a;
                const auto& row = bitsets[idx].first;
                set_rsb(row);
                const auto& a_singles = tables.alpha_single[a];
                const auto& a_doubles = tables.alpha_double[a];

                walk_list(a_singles, row, a, idx, beta_base, 1, tri); // aa
                walk_list(a_doubles, row, a, idx, beta_base, 1, tri); // aaaa

                // aabb: alpha-single x beta-single. The partner sits on a different row
                // AND column, so it is not this task's to write -- gather-only, never
                // symmetrized (it is emitted again from the partner's own row).
                for(const auto& ca : a_singles)
                {
                    if(ca.pair_rank < 0)
                        continue;
                    const std::size_t bse = static_cast<std::size_t>(ca.pair_rank) * num_b_pairs;
                    const int sa = ca.sign;
                    const std::size_t col_a = ca.col;
                    for(const auto& cb : b_singles)
                    {
                        if(cb.pair_rank < 0)
                            continue;
                        const T coeff = aabb_val_2d[bse + static_cast<std::size_t>(cb.pair_rank)];
                        if(std::abs(coeff) <= ATOL)
                            continue;
                        sink(idx, cb.col * num_alpha + col_a, coeff * static_cast<T>(sa * cb.sign));
                    }
                }
            }
        }

        // pass B: beta same-sector, over alpha columns
        // schedule chunk 8 so a thread takes 8 consecutive columns: out[] entries for
        // adjacent columns share a cache line, so handing them to different threads
        // would false-share on every write.
#pragma omp for schedule(dynamic, 8)
        for(std::size_t t = 0; t < tables.task_b.size(); t++)
        {
            const FsTask tk = tables.task_b[t];
            const std::size_t a = tk.grp;
            const bool tri = sym && (tk.beg == 0 && tk.end == num_beta);

            for(std::size_t b = tk.beg; b < tk.end; b++)
            {
                const std::size_t idx = b * num_alpha + a;
                const auto& row = bitsets[idx].first;
                set_rsb(row);
                const auto& b_singles = tables.beta_single[b];
                const auto& b_doubles = tables.beta_double[b];

                walk_list(b_singles, row, b, idx, a, num_alpha, tri); // bb
                walk_list(b_doubles, row, b, idx, a, num_alpha, tri); // bbbb
            }
        }
    }
}
