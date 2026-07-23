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
#include <cstring>
#include <unordered_map>
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
// type2_visit_non_cartesian: the subspace is an arbitrary (non-Cartesian) subset of the
// beta x alpha grid, so a column index can no longer be computed arithmetically.
//
// We can break down each full-str or full-det into an alpha and a beta half and collect
// them into two separate lists of unique alphas and unique betas. The full Cartesian
// grid will be unique_beta x unique_alpha, and our non-Cartesian subspace is a subset
// of the full grid. We can visualize it as following; Consider unique betas are rows
// and unique alphas are columns of the grid. A point in the grid represent a full-det
// consisting of a (beta, alpha) pair. In the non-Cartesian case, some of these grid points
// exists in the subspace and others are absent.

// Now, Fulqrum uses following steps:
// 1. For a row full-det, find connected column full-dets.
// 2. Once a connection (we also name it partner det) is found, evaluate the matrix element.

// When the subspace is a clean Cartesian grid (every point in the grid exists in the subspace), step # 1 is easier as connected column index
// can be arthimetically computed. However, for non-Cartesian subspaces where there are
// absent grid points, we cannot compute connected columns arithmetically, and that makes
// non-Cartesian path essentially different from the Cartesian path. We follow different
// strategies to find partners here, separated in same-spin/sector (aa/aaaa,bb/bbbb) and
// mixed-spin (aabb) paths.

// ++++++++++++++++++++++++++++++
// Same sector (aa/aaaa,bb/bbbb):
// ++++++++++++++++++++++++++++++

// Only one half det changes and other half remains same for this type of groups.
// Consider a beta row in the above unique_beta x unique_alpha grid. The row has many
// present grid-points, i.e., columns that represent alpha dets. The columns or alpha dets
// that are either 2 or 4 Hamming distance away from each other are only valid partners (in
// other wrdfs may have non-zero Hamiltonian element). One way to discover the connected alpha
// dets is to perform XOR scan and look for Hamming distance 2 or 4. However, there can be
// row betas which have too many (present) alpha columns. For example, an Fe4S4 subspace had a beta row
// with 499K+ alpha columns. Doing the XOR scan will need O(499K^2) computations, making it too slow.
// (A beta row can be thought as a line in the grid (grid line), and there are points on the
// line representing present alpha columns from full-dets in the subspace. One grid line
// has L=499K+ points in the above example).

// Therefore, we adopt a three tier process to tackle this partner discovery overhead.
// 1. Tier-1 (scan_line):
//     a. Slowest, but most generic. No norbs limit. Performs above O(L^2) scans. When L is smaller,
//         works just fine.
// 2. Tier-2 (residue_line):
//     a. Faster than Tier-1. Supports norbs <= 64.
//     b. Groups determinants that can be single (double) excitations away from each other.
//         Consider half-det-len=4 with 2 electrons. Possible half-dets are:
//         0011
//         0101
//         1001
//         0110
//         1010
//         1100

//         Now, single excitation removes one electron, i.e., flips one "1" (and places it into another "0" orbital).
//         By removing a single electron, we get following partial dets, and we name then residues:
//         0011: remove 0th -> 0010 (residue B), remove 1st -> 0001 (residue A)
//         0101: remove 0th -> 0100 (residue C), remove 2nd -> 0001 (residue A)
//         1001: remove 0th -> 1000 (residue D), remove 3rd -> 0001 (residue A)
//         0110: remove 1st -> 0100 (residue C), remove 2nd -> 0010 (residue B)
//         1010: remove 1st -> 1000 (residue D), remove 3rd -> 0010 (residue B)
//         1100: remove 2nd -> 1000 (residue D), remove 3rd -> 0100 (residue C)

//         We bucket the dets by residue:
//         -------------------------------
//         Residue     | Dets
//         -------------------------------
//         A (0001)    | 0011, 0101, 1001
//         B (0010)    | 0011, 0110, 1010
//         C (0100)    | 0101, 0110, 1100
//         D (1000)    | 1001, 1010, 1100
//         -------------------------------

//         The significance of residue buckets is inside each buckets dets are
//         single excitation or Hamming distance 2 away from each other, guaranteed.
//         So, we do XOR scans to find flip positions inside each bucket not over
//         all pairs of dets, which reduces number of scans significantly.

//         Double excitation follow similar pattern as above, but does not have Hamming distance = 4
//         guarantee within a bucket. So, we do one extra check for double excitation buckets.

// 3. Tier-3 (radix_residue_line):
//     a. Fastest. norbs < 64 (+ extra condition ((L-1) >> (64 - norbs) == 0).
//     b. Same as Tier-2 just bucketing is based on radix sort.

// ++++++++++++++++++++
// Cross sector (aabb):
// ++++++++++++++++++++

// Unlike same sector, BOTH halves change together: one alpha electron moves AND one beta
// electron moves, at the same time. So a partner det sits at a different row (beta changed)
// AND a different column (alpha changed) on the . This is why aabb needs a completely different startegy.

// TODO: Add description.
//
// The sink is the ONLY thing that differs between matvec2.hpp, csr2.hpp and
// csrlike_builder2.hpp: a matvec accumulates out[row] += val * in[col], a CSR builder
// appends (col, val) to that row.
//
// sink(row, col, val) is called concurrently, but each row is touched by exactly one
// thread for the duration of the call, so no sink needs locking. That row ownership is
// also what lets the symmetric path write both (i,j) and (j,i) without contention.
template <typename T, typename Sink>
void type2_visit_non_cartesian(const HalfStrTables<T>& tables,
                               const std::vector<OperatorTerm_t>& terms,
                               const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                               const std::size_t subspace_dim,
                               const width_t* __restrict group_rowint_length,
                               const unsigned int ladder_offset,
                               Sink sink)
{
    (void)subspace_dim;
    const auto* bitsets = subspace.get_bitsets();
    const std::size_t kw = tables.kw;
    const std::size_t nbp = tables.num_b_pairs;
    const bool sym = tables.sym;
    const width_t* __restrict flat_inds = tables.flat_inds.data();
    const std::size_t* __restrict inds_offsets = tables.inds_offsets.data();
    const std::uint32_t* __restrict ladder32 = tables.ladder32.data();
    const T* __restrict aabb_val_t = tables.aabb_val_2d_t.data(); // rb-major (see tables)
    const std::size_t nap = tables.num_a_pairs;
    const std::uint64_t* __restrict akeyw = tables.akeyw.data();
    const std::int32_t* __restrict a_pair_rank = tables.a_pair_rank.data();
    const std::size_t hw = tables.half_width;

    auto gview = [&](std::size_t g) -> GroupIndsView {
        const std::size_t off = inds_offsets[g];
        return GroupIndsView{flat_inds + off, inds_offsets[g + 1] - off};
    };

    const std::size_t residue_min = (kw == 1) ? tables.residue_min : 0;
    // Radix residue (fastest) needs the key + line position to pack into one
    // uint64 (hw < 64 and len fits pos_bits = 64 - hw); FQ_FS_NORADIX forces the Tier-2
    // residue_line instead. packable is re-checked per line below.
    const bool no_radix = std::getenv("FQ_FS_NORADIX") != nullptr;
    auto radix_ok = [&](std::size_t len) { return hw < 64 && ((len - 1) >> (64 - hw)) == 0; };

#pragma omp parallel
    {
        std::vector<std::uint8_t> rsb(tables.rsb_w, 0);
        boost::dynamic_bitset<std::size_t> col_vec;
        std::vector<std::int32_t> map;
        // aabb membership bitmap for cheaper pre-check.
        std::vector<std::uint64_t> bmap;
        // Tier-2 residue: a flat
        // CSR of residue-buckets, built by counting sort.
        // the CSR is contiguous (better cache hit rate). resid2id maps a residue to a dense bucket id
        // (value is a plain uint32, NOT a vector). PackedFlipHash is mandatory.
        emhash8::HashMap<std::uint64_t, std::uint32_t, PackedFlipHash> resid2id;
        std::vector<std::uint32_t> csr_off; // per-bucket start offset (then reused as end)
        std::vector<std::uint32_t> csr_pos; // positions, bucket-major
        std::vector<std::uint32_t> csr_cur; // fill cursors
        // Tier-3 (radix residue): LSD radix
        // sort of packed (residue<<pos_bits | pos) uint64 pairs replaces the hash-CSR above.
        // Many times faster on the giant line because it never touches a hash map.
        std::vector<std::uint64_t> rp_a, rp_b;

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
            const GroupIndsView gi = gview(g);
            const unsigned int row_int =
                bitset_ladder_int(rsb.data(), gi.data(), group_rowint_length[g]);
            const std::size_t s = static_cast<std::size_t>(ladder32[g * ladder_offset + row_int]);
            const std::size_t e =
                static_cast<std::size_t>(ladder32[g * ladder_offset + row_int + 1]);
            if(s >= e)
                return T(0);
            col_vec = row;
            flip_bits(col_vec, gi.data(), gi.size());
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

        // Scan one grid line for same-sector partners of the determinants.
        // TODO: Add description of inputs to this lambda.
        auto scan_line =
            [&](const std::uint64_t* __restrict keyw,
                const std::uint32_t* __restrict ids,
                const std::uint32_t* __restrict flats,
                std::size_t len,
                std::size_t beg,
                std::size_t end,
                bool triangle,
                const emhash8::HashMap<std::uint32_t, std::size_t, PackedFlipHash>& smap,
                const emhash8::HashMap<std::uint64_t, std::size_t, PackedFlipHash>& dmap) {
                for(std::size_t p = beg; p < end; p++)
                {
                    const std::size_t i = flats[p];
                    const std::uint64_t* __restrict ki =
                        keyw + static_cast<std::size_t>(ids[p]) * kw;
                    const auto& row = bitsets[i].first;
                    bool rsb_ready = false;
                    for(std::size_t q = triangle ? p + 1 : 0; q < len; q++)
                    {
                        if(q == p)
                            continue;
                        const std::uint64_t* __restrict kq =
                            keyw + static_cast<std::size_t>(ids[q]) * kw;
                        std::uint64_t xw[4];
                        int hd = 0;
                        for(std::size_t w = 0; w < kw; w++)
                        {
                            xw[w] = ki[w] ^ kq[w];
                            hd += __builtin_popcountll(xw[w]);
                        }
                        if(hd != 2 && hd != 4)
                            continue;
                        width_t pos[4];
                        int np = 0;
                        for(std::size_t w = 0; w < kw; w++)
                        {
                            std::uint64_t v = xw[w];
                            while(v != 0)
                            {
                                pos[np++] = static_cast<width_t>(w * 64 + __builtin_ctzll(v));
                                v &= v - 1;
                            }
                        }
                        std::size_t g = NO_HALF_G;
                        if(hd == 2)
                        {
                            auto it = smap.find(pack2(pos[0], pos[1]));
                            if(it != smap.end())
                                g = it->second;
                        }
                        else
                        {
                            auto it = dmap.find(pack4(pos[0], pos[1], pos[2], pos[3]));
                            if(it != dmap.end())
                                g = it->second;
                        }
                        if(g == NO_HALF_G)
                            continue;
                        if(!rsb_ready)
                        {
                            set_rsb(row);
                            rsb_ready = true;
                        }
                        const T val = eval(g, row);
                        if(std::abs(val) <= ATOL)
                            continue;
                        const std::size_t j = flats[q];
                        sink(i, j, val);
                        if(triangle)
                            sink(j, i, val); // H real symmetric; j is on this line, so we own it
                    }
                }
            };

        // Tier 2: residue-hashing
        auto residue_line =
            [&](const std::uint64_t* __restrict keyw,
                const std::uint32_t* __restrict ids,
                const std::uint32_t* __restrict flats,
                std::size_t len,
                const emhash8::HashMap<std::uint32_t, std::size_t, PackedFlipHash>& smap,
                const emhash8::HashMap<std::uint64_t, std::size_t, PackedFlipHash>& dmap) {
                // run cb(residue) for each n-removal residue of the det at line position r
                auto gen = [&](int nrem, std::size_t r, auto&& cb) {
                    const std::uint64_t k = keyw[static_cast<std::size_t>(ids[r])];
                    std::uint64_t v1 = k;
                    while(v1)
                    {
                        const std::uint64_t b1 = v1 & (0ULL - v1); // lowest set bit
                        if(nrem == 1)
                            cb(k ^ b1);
                        else
                        {
                            std::uint64_t v2 = v1 & (v1 - 1);
                            while(v2)
                            {
                                const std::uint64_t b2 = v2 & (0ULL - v2);
                                cb(k ^ b1 ^ b2);
                                v2 &= v2 - 1;
                            }
                        }
                        v1 &= v1 - 1;
                    }
                };
                // build the residue-bucket CSR for one removal order (counting sort)
                auto process = [&](int nrem, bool need_hd4) {
                    resid2id.clear();
                    csr_off.clear();
                    std::uint32_t nb = 0;
                    for(std::size_t r = 0; r < len; r++)
                        gen(nrem, r, [&](std::uint64_t resid) {
                            auto it = resid2id.find(resid);
                            if(it == resid2id.end())
                            {
                                resid2id[resid] = nb++;
                                csr_off.push_back(1);
                            }
                            else
                                csr_off[it->second]++;
                        });
                    if(nb == 0)
                        return;
                    std::uint32_t total = 0; // counts -> exclusive prefix offsets
                    for(std::uint32_t b = 0; b < nb; b++)
                    {
                        const std::uint32_t c = csr_off[b];
                        csr_off[b] = total;
                        total += c;
                    }
                    csr_cur.assign(csr_off.begin(), csr_off.end()); // fill cursors = starts
                    csr_pos.resize(total);
                    for(std::size_t r = 0; r < len; r++)
                        gen(nrem, r, [&](std::uint64_t resid) {
                            csr_pos[csr_cur[resid2id[resid]]++] = static_cast<std::uint32_t>(r);
                        });

                    for(std::uint32_t b = 0; b < nb; b++)
                    {
                        const std::uint32_t bs = csr_off[b], be = csr_cur[b];
                        for(std::uint32_t a = bs; a < be; a++)
                        {
                            const std::uint32_t pa = csr_pos[a];
                            const std::size_t ia = flats[pa];
                            const std::uint64_t ka = keyw[static_cast<std::size_t>(ids[pa])];
                            const auto& row = bitsets[ia].first;
                            bool rsb_ready = false;
                            for(std::uint32_t c = a + 1; c < be; c++)
                            {
                                const std::uint32_t pb = csr_pos[c];
                                const std::uint64_t x =
                                    ka ^ keyw[static_cast<std::size_t>(ids[pb])];
                                const int hd = __builtin_popcountll(x);
                                if(need_hd4 && hd != 4)
                                    continue;
                                width_t pos[4];
                                int np = 0;
                                std::uint64_t v = x;
                                while(v)
                                {
                                    pos[np++] = static_cast<width_t>(__builtin_ctzll(v));
                                    v &= v - 1;
                                }
                                std::size_t g = NO_HALF_G;
                                if(hd == 2)
                                {
                                    auto it = smap.find(pack2(pos[0], pos[1]));
                                    if(it != smap.end())
                                        g = it->second;
                                }
                                else
                                {
                                    auto it = dmap.find(pack4(pos[0], pos[1], pos[2], pos[3]));
                                    if(it != dmap.end())
                                        g = it->second;
                                }
                                if(g == NO_HALF_G)
                                    continue;
                                if(!rsb_ready)
                                {
                                    set_rsb(row);
                                    rsb_ready = true;
                                }
                                const T val = eval(g, row);
                                if(std::abs(val) <= ATOL)
                                    continue;
                                const std::size_t j = flats[pb];
                                sink(ia, j, val);
                                sink(j, ia, val);
                            }
                        }
                    }
                };
                process(1, false); // 1-removal buckets -> hd2 (aa/bb)
                process(2, true); // 2-removal buckets -> hd4 (aaaa/bbbb), filtered
            };

        // Teir 3: Radix residue
        auto radix_residue_line =
            [&](const std::uint64_t* __restrict keyw,
                const std::uint32_t* __restrict ids,
                const std::uint32_t* __restrict flats,
                std::size_t len,
                const emhash8::HashMap<std::uint32_t, std::size_t, PackedFlipHash>& smap,
                const emhash8::HashMap<std::uint64_t, std::size_t, PackedFlipHash>& dmap) {
                const unsigned pos_bits = 64u - static_cast<unsigned>(hw);
                const std::uint64_t pos_mask = (1ULL << pos_bits) - 1;
                // set-bit count is constant across a same-sector line (fixed electron count),
                // so the 2-removal pass emits exactly C(nset,2) residues/det. Size to that.
                const int nset = __builtin_popcountll(keyw[static_cast<std::size_t>(ids[0])]);
                const std::size_t per2 = static_cast<std::size_t>(nset) * (nset - 1) / 2;
                if(rp_a.size() < len * per2)
                {
                    rp_a.resize(len * per2);
                    rp_b.resize(len * per2);
                }

                auto process = [&](int nrem, bool need_hd4) {
                    // generate packed (residue<<pos_bits | pos) into rp_a
                    std::size_t N = 0;
                    for(std::size_t r = 0; r < len; r++)
                    {
                        const std::uint64_t k = keyw[static_cast<std::size_t>(ids[r])];
                        std::uint64_t v1 = k;
                        while(v1)
                        {
                            const std::uint64_t b1 = v1 & (0ULL - v1);
                            if(nrem == 1)
                                rp_a[N++] = ((k ^ b1) << pos_bits) | r;
                            else
                            {
                                std::uint64_t v2 = v1 & (v1 - 1);
                                while(v2)
                                {
                                    const std::uint64_t b2 = v2 & (0ULL - v2);
                                    rp_a[N++] = ((k ^ b1 ^ b2) << pos_bits) | r;
                                    v2 &= v2 - 1;
                                }
                            }
                            v1 &= v1 - 1;
                        }
                    }
                    if(N == 0)
                        return;
                    // LSD radix sort by the residue field (bits pos_bits..64)
                    std::uint64_t* src = rp_a.data();
                    std::uint64_t* dst = rp_b.data();
                    for(unsigned shift = pos_bits; shift < 64; shift += 8)
                    {
                        std::uint32_t cnt[256];
                        std::memset(cnt, 0, sizeof(cnt));
                        for(std::size_t i = 0; i < N; i++)
                            cnt[(src[i] >> shift) & 0xFF]++;
                        std::uint32_t sum = 0;
                        for(int d = 0; d < 256; d++)
                        {
                            const std::uint32_t c = cnt[d];
                            cnt[d] = sum;
                            sum += c;
                        }
                        for(std::size_t i = 0; i < N; i++)
                            dst[cnt[(src[i] >> shift) & 0xFF]++] = src[i];
                        std::uint64_t* tmp = src;
                        src = dst;
                        dst = tmp;
                    }
                    // enumerate each residue-bucket (run of equal residue)
                    std::size_t a = 0;
                    while(a < N)
                    {
                        const std::uint64_t res = src[a] >> pos_bits;
                        std::size_t be = a + 1;
                        while(be < N && (src[be] >> pos_bits) == res)
                            be++;
                        for(std::size_t x = a; x < be; x++)
                        {
                            const std::uint32_t pa = static_cast<std::uint32_t>(src[x] & pos_mask);
                            const std::size_t ia = flats[pa];
                            const std::uint64_t ka = keyw[static_cast<std::size_t>(ids[pa])];
                            const auto& row = bitsets[ia].first;
                            bool rsb_ready = false;
                            for(std::size_t y = x + 1; y < be; y++)
                            {
                                const std::uint32_t pb =
                                    static_cast<std::uint32_t>(src[y] & pos_mask);
                                const std::uint64_t xr =
                                    ka ^ keyw[static_cast<std::size_t>(ids[pb])];
                                const int hd = __builtin_popcountll(xr);
                                if(need_hd4 && hd != 4)
                                    continue;
                                width_t pos[4];
                                int np = 0;
                                std::uint64_t v = xr;
                                while(v)
                                {
                                    pos[np++] = static_cast<width_t>(__builtin_ctzll(v));
                                    v &= v - 1;
                                }
                                std::size_t g = NO_HALF_G;
                                if(hd == 2)
                                {
                                    auto it = smap.find(pack2(pos[0], pos[1]));
                                    if(it != smap.end())
                                        g = it->second;
                                }
                                else
                                {
                                    auto it = dmap.find(pack4(pos[0], pos[1], pos[2], pos[3]));
                                    if(it != dmap.end())
                                        g = it->second;
                                }
                                if(g == NO_HALF_G)
                                    continue;
                                if(!rsb_ready)
                                {
                                    set_rsb(row);
                                    rsb_ready = true;
                                }
                                const T val = eval(g, row);
                                if(std::abs(val) <= ATOL)
                                    continue;
                                const std::size_t j = flats[pb];
                                sink(ia, j, val);
                                sink(j, ia, val);
                            }
                        }
                        a = be;
                    }
                };
                process(1, false); // 1-removal buckets -> hd2 (aa/bb)
                process(2, true); // 2-removal buckets -> hd4 (aaaa/bbbb), filtered
            };

        // pass A: alpha same-sector + aabb, over beta rows
#pragma omp for schedule(dynamic, 1)
        for(std::size_t t = 0; t < tables.task_a.size(); t++)
        {
            const FsTask tk = tables.task_a[t];
            const std::size_t r0 = tables.row_start[tk.grp];
            const std::size_t len = tables.row_start[tk.grp + 1] - r0;
            const bool whole = (tk.beg == 0 && tk.end == len);

            if(residue_min && whole && sym && len > residue_min)
            {
                if(!no_radix && radix_ok(len))
                    radix_residue_line(tables.akeyw.data(),
                                       tables.row_alpha.data() + r0,
                                       tables.row_flat.data() + r0,
                                       len,
                                       tables.aa_map,
                                       tables.aaaa_map);
                else
                    residue_line(tables.akeyw.data(),
                                 tables.row_alpha.data() + r0,
                                 tables.row_flat.data() + r0,
                                 len,
                                 tables.aa_map,
                                 tables.aaaa_map);
            }
            else
                scan_line(tables.akeyw.data(),
                          tables.row_alpha.data() + r0,
                          tables.row_flat.data() + r0,
                          len,
                          tk.beg,
                          tk.end,
                          sym && whole,
                          tables.aa_map,
                          tables.aaaa_map);

            const std::size_t cb0 = tables.b_aabb_off[tk.grp];
            const std::size_t cb1 = tables.b_aabb_off[tk.grp + 1];
            if(cb0 == cb1 || nbp == 0)
                continue;
            for(std::size_t ci = cb0; ci < cb1; ci++)
            {
                const AabbConn cb = tables.b_aabb[ci];
                const std::size_t t0 = tables.row_start[cb.col];
                const std::size_t t1 = tables.row_start[cb.col + 1];
                const std::size_t tlen = t1 - t0;
                const std::size_t rb = static_cast<std::size_t>(cb.rank);
                const int sb = cb.sign;
                // this target row's coeffs, contiguous (rb-major table): vrow[ra].
                const T* __restrict vrow = aabb_val_t + rb * nap;

                // Two ways to find the alpha partners present in target row b':
                //  (a) SCAN it: XOR+popcounts.
                //  (b) PROBE it: lookups into a dense alpha-id -> flat map.
                // (a) wins whenever the target row is shorter than the source's single list,
                // which is the common case on a sparse grid.
                if(tlen <= tables.scan_max)
                {
                    for(std::size_t p = tk.beg; p < tk.end; p++)
                    {
                        const std::size_t i = tables.row_flat[r0 + p];
                        const std::uint64_t* __restrict ka =
                            akeyw + static_cast<std::size_t>(tables.row_alpha[r0 + p]) * kw;
                        for(std::size_t q = t0; q < t1; q++)
                        {
                            const std::uint64_t* __restrict kq =
                                akeyw + static_cast<std::size_t>(tables.row_alpha[q]) * kw;
                            std::uint64_t xw[4];
                            int hd = 0;
                            for(std::size_t w = 0; w < kw; w++)
                            {
                                xw[w] = ka[w] ^ kq[w];
                                hd += __builtin_popcountll(xw[w]);
                            }
                            if(hd != 2)
                                continue;
                            width_t pp[2];
                            int np = 0;
                            for(std::size_t w = 0; w < kw && np < 2; w++)
                            {
                                std::uint64_t v = xw[w];
                                while(v != 0)
                                {
                                    pp[np++] = static_cast<width_t>(w * 64 + __builtin_ctzll(v));
                                    v &= v - 1;
                                }
                            }
                            const std::int32_t ra =
                                a_pair_rank[static_cast<std::size_t>(pp[0]) * hw + pp[1]];
                            if(ra < 0)
                                continue;
                            const T coeff = vrow[static_cast<std::size_t>(ra)];
                            if(std::abs(coeff) <= ATOL)
                                continue;
                            const int sa = words_range_parity(ka, kw, pp[0] + 1u, pp[1]) ? -1 : 1;
                            sink(i,
                                 static_cast<std::size_t>(tables.row_flat[q]),
                                 coeff * static_cast<T>(sa * sb));
                        }
                    }
                    continue;
                }

                if(map.empty())
                {
                    map.assign(tables.num_alpha, -1);
                    bmap.assign((tables.num_alpha + 63) / 64, 0);
                }
                for(std::size_t q = t0; q < t1; q++)
                {
                    const std::uint32_t a = tables.row_alpha[q];
                    map[a] = static_cast<std::int32_t>(tables.row_flat[q]);
                    bmap[a >> 6] |= 1ULL << (a & 63);
                }
                for(std::size_t p = tk.beg; p < tk.end; p++)
                {
                    const std::size_t i = tables.row_flat[r0 + p];
                    const std::size_t aid = tables.row_alpha[r0 + p];
                    const std::size_t a0 = tables.a_aabb_off[aid];
                    const std::size_t a1 = tables.a_aabb_off[aid + 1];
                    for(std::size_t ai = a0; ai < a1; ai++)
                    {
                        const AabbConn ca = tables.a_aabb[ai];
                        if(!(bmap[ca.col >> 6] >> (ca.col & 63) & 1))
                            continue;
                        const std::int32_t m = map[ca.col];
                        const T coeff = vrow[static_cast<std::size_t>(ca.rank)];
                        if(std::abs(coeff) <= ATOL)
                            continue;
                        sink(i, static_cast<std::size_t>(m), coeff * static_cast<T>(ca.sign * sb));
                    }
                }
                for(std::size_t q = t0; q < t1; q++)
                {
                    const std::uint32_t a = tables.row_alpha[q];
                    map[a] = -1;
                    bmap[a >> 6] &= ~(1ULL << (a & 63));
                }
            }
        }

        // pass B: beta same-sector, over alpha columns
#pragma omp for schedule(dynamic, 1)
        for(std::size_t t = 0; t < tables.task_b.size(); t++)
        {
            const FsTask tk = tables.task_b[t];
            const std::size_t c0 = tables.col_start[tk.grp];
            const std::size_t len = tables.col_start[tk.grp + 1] - c0;
            const bool whole = (tk.beg == 0 && tk.end == len);
            if(residue_min && whole && sym && len > residue_min)
            {
                if(!no_radix && radix_ok(len))
                    radix_residue_line(tables.bkeyw.data(),
                                       tables.col_beta.data() + c0,
                                       tables.col_flat.data() + c0,
                                       len,
                                       tables.bb_map,
                                       tables.bbbb_map);
                else
                    residue_line(tables.bkeyw.data(),
                                 tables.col_beta.data() + c0,
                                 tables.col_flat.data() + c0,
                                 len,
                                 tables.bb_map,
                                 tables.bbbb_map);
            }
            else
                scan_line(tables.bkeyw.data(),
                          tables.col_beta.data() + c0,
                          tables.col_flat.data() + c0,
                          len,
                          tk.beg,
                          tk.end,
                          sym && whole,
                          tables.bb_map,
                          tables.bbbb_map);
        }
    }
}
