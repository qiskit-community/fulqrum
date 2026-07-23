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
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "external/hash_table8.hpp"

#include "base.hpp"
#include "bitset_hashmap.hpp"
#include "bitset_utils.hpp"
#include "constants.hpp"
#include "elements.hpp"
#include "offdiag_grouping.hpp"
#include "type2_common.hpp"
#include <boost/dynamic_bitset.hpp>
#ifdef _OPENMP
#    include <omp.h>
#endif

// ********************************************************************************
// HalfStrTables: everything the fast type-2 visitors need, precomputed.
//
// Every determinant is factorized into its alpha and beta half strings; these tables
// are what that factorization buys -- half keys, per-string connection lists, the grid
// CSR and its transpose, flip-position -> group maps, the aabb value table, and the
// parallel task lists. That is why the name is HalfStr and not Cartesian: the tables are
// built the same way for both catersian and non-cartesian paths.
//
// Built once per (operator, subspace) and then reused by every matvec / CSR build (see
// _ensure_hs_tables in spmv.pyx), so the cost of building them is amortized over many
// cheap applies.
//
// Consumed by type2_visit_cartesian (tables.cartesian == true) and
// type2_visit_non_cartesian (tables.cartesian == false). Note that `cartesian` is
// 'detected' from the determinant set, not taken from the Subspace's half-strs/full-strs
// mode. A full-strs subspace that happens to be a clean beta x alpha product is
// Cartesian and takes the Cartesian visitor.
//
// If the tables cannot be built, usable stays false and the caller falls back to
// type2_visit_groups.
// ********************************************************************************

// One single-excitation connection within a spin sector.
struct HalfConnSingle
{
    std::size_t col; // connected alpha (or beta) string index
    std::size_t g; // group for the same-sector (aa/bb) value; NO_HALF_G if none
    int pair_rank; // -1 if pair not in any aabb group
    int sign; // factorized aabb sign (+/-1) for this string + pair
};

// One double-excitation connection within a spin sector.
struct HalfConnDouble
{
    std::size_t col; // connected string index
    std::size_t g; // group for the same-sector (aaaa/bbbb) value
};

// Packed single-excitation connection for the non-Cartesian aabb
struct AabbConn
{
    std::uint32_t col; // connected string index
    std::int16_t rank; // aabb pair rank (index into aabb_val_2d)
    std::int16_t sign; // factorized aabb sign (+/-1)
};
static_assert(sizeof(AabbConn) == 8, "AabbConn must stay 8 bytes (aabb is bandwidth-bound)");

// Parity of set bits in [lo, hi) of a raw half key. Mirrors hk_range_parity, but the
// full-strs enumerator carries keys as plain 64-bit words rather than HalfKey<K>.
inline int words_range_parity(const std::uint64_t* __restrict k,
                              std::size_t kw,
                              unsigned lo,
                              unsigned hi) noexcept
{
    if(lo >= hi)
        return 0;
    int acc = 0;
    for(std::size_t w = 0; w < kw; w++)
    {
        const unsigned wlo = static_cast<unsigned>(w * 64);
        const unsigned whi = wlo + 64;
        if(whi <= lo || wlo >= hi)
            continue;
        std::uint64_t v = k[w];
        if(lo > wlo)
            v &= ~std::uint64_t(0) << (lo - wlo);
        if(hi < whi)
            v &= (hi - wlo) >= 64 ? ~std::uint64_t(0) : ((std::uint64_t(1) << (hi - wlo)) - 1);
        acc += __builtin_popcountll(v);
    }
    return acc & 1;
}

// One unit of work in type2_visit_non_cartesian: determinants [beg, end) of grid line
// `grp` (a beta row in pass A, an alpha column in pass B). Lines partition the
// determinants, so a task owns every determinant it writes -> lock-free.
struct FsTask
{
    std::uint32_t grp;
    std::uint32_t beg;
    std::uint32_t end;
};

template <typename T>
struct HalfStrTables
{
    bool usable = false;
    width_t width = 0;
    width_t half_width = 0;
    std::size_t num_alpha = 0;
    std::size_t num_beta = 0;
    std::size_t num_a_pairs = 0;
    std::size_t num_b_pairs = 0;
    std::size_t BLK = 128;
    std::size_t rsb_w = 0;
    std::size_t num_blocks = 0;
    // flattened offdiag inds + ladder (for same-sector value eval)
    std::vector<width_t> flat_inds;
    std::vector<std::size_t> inds_offsets;
    std::vector<std::uint32_t> ladder32;
    // per-string excitation lists
    std::vector<std::vector<HalfConnSingle>> alpha_single;
    std::vector<std::vector<HalfConnDouble>> alpha_double;
    std::vector<std::vector<HalfConnSingle>> beta_single;
    std::vector<std::vector<HalfConnDouble>> beta_double;
    // aabb coupling: aabb_val_2d[ra * num_b_pairs + rb] = direct coeff (0 if none)
    std::vector<T> aabb_val_2d;
    // rb-major transpose, built for the NON-Cartesian path only (+3.2 MB at fe4s4).
    // profiling shows ra-major is not cache-friendly for aabb. T
    // So, we are buying cache-friendliness at the cost of extra data.
    std::vector<T> aabb_val_2d_t;

    // non-Cartesian subspace
    // cartesian == true  -> the subspace is a clean beta x alpha product and the
    //   flat column index is arithmetic; type2_visit_cartesian (excitation-list driven) runs.
    // cartesian == false -> the subspace is a sparse subset of the grid; type2_visit_non_cartesian
    //   runs instead.
    bool cartesian = true;
    bool sym = true; // one accum_element per unordered same-sector pair
    std::size_t kw = 0; // 64-bit words per half key
    std::size_t chunk = 4096; // determinants per task on long grid lines
    std::vector<std::uint64_t> akeyw; // num_alpha * kw
    std::vector<std::uint64_t> bkeyw; // num_beta  * kw
    // grid CSR (beta-major) and its transpose (alpha-major)
    std::vector<std::size_t> row_start; // num_beta + 1
    std::vector<std::uint32_t> row_alpha; // subspace_dim
    std::vector<std::uint32_t> row_flat; // subspace_dim
    std::vector<std::size_t> col_start; // num_alpha + 1
    std::vector<std::uint32_t> col_beta; // subspace_dim
    std::vector<std::uint32_t> col_flat; // subspace_dim
    std::vector<FsTask> task_a; // pass A: beta rows
    std::vector<FsTask> task_b; // pass B: alpha columns
    std::size_t scan_max = 64; // target rows this short are scanned, not probed
    std::size_t residue_min = 4096; // same-sector lines longer than this use residue
        // hashing (item 2) instead of the O(len^2) scan;
        // 0 = off (overridden by the builder; see FQ_FS_RESIDUE_MIN)
    std::vector<std::int32_t> a_pair_rank; // half_width^2, -1 = pair not in any aabb group
    // flip-position -> group, for recovering g at a scan hit
    emhash8::HashMap<std::uint32_t, std::size_t, PackedFlipHash> aa_map, bb_map;
    emhash8::HashMap<std::uint64_t, std::size_t, PackedFlipHash> aaaa_map, bbbb_map;
    // Packed aabb connection CSRs (non-Cartesian only; see AabbConn). Replace
    // alpha_single/beta_single in the visitor's aabb loops; the fat lists are freed
    // after packing (-1.4 GB at 42M on top of the 3x bandwidth cut).
    std::vector<std::uint64_t> a_aabb_off; // num_alpha + 1
    std::vector<AabbConn> a_aabb;
    std::vector<std::uint64_t> b_aabb_off; // num_beta + 1
    std::vector<AabbConn> b_aabb;
};

template <std::size_t K>
struct HalfKey
{
    std::array<std::uint64_t, K> w{};
    bool operator==(const HalfKey& o) const noexcept
    {
        return w == o.w;
    }
    bool operator!=(const HalfKey& o) const noexcept
    {
        return !(w == o.w);
    }
};

template <std::size_t K>
struct HalfKeyHash
{
    std::size_t operator()(const HalfKey<K>& k) const noexcept
    {
        return static_cast<std::size_t>(rapidhashMicro(k.w.data(), K * sizeof(std::uint64_t)));
    }
};

template <std::size_t K>
inline HalfKey<K>
extract_half(const boost::dynamic_bitset<std::size_t>& bs, width_t lo, width_t len)
{
    HalfKey<K> out;
    const std::size_t nb = bs.num_blocks();
    const std::size_t word_shift = static_cast<std::size_t>(lo) >> 6;
    const unsigned bit_shift = static_cast<unsigned>(lo) & 63u;
    for(std::size_t wi = 0; wi < K; wi++)
    {
        std::uint64_t v = (word_shift + wi < nb) ? (bs.m_bits[word_shift + wi] >> bit_shift) : 0;
        if(bit_shift && (word_shift + wi + 1 < nb))
            v |= bs.m_bits[word_shift + wi + 1] << (64u - bit_shift);
        out.w[wi] = v;
    }
    for(std::size_t wi = 0; wi < K; wi++)
    {
        const std::size_t word_lo = wi * 64;
        if(word_lo >= len)
            out.w[wi] = 0;
        else if(static_cast<std::size_t>(len) - word_lo < 64)
            out.w[wi] &= (std::uint64_t(1) << (static_cast<std::size_t>(len) - word_lo)) - 1;
    }
    return out;
}

template <std::size_t K>
inline void hk_flip(HalfKey<K>& k, width_t p) noexcept
{
    k.w[static_cast<std::size_t>(p) >> 6] ^= std::uint64_t(1) << (static_cast<unsigned>(p) & 63u);
}

template <std::size_t K>
inline bool hk_test(const HalfKey<K>& k, width_t p) noexcept
{
    return (k.w[static_cast<std::size_t>(p) >> 6] >> (static_cast<unsigned>(p) & 63u)) & 1ULL;
}

// Parity of set bits in [lo, hi). Used for the factorized aabb sign.
template <std::size_t K>
inline int hk_range_parity(const HalfKey<K>& k, width_t lo, width_t hi) noexcept
{
    if(lo >= hi)
        return 0;
    const width_t last = static_cast<width_t>(hi - 1);
    const std::uint64_t lo_mask = ~std::uint64_t(0) << (static_cast<unsigned>(lo) & 63u);
    const std::uint64_t hi_mask = ~std::uint64_t(0) >> (63u - (static_cast<unsigned>(last) & 63u));
    if constexpr(K == 1)
    {
        return __builtin_popcountll(k.w[0] & lo_mask & hi_mask) & 1;
    }
    else
    {
        const std::size_t lo_w = static_cast<std::size_t>(lo) >> 6;
        const std::size_t hi_w = static_cast<std::size_t>(last) >> 6;
        int acc;
        if(lo_w == hi_w)
            acc = __builtin_popcountll(k.w[lo_w] & lo_mask & hi_mask);
        else
        {
            acc = __builtin_popcountll(k.w[lo_w] & lo_mask);
            for(std::size_t w = lo_w + 1; w < hi_w; w++)
                acc += __builtin_popcountll(k.w[w]);
            acc += __builtin_popcountll(k.w[hi_w] & hi_mask);
        }
        return acc & 1;
    }
}

template <std::size_t K>
struct FixedKeyOps
{
    using Key = HalfKey<K>;
    template <typename V>
    using Map = emhash8::HashMap<Key, V, HalfKeyHash<K>>;
    static constexpr std::size_t words = K;
    static Key extract(const boost::dynamic_bitset<std::size_t>& bs, width_t lo, width_t len)
    {
        return extract_half<K>(bs, lo, len);
    }
    static void flip(Key& k, width_t p) noexcept
    {
        hk_flip<K>(k, p);
    }
    static bool test(const Key& k, width_t p) noexcept
    {
        return hk_test<K>(k, p);
    }
    static int range_parity(const Key& k, width_t lo, width_t hi) noexcept
    {
        return hk_range_parity<K>(k, lo, hi);
    }
};

struct DynKeyOps
{
    using Key = boost::dynamic_bitset<std::size_t>;
    template <typename V>
    using Map = std::unordered_map<Key, V, BitsetHasherRapid>; // bitset_hashmap.hpp
    static constexpr std::size_t words = 0; // 0 => reported as "dyn" in verbose
    static Key extract(const boost::dynamic_bitset<std::size_t>& bs, width_t lo, width_t len)
    {
        Key out(len);
        for(width_t i = 0; i < len; i++)
            if(bs.test(static_cast<std::size_t>(lo) + i))
                out.set(static_cast<std::size_t>(i));
        return out;
    }
    static void flip(Key& k, width_t p) noexcept
    {
        k.flip(static_cast<std::size_t>(p));
    }
    static bool test(const Key& k, width_t p) noexcept
    {
        return k.test(static_cast<std::size_t>(p));
    }
    static int range_parity(const Key& k, width_t lo, width_t hi) noexcept
    {
        int acc = 0;
        for(width_t p = lo; p < hi; p++)
            acc ^= static_cast<int>(k.test(static_cast<std::size_t>(p)));
        return acc;
    }
};

template <typename T, typename Policy>
void build_halfstr_tables_impl(const std::vector<OperatorTerm_t>& terms,
                               const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                               const width_t width,
                               const std::size_t subspace_dim,
                               const std::size_t* __restrict group_ptrs,
                               const std::size_t* __restrict group_ladder_ptrs,
                               const std::vector<std::vector<width_t>>& group_offdiag_inds,
                               const unsigned int num_groups,
                               const unsigned int ladder_offset,
                               HalfStrTables<T>& tables)
{
    using Key = typename Policy::Key;
    const width_t hw = static_cast<width_t>(width / 2);
    const width_t bw = static_cast<width_t>(width - hw);
    tables.width = width;
    tables.half_width = hw;

    const auto* bitsets = subspace.get_bitsets();
    auto akey = [&](const boost::dynamic_bitset<std::size_t>& bs) -> Key {
        return Policy::extract(bs, 0, hw);
    };
    auto bkey = [&](const boost::dynamic_bitset<std::size_t>& bs) -> Key {
        return Policy::extract(bs, hw, bw);
    };

    // Fast path: clean beta x alpha Cartesian product (half-strs mode). Detected by a
    // run-length on beta + full verify.
    std::size_t num_alpha = 0, num_beta = 0;
    std::vector<Key> akeys, bkeys;
    bool cart = false; // is Cartesian?
    {
        const Key b0 = bkey(bitsets[0].first);
        std::size_t na = 1;
        while(na < subspace_dim && bkey(bitsets[na].first) == b0)
            na++;
        if(na > 0 && subspace_dim % na == 0)
        {
            const std::size_t nb = subspace_dim / na;
            std::vector<Key> ak(na), bk(nb);
            for(std::size_t a = 0; a < na; a++)
                ak[a] = akey(bitsets[a].first);
            for(std::size_t b = 0; b < nb; b++)
                bk[b] = bkey(bitsets[b * na].first);
            cart = true;
            for(std::size_t idx = 0; idx < subspace_dim && cart; idx++)
            {
                const auto& bs = bitsets[idx].first;
                if(akey(bs) != ak[idx % na] || bkey(bs) != bk[idx / na])
                    cart = false;
            }
            if(cart)
            {
                num_alpha = na;
                num_beta = nb;
                akeys.swap(ak);
                bkeys.swap(bk);
            }
        }
    }

    // Profiling knob (FQ_FORCE_NONCART): treat a clean Cartesian product as if it were
    // non-Cartesian, so the *identical* determinant set can be driven down both Cartesian
    // and non-Cartesian paths for apples-to-apples comparison.
    // Not used in production.
    if(cart && std::getenv("FQ_FORCE_NONCART"))
    {
        cart = false;
        akeys.clear();
        bkeys.clear();
        num_alpha = 0;
        num_beta = 0;
    }

    // Full-strs: arbitrary (non-Cartesian) determinant set. Assign alpha/beta string
    // ids by first appearance -- for a Cartesian subspace this reproduces the ids the
    // fast path above uses, so the two agree by construction.
    std::vector<std::uint32_t> det_a, det_b;
    if(!cart)
    {
        // The grid-line scan needs raw 64-bit half keys; the dynamic_bitset tier has
        // none, so >256 orbitals/sector stays on type2_visit_groups for now.
        if constexpr(Policy::words == 0)
        {
            warn_halfstr_fallback("this is a full-strs (non-Cartesian) subspace with more than "
                                  "256 orbitals per spin sector, so its half keys land on the "
                                  "dynamic_bitset tier, which has no raw 64-bit words for the "
                                  "grid-line scan to XOR");
            return;
        }
        else
        {
            // The grid CSR stores a determinant's flat index (row_flat / col_flat) and its
            // alpha/beta string id (row_alpha / col_beta) as uint32 -- four subspace_dim-sized
            // arrays, so 16 B/det rather than 32 B. Nothing needs 64-bit indices: at 2^32
            // determinants those arrays alone are 69 GB and a single matvec is already far
            // out of reach, so the halved footprint is worth more than the headroom.
            // Revisit once Fulqrum supports distributed computing.
            if(subspace_dim > UINT32_MAX)
            {
                warn_halfstr_fallback("this is a full-strs (non-Cartesian) subspace with more "
                                      "than 2^32 determinants, which overflows the uint32 flat "
                                      "indices in the grid CSR");
                return;
            }
            det_a.resize(subspace_dim);
            det_b.resize(subspace_dim);
            typename Policy::template Map<std::uint32_t> am, bm;
            for(std::size_t idx = 0; idx < subspace_dim; idx++)
            {
                const auto& bs = bitsets[idx].first;
                const Key ka = akey(bs), kb = bkey(bs);
                auto ai = am.find(ka);
                if(ai == am.end())
                {
                    det_a[idx] = static_cast<std::uint32_t>(akeys.size());
                    am[ka] = det_a[idx];
                    akeys.push_back(ka);
                }
                else
                    det_a[idx] = ai->second;
                auto bi = bm.find(kb);
                if(bi == bm.end())
                {
                    det_b[idx] = static_cast<std::uint32_t>(bkeys.size());
                    bm[kb] = det_b[idx];
                    bkeys.push_back(kb);
                }
                else
                    det_b[idx] = bi->second;
            }
            num_alpha = akeys.size();
            num_beta = bkeys.size();
        }
    }
    tables.cartesian = cart;

    typename Policy::template Map<std::uint32_t> amap, bmap;
    amap.reserve(num_alpha * 2 + 1);
    bmap.reserve(num_beta * 2 + 1);
    for(std::uint32_t a = 0; a < num_alpha; a++)
        amap[akeys[a]] = a;
    for(std::uint32_t b = 0; b < num_beta; b++)
        bmap[bkeys[b]] = b;

    flatten_offdiag_inds(group_offdiag_inds, tables.flat_inds, tables.inds_offsets);
    auto gview = [&](std::size_t g) -> GroupIndsView {
        const std::size_t off = tables.inds_offsets[g];
        return GroupIndsView{tables.flat_inds.data() + off, tables.inds_offsets[g + 1] - off};
    };
    const std::size_t ladder_len = static_cast<std::size_t>(num_groups) * ladder_offset + 1;
    tables.ladder32.resize(ladder_len);
    for(std::size_t i = 0; i < ladder_len; i++)
        tables.ladder32[i] = static_cast<std::uint32_t>(group_ladder_ptrs[i]);

    // Invert groups: flip-position set -> group id. Kept in the tables because the
    // type2_visit_non_cartesian recovers g from the flip XOR at each scan hit.
    auto& aa_map = tables.aa_map;
    auto& bb_map = tables.bb_map;
    auto& aaaa_map = tables.aaaa_map;
    auto& bbbb_map = tables.bbbb_map;
    std::unordered_map<std::uint32_t, int> a_pair_id, b_pair_id;
    struct AabbTmp
    {
        int ra;
        int rb;
        T val;
    };
    std::vector<AabbTmp> aabb_tmp;

    for(std::size_t g = 0; g < num_groups; g++)
    {
        const GroupIndsView inds = gview(g);
        const std::size_t sz = inds.size();
        const bool nonempty = group_ptrs[g + 1] > group_ptrs[g];
        if(sz == 2)
        {
            if(inds[1] < hw)
                aa_map[pack2(inds[0], inds[1])] = g;
            else if(inds[0] >= hw)
                bb_map[pack2(static_cast<width_t>(inds[0] - hw),
                             static_cast<width_t>(inds[1] - hw))] = g;
            else if(nonempty)
            {
                warn_halfstr_fallback("the operator has a mixed single excitation (one alpha + "
                                      "one beta index), which no enumerator models");
                return;
            }
        }
        else if(sz == 4)
        {
            if(inds[3] < hw)
                aaaa_map[pack4(inds[0], inds[1], inds[2], inds[3])] = g;
            else if(inds[0] >= hw)
                bbbb_map[pack4(static_cast<width_t>(inds[0] - hw),
                               static_cast<width_t>(inds[1] - hw),
                               static_cast<width_t>(inds[2] - hw),
                               static_cast<width_t>(inds[3] - hw))] = g;
            else if(inds[1] < hw && inds[2] >= hw)
            {
                if(!nonempty)
                    continue;
                // Direct-formula eligibility: all terms share coeff + real_phase.
                const OperatorTerm_t& first = terms[group_ptrs[g]];
                const std::complex<double> gc = first.coeff;
                const int gp = first.real_phase;
                for(std::size_t i = group_ptrs[g]; i < group_ptrs[g + 1]; i++)
                    if(terms[i].real_phase != gp || std::abs(terms[i].coeff - gc) > 1e-14)
                    {
                        warn_halfstr_fallback("an aabb group needs the per-term accum_element "
                                              "slow path (its terms do not share a single coeff "
                                              "+ real_phase), so the direct asign*bsign formula "
                                              "does not apply");
                        return;
                    }
                T val;
                if constexpr(std::is_same_v<T, double>)
                    val = gp * gc.real();
                else
                    val = static_cast<T>(gp) * gc;
                const std::uint32_t ak = pack2(inds[0], inds[1]);
                const std::uint32_t bk =
                    pack2(static_cast<width_t>(inds[2] - hw), static_cast<width_t>(inds[3] - hw));
                auto ai = a_pair_id.find(ak);
                const int ra = (ai == a_pair_id.end())
                                   ? (a_pair_id[ak] = static_cast<int>(a_pair_id.size()))
                                   : ai->second;
                auto bi = b_pair_id.find(bk);
                const int rb = (bi == b_pair_id.end())
                                   ? (b_pair_id[bk] = static_cast<int>(b_pair_id.size()))
                                   : bi->second;
                aabb_tmp.push_back({ra, rb, val});
            }
            else if(nonempty)
            {
                warn_halfstr_fallback("the operator has a 4-index term outside the aaaa / bbbb / "
                                      "aabb classes");
                return;
            }
        }
        else if(nonempty)
        {
            warn_halfstr_fallback("the operator has a term of Pauli weight other than 2 or 4");
            return;
        }
    }

    const std::size_t num_a_pairs = a_pair_id.size();
    const std::size_t num_b_pairs = b_pair_id.size();
    tables.aabb_val_2d.assign(num_a_pairs * num_b_pairs, T(0));
    for(const auto& e : aabb_tmp)
        tables.aabb_val_2d[static_cast<std::size_t>(e.ra) * num_b_pairs + e.rb] = e.val;

    tables.alpha_single.assign(num_alpha, {});
    tables.alpha_double.assign(num_alpha, {});
    tables.beta_single.assign(num_beta, {});
    tables.beta_double.assign(num_beta, {});

    // parity of set bits strictly between p0 and p1 (i.e. in [p0+1, p1)).
    auto pair_sign = [&](const Key& key, width_t p0, width_t p1) -> int {
        return Policy::range_parity(key, static_cast<width_t>(p0 + 1), p1) ? -1 : 1;
    };

    // with_doubles=false (full-strs): the double-excitation lists need huge memory
    // (~1.3e9 conns/sector at fe4s4 scale) and the grid-line scan finds those partners
    // , so they are not built. Singles are still needed to drive aabb.
    auto build_sector =
        [&](std::size_t n,
            const std::vector<Key>& keys,
            width_t nbits,
            typename Policy::template Map<std::uint32_t>& idxmap,
            emhash8::HashMap<std::uint32_t, std::size_t, PackedFlipHash>& single_map,
            emhash8::HashMap<std::uint64_t, std::size_t, PackedFlipHash>& double_map,
            std::unordered_map<std::uint32_t, int>& pair_id,
            std::vector<std::vector<HalfConnSingle>>& singles_out,
            std::vector<std::vector<HalfConnDouble>>& doubles_out,
            bool with_doubles) {

#pragma omp parallel if(n > 256)
            {
                std::vector<width_t> occ(nbits), emp(nbits);
#pragma omp for schedule(dynamic)
                for(std::size_t s = 0; s < n; s++)
                {
                    const Key kv = keys[s];
                    unsigned no = 0, ne = 0;
                    for(width_t p = 0; p < nbits; p++)
                    {
                        if(Policy::test(kv, p))
                            occ[no++] = p;
                        else
                            emp[ne++] = p;
                    }
                    auto& singles = singles_out[s];
                    auto& doubles = doubles_out[s];
                    for(unsigned oi = 0; oi < no; oi++)
                        for(unsigned ej = 0; ej < ne; ej++)
                        {
                            const width_t i = occ[oi], j = emp[ej];
                            Key Kp = kv;
                            Policy::flip(Kp, i);
                            Policy::flip(Kp, j);
                            auto it = idxmap.find(Kp);
                            if(it == idxmap.end())
                                continue;
                            const width_t p0 = i < j ? i : j;
                            const width_t p1 = i < j ? j : i;
                            const std::uint32_t pk = pack2(p0, p1);
                            std::size_t g_same = NO_HALF_G;
                            auto gi = single_map.find(pk);
                            if(gi != single_map.end())
                                g_same = gi->second;
                            int rank = -1;
                            auto ri = pair_id.find(pk);
                            if(ri != pair_id.end())
                                rank = ri->second;
                            singles.push_back({it->second, g_same, rank, pair_sign(kv, p0, p1)});
                        }
                    if(!with_doubles)
                        continue;
                    for(unsigned oi = 0; oi < no; oi++)
                        for(unsigned oj = oi + 1; oj < no; oj++)
                            for(unsigned ei = 0; ei < ne; ei++)
                                for(unsigned ej = ei + 1; ej < ne; ej++)
                                {
                                    const width_t i1 = occ[oi], i2 = occ[oj], j1 = emp[ei],
                                                  j2 = emp[ej];
                                    Key Kp = kv;
                                    Policy::flip(Kp, i1);
                                    Policy::flip(Kp, i2);
                                    Policy::flip(Kp, j1);
                                    Policy::flip(Kp, j2);
                                    auto it = idxmap.find(Kp);
                                    if(it == idxmap.end())
                                        continue;
                                    width_t sd[4] = {i1, i2, j1, j2};
                                    std::sort(sd, sd + 4);
                                    auto gi = double_map.find(pack4(sd[0], sd[1], sd[2], sd[3]));
                                    if(gi == double_map.end())
                                        continue;
                                    doubles.push_back({it->second, gi->second});
                                }
                }
            }
        };

    build_sector(num_alpha,
                 akeys,
                 hw,
                 amap,
                 aa_map,
                 aaaa_map,
                 a_pair_id,
                 tables.alpha_single,
                 tables.alpha_double,
                 cart);
    build_sector(num_beta,
                 bkeys,
                 bw,
                 bmap,
                 bb_map,
                 bbbb_map,
                 b_pair_id,
                 tables.beta_single,
                 tables.beta_double,
                 cart);

    tables.sym = (std::getenv("FQ_FS_NOSYM") == nullptr);
    tables.chunk = 4096;
    if(const char* e = std::getenv("FQ_FS_CHUNK"))
    {
        const long v = std::atol(e);
        if(v > 0)
            tables.chunk = static_cast<std::size_t>(v);
    }

    // Cartesian: grid lines are implicit, so only the task lists are needed
    if(cart)
    {
#ifdef _OPENMP
        const std::size_t nthr = static_cast<std::size_t>(omp_get_max_threads());
#else
        const std::size_t nthr = 1;
#endif
        auto mk = [&](std::size_t nlines, std::size_t linelen, std::vector<FsTask>& out) {
            const bool whole = (nlines >= 4 * nthr) && (linelen <= UINT32_MAX);
            for(std::size_t g = 0; g < nlines; g++)
            {
                if(whole)
                    out.push_back(
                        {static_cast<std::uint32_t>(g), 0, static_cast<std::uint32_t>(linelen)});
                else
                    for(std::size_t s = 0; s < linelen; s += tables.chunk)
                        out.push_back(
                            {static_cast<std::uint32_t>(g),
                             static_cast<std::uint32_t>(s),
                             static_cast<std::uint32_t>(std::min(s + tables.chunk, linelen))});
            }
        };
        mk(num_beta, num_alpha, tables.task_a);
        mk(num_alpha, num_beta, tables.task_b);
    }

    // full-strs: sparse grid CSR (beta-major) + transpose (alpha-major)
    if constexpr(Policy::words > 0)
    {
        if(!cart)
        {
            const std::size_t kw = Policy::words;
            tables.kw = kw;
            tables.akeyw.resize(num_alpha * kw);
            tables.bkeyw.resize(num_beta * kw);
            for(std::size_t a = 0; a < num_alpha; a++)
                for(std::size_t w = 0; w < kw; w++)
                    tables.akeyw[a * kw + w] = akeys[a].w[w];
            for(std::size_t b = 0; b < num_beta; b++)
                for(std::size_t w = 0; w < kw; w++)
                    tables.bkeyw[b * kw + w] = bkeys[b].w[w];

            tables.a_pair_rank.assign(static_cast<std::size_t>(hw) * hw, -1);
            for(const auto& kv : a_pair_id)
            {
                const width_t p0 = static_cast<width_t>(kv.first >> 16);
                const width_t p1 = static_cast<width_t>(kv.first & 0xFFFFu);
                tables.a_pair_rank[static_cast<std::size_t>(p0) * hw + p1] = kv.second;
            }
            tables.scan_max = 64;
            if(const char* e = std::getenv("FQ_FS_SCANMAX"))
            {
                const long v = std::atol(e);
                if(v >= 0)
                    tables.scan_max = static_cast<std::size_t>(v);
            }
            // Tier 2/3: same-sector lines longer than residue_min uses Tier 2/3.
            tables.residue_min = 4096;
            if(const char* e = std::getenv("FQ_FS_RESIDUE_MIN"))
            {
                const long v = std::atol(e);
                if(v >= 0)
                    tables.residue_min = static_cast<std::size_t>(v);
            }

            tables.row_start.assign(num_beta + 1, 0);
            tables.col_start.assign(num_alpha + 1, 0);
            for(std::size_t i = 0; i < subspace_dim; i++)
            {
                tables.row_start[det_b[i] + 1]++;
                tables.col_start[det_a[i] + 1]++;
            }
            for(std::size_t b = 0; b < num_beta; b++)
                tables.row_start[b + 1] += tables.row_start[b];
            for(std::size_t a = 0; a < num_alpha; a++)
                tables.col_start[a + 1] += tables.col_start[a];

            tables.row_alpha.resize(subspace_dim);
            tables.row_flat.resize(subspace_dim);
            tables.col_beta.resize(subspace_dim);
            tables.col_flat.resize(subspace_dim);
            {
                std::vector<std::size_t> rc(tables.row_start.begin(), tables.row_start.end() - 1);
                std::vector<std::size_t> cc(tables.col_start.begin(), tables.col_start.end() - 1);
                for(std::size_t i = 0; i < subspace_dim; i++)
                {
                    const std::size_t p = rc[det_b[i]]++;
                    tables.row_alpha[p] = det_a[i];
                    tables.row_flat[p] = static_cast<std::uint32_t>(i);
                    const std::size_t q = cc[det_a[i]]++;
                    tables.col_beta[q] = det_b[i];
                    tables.col_flat[q] = static_cast<std::uint32_t>(i);
                }
            }

            auto make_tasks = [&](const std::vector<std::size_t>& start,
                                  std::size_t n,
                                  std::vector<FsTask>& out) {
                for(std::size_t g = 0; g < n; g++)
                {
                    const std::size_t len = start[g + 1] - start[g];
                    if(len == 0)
                        continue;
                    const bool residue = tables.residue_min && len > tables.residue_min;
                    if(len <= tables.chunk || residue)
                        out.push_back(
                            {static_cast<std::uint32_t>(g), 0, static_cast<std::uint32_t>(len)});
                    else
                        for(std::size_t s = 0; s < len; s += tables.chunk)
                            out.push_back(
                                {static_cast<std::uint32_t>(g),
                                 static_cast<std::uint32_t>(s),
                                 static_cast<std::uint32_t>(std::min(s + tables.chunk, len))});
                }
            };
            make_tasks(tables.row_start, num_beta, tables.task_a);
            make_tasks(tables.col_start, num_alpha, tables.task_b);

            auto sort_tasks = [&](std::vector<FsTask>& tv, const std::vector<std::size_t>& start) {
                auto cost = [&](const FsTask& t) -> double {
                    const double len = static_cast<double>(start[t.grp + 1] - start[t.grp]);
                    const double span = static_cast<double>(t.end - t.beg);
                    const bool whole = (t.beg == 0 && static_cast<double>(t.end) == len);
                    if(tables.residue_min && len > static_cast<double>(tables.residue_min))
                        return len * 8200.0; // residue task ~8.2 us/det (measured)
                    if(whole)
                        return len * len * 6.7; // whole-line triangle scan (~6.7 ns/pair)
                    return span * len * 6.7; // chunk scans the whole line per det
                };
                std::sort(tv.begin(), tv.end(), [&](const FsTask& a, const FsTask& b) {
                    return cost(a) > cost(b);
                });
            };
            sort_tasks(tables.task_a, tables.row_start);
            sort_tasks(tables.task_b, tables.col_start);

            auto pack_aabb = [](std::vector<std::vector<HalfConnSingle>>& src,
                                std::vector<std::uint64_t>& off,
                                std::vector<AabbConn>& out) {
                off.assign(src.size() + 1, 0);
                for(std::size_t s = 0; s < src.size(); s++)
                {
                    std::size_t c = 0;
                    for(const auto& e : src[s])
                        if(e.pair_rank >= 0)
                            c++;
                    off[s + 1] = off[s] + c;
                }
                out.resize(off.back());
                for(std::size_t s = 0; s < src.size(); s++)
                {
                    std::size_t w = off[s];
                    for(const auto& e : src[s])
                        if(e.pair_rank >= 0)
                            out[w++] = {static_cast<std::uint32_t>(e.col),
                                        static_cast<std::int16_t>(e.pair_rank),
                                        static_cast<std::int16_t>(e.sign)};
                }
                src.clear();
                src.shrink_to_fit();
            };
            pack_aabb(tables.alpha_single, tables.a_aabb_off, tables.a_aabb);
            pack_aabb(tables.beta_single, tables.b_aabb_off, tables.b_aabb);

            // rb-major coeff table for the non-Cartesian aabb loops (see field comment).
            tables.aabb_val_2d_t.assign(num_b_pairs * num_a_pairs, T(0));
            for(std::size_t ra = 0; ra < num_a_pairs; ra++)
                for(std::size_t rb = 0; rb < num_b_pairs; rb++)
                    tables.aabb_val_2d_t[rb * num_a_pairs + ra] =
                        tables.aabb_val_2d[ra * num_b_pairs + rb];
        }
    }

    tables.num_alpha = num_alpha;
    tables.num_beta = num_beta;
    tables.num_a_pairs = num_a_pairs;
    tables.num_b_pairs = num_b_pairs;
    tables.BLK = 128;
    if(const char* e = std::getenv("FQ_BLK"))
    {
        long v = std::atol(e);
        if(v > 0)
            tables.BLK = static_cast<std::size_t>(v);
    }
    tables.rsb_w = width;
    tables.num_blocks = (subspace_dim + tables.BLK - 1) / tables.BLK;
    tables.usable = true;

    if(std::getenv("FQ_HALFSTR_VERBOSE"))
    {
        std::size_t asc = 0, adc = 0, bsc = 0, bdc = 0;
        for(const auto& v : tables.alpha_single)
            asc += v.size();
        for(const auto& v : tables.alpha_double)
            adc += v.size();
        for(const auto& v : tables.beta_single)
            bsc += v.size();
        for(const auto& v : tables.beta_double)
            bdc += v.size();
        char ktag[16];
        if(Policy::words)
            std::snprintf(ktag, sizeof(ktag), "%zu", Policy::words);
        else
            std::snprintf(ktag, sizeof(ktag), "dyn");
        std::fprintf(stderr,
                     "[%s] ACTIVE (K=%s): num_alpha=%zu num_beta=%zu a_pairs=%zu b_pairs=%zu"
                     " | conns alpha_single=%zu alpha_double=%zu beta_single=%zu beta_double=%zu"
                     " | aabb_packed a=%zu b=%zu\n",
                     cart ? "halfstr" : "fullstr",
                     ktag,
                     num_alpha,
                     num_beta,
                     num_a_pairs,
                     num_b_pairs,
                     asc,
                     adc,
                     bsc,
                     bdc,
                     tables.a_aabb.size(),
                     tables.b_aabb.size());
        if(!cart)
            std::fprintf(stderr,
                         "[fullstr] grid: dim=%zu occupancy=%.3e tasks=%zu/%zu chunk=%zu sym=%d\n",
                         subspace_dim,
                         static_cast<double>(subspace_dim) /
                             (static_cast<double>(num_alpha) * static_cast<double>(num_beta)),
                         tables.task_a.size(),
                         tables.task_b.size(),
                         tables.chunk,
                         static_cast<int>(tables.sym));
    }
}

// Templated on half_width: pick the smallest key representation that holds a half
// string, then build with it. All tiers run the same connected-det algorithm
// (build_halfstr_tables_impl).
//   FixedKeyOps<1>: <=64 orbitals/spin-sector  (full det <=128 bits)
//   FixedKeyOps<2>: 65-128                     (<=256 bits)
//   FixedKeyOps<4>: 129-256                    (<=512 bits)
//   DynKeyOps     : >256                       (arbitrary; dynamic_bitset)
// A Cartesian subspace runs on every tier, so it never falls back. A non-Cartesian one
// (type2_visit_non_cartesian) needs raw 64-bit half keys for the grid-line scan, which DynKeyOps has
// no equivalent of -- so >256 orbitals/spin-sector is the one case that still leaves
// usable=false and drops to type2_visit_groups.
// FQ_HALFSTR_FORCE_K={1,2,4} forces a wider fixed tier instead of auto-picking;
// FQ_HALFSTR_FORCE_K=0 forces the dynamic tier.
template <typename T>
void build_halfstr_tables(const std::vector<OperatorTerm_t>& terms,
                          const bitset_map_namespace::BitsetHashMapWrapper& subspace,
                          const width_t width,
                          const std::size_t subspace_dim,
                          const std::size_t* __restrict group_ptrs,
                          const std::size_t* __restrict group_ladder_ptrs,
                          const std::vector<std::vector<width_t>>& group_offdiag_inds,
                          const unsigned int num_groups,
                          const unsigned int ladder_offset,
                          HalfStrTables<T>& tables)
{
    tables = HalfStrTables<T>{};
    const width_t hw = static_cast<width_t>(width / 2);
    const width_t bw = static_cast<width_t>(width - hw);
    tables.width = width;
    tables.half_width = hw;

    if(subspace_dim == 0 || hw == 0 || terms.size() > UINT32_MAX)
        return;

    const std::size_t half_bits = static_cast<std::size_t>(hw > bw ? hw : bw);
    const std::size_t words = (half_bits + 63) / 64;

    std::size_t K = (words <= 1) ? 1 : (words <= 2) ? 2 : (words <= 4) ? 4 : 0;

    if(const char* fk = std::getenv("FQ_HALFSTR_FORCE_K"))
    {
        const long v = std::atol(fk);
        if(v == 0)
            K = 0;
        else if((v == 1 || v == 2 || v == 4) && static_cast<std::size_t>(v) >= words)
            K = static_cast<std::size_t>(v);
    }
    switch(K)
    {
    case 1:
        build_halfstr_tables_impl<T, FixedKeyOps<1>>(terms,
                                                     subspace,
                                                     width,
                                                     subspace_dim,
                                                     group_ptrs,
                                                     group_ladder_ptrs,
                                                     group_offdiag_inds,
                                                     num_groups,
                                                     ladder_offset,
                                                     tables);
        break;
    case 2:
        build_halfstr_tables_impl<T, FixedKeyOps<2>>(terms,
                                                     subspace,
                                                     width,
                                                     subspace_dim,
                                                     group_ptrs,
                                                     group_ladder_ptrs,
                                                     group_offdiag_inds,
                                                     num_groups,
                                                     ladder_offset,
                                                     tables);
        break;
    case 4:
        build_halfstr_tables_impl<T, FixedKeyOps<4>>(terms,
                                                     subspace,
                                                     width,
                                                     subspace_dim,
                                                     group_ptrs,
                                                     group_ladder_ptrs,
                                                     group_offdiag_inds,
                                                     num_groups,
                                                     ladder_offset,
                                                     tables);
        break;
    default:
        // >256 orbitals/sector (or forced): arbitrary-width dynamic_bitset tier.
        build_halfstr_tables_impl<T, DynKeyOps>(terms,
                                                subspace,
                                                width,
                                                subspace_dim,
                                                group_ptrs,
                                                group_ladder_ptrs,
                                                group_offdiag_inds,
                                                num_groups,
                                                ladder_offset,
                                                tables);
        break;
    }
}
