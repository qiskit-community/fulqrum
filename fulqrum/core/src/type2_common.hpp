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
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "constants.hpp"
// For rapidhashNano (used by PackedFlipHash below). NOTE: external/rapidhash.h has no
// include guard and no #pragma once, so it must never be included directly from more than
// one header -- go through bitset_hashmap.hpp, which owns the single include of it.
// We did not edit rapidHash's cource code.
#include "bitset_hashmap.hpp"

// Primitives shared by all three type-2 element visitors (type2_visit_groups,
// type2_visit_cartesian, type2_visit_non_cartesian) and by build_halfstr_tables.

// A group has no same-sector partner value.
constexpr std::size_t NO_HALF_G = MAX_SIZE_T;

// Pack the flip positions of a group into one integer key. Two orbital indices for a
// single excitation, four for a double. These are the keys of the group lookup maps, so
// they must be collision-free.
//
// Widening width_t silently truncates the 16-bit fields: distinct groups would collide
// onto one key and every type-2 path would return wrong numbers with no crash and no
// warning. Fail the build instead.
static_assert(sizeof(width_t) == 2,
              "type-2 paths pack orbital indices into 16-bit fields (pack2/pack4). A wider "
              "width_t would silently truncate them into key collisions -- widen the packed "
              "key types and their shifts before changing width_t.");

inline std::uint32_t pack2(width_t a, width_t b) noexcept
{
    return (static_cast<std::uint32_t>(a) << 16) | static_cast<std::uint32_t>(b);
}

inline std::uint64_t pack4(width_t a, width_t b, width_t c, width_t d) noexcept
{
    return (static_cast<std::uint64_t>(a) << 48) | (static_cast<std::uint64_t>(b) << 32) |
           (static_cast<std::uint64_t>(c) << 16) | static_cast<std::uint64_t>(d);
}

// Hash for the packed flip-position keys produced by pack2/pack4 above.
struct PackedFlipHash
{
    std::size_t operator()(std::uint64_t k) const noexcept
    {
        return static_cast<std::size_t>(rapidhashNano(&k, sizeof(k)));
    }
    std::size_t operator()(std::uint32_t k) const noexcept
    {
        return (*this)(static_cast<std::uint64_t>(k));
    }
};

inline double pt_eval_terms(const std::uint64_t* __restrict pt_masks,
                            const double* __restrict pt_mag,
                            const std::size_t tw,
                            const std::size_t* __restrict rw,
                            const std::size_t s,
                            const std::size_t e) noexcept
{
    double val = 0;
    if(tw == 2)
    {
        const std::uint64_t r0 = rw[0], r1 = rw[1];
        const std::uint64_t* __restrict p = pt_masks + s * 6;
        for(std::size_t idx = s; idx < e; idx++, p += 6)
        {
            const std::uint64_t neq = ((r0 & p[0]) ^ p[2]) | ((r1 & p[1]) ^ p[3]);
            const std::uint64_t par = (r0 & p[4]) ^ (r1 & p[5]);
            std::uint64_t mb;
            std::memcpy(&mb, &pt_mag[idx], 8);
            mb ^= static_cast<std::uint64_t>(__builtin_popcountll(par) & 1) << 63;
            mb &= -static_cast<std::uint64_t>(neq == 0);
            double f;
            std::memcpy(&f, &mb, 8);
            val += f;
        }
    }
    else if(tw == 1)
    {
        const std::uint64_t r0 = rw[0];
        const std::uint64_t* __restrict p = pt_masks + s * 3;
        for(std::size_t idx = s; idx < e; idx++, p += 3)
        {
            const std::uint64_t neq = (r0 & p[0]) ^ p[1];
            const std::uint64_t par = r0 & p[2];
            std::uint64_t mb;
            std::memcpy(&mb, &pt_mag[idx], 8);
            mb ^= static_cast<std::uint64_t>(__builtin_popcountll(par) & 1) << 63;
            mb &= -static_cast<std::uint64_t>(neq == 0);
            double f;
            std::memcpy(&f, &mb, 8);
            val += f;
        }
    }
    else
    {
        const std::size_t stride = 3 * tw;
        const std::uint64_t* __restrict p = pt_masks + s * stride;
        for(std::size_t idx = s; idx < e; idx++, p += stride)
        {
            std::uint64_t neq = 0, par = 0;
            for(std::size_t w = 0; w < tw; w++)
            {
                const std::uint64_t r = rw[w];
                neq |= (r & p[w]) ^ p[tw + w];
                par ^= r & p[2 * tw + w];
            }
            std::uint64_t mb;
            std::memcpy(&mb, &pt_mag[idx], 8);
            mb ^= static_cast<std::uint64_t>(__builtin_popcountll(par) & 1) << 63;
            mb &= -static_cast<std::uint64_t>(neq == 0);
            double f;
            std::memcpy(&f, &mb, 8);
            val += f;
        }
    }
    return val;
}

// Every early return in build_halfstr_tables_impl leaves usable=false, which drops the caller
// onto type2_visit_groups. That answer is still correct; just orders of magnitude slower
// (measured 634 s vs 5.6 s per matvec on 1M fe4s4 determinants). Taken silently it is
// indistinguishable from a run that is merely big, so explicitly show the issue.
inline void warn_halfstr_fallback(const char* why)
{
    std::fprintf(stderr,
                 "[fulqrum] WARNING: the accelerated type-2 enumerator is unavailable: %s.\n"
                 "[fulqrum]          Falling back to type2_visit_groups. It is correct but orders\n"
                 "[fulqrum]          of magnitude slower on large subspaces.\n",
                 why);
    std::fflush(stderr);
}
