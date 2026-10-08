/**
 * This code is part of Fulqrum.
 *
 * (C) Copyright IBM 2026.
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
#include <cstdint>

/**
 * Pack operator index and value into a single width_t
 *
 * @param[in] ind The index on which the operator acts
 * @param[in] val Operator value
 * @param[out] packed_data Ind and val packed into width_t
 */
inline width_t pack_indval(const width_t ind, const unsigned char val)
{
    return static_cast<width_t>((ind << 3) | val);
}

/**
 * Extract operator index from a packed width_t
 *
 * @param[in] packed_data Packed index and operator value
 * @param[out] ind Indice as a width_t
 */
inline width_t unpack_ind(const width_t packed_data)
{
    return static_cast<width_t>(packed_data >> 3);
}

/**
 * Extract operator value from a packed width_t
 *
 * @param[in] packed_data Packed index and operator value
 * @param[out] val Value as unsigned char
 */
inline unsigned char unpack_val(const width_t packed_data)
{
    return static_cast<unsigned char>(packed_data & 7);
}

/**
 * Extract operator index AND value from a packed width_t
 *
 * @param[in] packed_data Packed index and operator value
 * @param[out] indval_pair Pair of indice and value
 */
inline std::pair<width_t, unsigned char> unpack_indval(const width_t packed_data)
{
    return {static_cast<width_t>(packed_data >> 3), static_cast<unsigned char>(packed_data & 7)};
}
