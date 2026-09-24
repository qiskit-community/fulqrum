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
#include <cstdint>
#include "constants.hpp"

/**
 * Pack operator index and value into a single width_t
 *
 * @param[in] ind The index on which the operator acts
 * @param[in] val Operator value
 * @param[out] packed_data Ind and val packed into width_t
 */
inline width_t pack_data(const width_t ind, const unsigned char val)
{
    return (ind << 3) | val;
}

/**
 * Extract operator index from a packed width_t
 *
 * @param[in] packed_data Packed index and operator value
 * @param[out] ind Indice as a width_t
 */
inline width_t get_ind(const width_t packed_data)
{
    return (packed_data >> 3);
}

/**
 * Extract operator value from a packed width_t
 *
 * @param[in] packed_data Packed index and operator value
 * @param[out] val Value as unsigned char
 */
inline unsigned char get_val(const width_t packed_data)
{
    return (packed_data & static_cast<width_t>(7));
}

/**
 * Extract operator index AND value from a packed width_t
 *
 * @param[in] packed_data Packed index and operator value
 * @param[out] indval_pair Pair of indice and value
 */
inline std::pair<width_t, unsigned char> get_indval_pair(const width_t packed_data)
{
    width_t ind = (packed_data >> 3);
    unsigned char val = (packed_data & static_cast<width_t>(7));
    return std::make_pair(ind, val);

}
