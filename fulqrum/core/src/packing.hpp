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
#include <cstdint>
#include "constants.hpp"


/**
 * Pack operator index and value into a single uint32
 *
 * @param ind The index on which the operator acts
 * @param val Operator value
 */
inline std::uint32_t pack_data(const width_t ind, const unsigned char val)
{
    return (static_cast<std::uint32_t>(ind) << 8) | val;
}

/**
 * Extract operator index from a packed uint32
 *
 * @param packed_data Packed index and operator value
 */
inline width_t get_ind(const std::uint32_t packed_data)
{
    return static_cast<width_t>((packed_data & 0xFFFFFF00) >> 8);
}

/**
 * Extract operator value from a packed uint32
 *
 * @param packed_data Packed index and operator value
 */
inline unsigned char get_val(const std::uint32_t packed_data)
{
    return static_cast<unsigned char>(packed_data & 0x000000FF);
}
