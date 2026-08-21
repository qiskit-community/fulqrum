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



inline std::uint32_t pack_data(const width_t ind, const unsigned char val)
{
    return (static_cast<std::uint32_t>(ind) << 8) | val;
}


inline width_t get_ind(const std::uint32_t packed_val)
{
    return static_cast<width_t>((packed_val & 0xFFFFFF00) >> 8);
}


inline unsigned char get_val(const std::uint32_t packed_val)
{
    return static_cast<unsigned char>(packed_val & 0x000000FF);
}
