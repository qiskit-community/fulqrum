/**
 * This code is a Qiskit project.
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
#include <cstdlib>

const double ATOL = 1e-14;
const std::size_t MAX_SIZE_T = (std::size_t)-1;
const unsigned int MAX_UINT = (unsigned int)-1;
const unsigned int BITS_PER_BLOCK = 8 * sizeof(std::size_t);
unsigned int DEFAULT_LADDER_WIDTH = 4;
const unsigned int BLOCK_EXPONENT = __builtin_ctz(BITS_PER_BLOCK);
const unsigned int BLOCK_SHIFT = BITS_PER_BLOCK - 1;
