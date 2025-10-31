/**
 * Fulqrum
 * Copyright (C) 2024, IBM
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
