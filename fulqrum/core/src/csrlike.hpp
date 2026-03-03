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
#include <chrono>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// Data types for CSR-like matrix structure

typedef struct RowData_Real64
{
    std::vector<std::vector<long long>> cols;
    std::vector<std::vector<double>> data;
} RowData_Real64_t;

typedef struct RowData_Real32
{
    std::vector<std::vector<int>> cols;
    std::vector<std::vector<double>> data;
} RowData_Real32_t;

typedef struct RowData_Complex64
{
    std::vector<std::vector<long long>> cols;
    std::vector<std::vector<std::complex<double>>> data;
} RowData_Complex64_t;

typedef struct RowData_Complex32
{
    std::vector<std::vector<int>> cols;
    std::vector<std::vector<std::complex<double>>> data;
} RowData_Complex32_t;

/**
 * Set the row-pointers array data from the data for a CSRLike struct
 *
 * @param row_data The data for each row in the matrix
 * @param ptrs A pointer to the indptr array for a CSR matrix of length num_rows+1
 *
 */
template <typename T, typename U>
void set_csr_ptr(const std::vector<std::vector<T>>& cols, U* __restrict ptrs)
{
    std::size_t num_rows = cols.size();
    std::size_t kk;
    std::size_t temp, _sum = 0;
    for(kk = 0; kk < num_rows; kk++)
    {
        ptrs[kk] = cols[kk].size();
        temp = _sum + ptrs[kk];
        ptrs[kk] = _sum;
        _sum = temp;
    }
    ptrs[num_rows] = _sum;
}

/**
 * Set the indices and data arrays for a CSR matrix from a CSRLike struct
 *
 * @param in_data The data for each row in the matrix
 * @param in_data The columns for the data in each row in the matrix
 * @param ptrs A pointer to the indptr array for a CSR matrix of length num_rows+1
 * @param inds A pointer to the indices array for a CSR matrix of length nnz
 * @param out_data A pointer to the output data array for a CSR matrix of length nnz
 *
 */
template <typename T, typename U, typename V>
void set_csr_data(std::vector<std::vector<T>>& in_data,
                  std::vector<std::vector<U>>& cols,
                  V* __restrict ptrs,
                  V* __restrict inds,
                  T* __restrict out_data)
{
    std::size_t num_rows = in_data.size();
	std::size_t kk;
	std::vector<V> diffs;
	diffs.resize(num_rows);

// #pragma omp parallel for schedule(dynamic)
#pragma omp parallel for simd
	for(kk = 0; kk < num_rows; kk++)
	{
		diffs[kk] = ptrs[kk + 1] - ptrs[kk];
	}

	size_t splits = 5;
	size_t base = num_rows / splits;
	size_t extra = num_rows % splits;
	auto tic = std::chrono::steady_clock::now();

	/// 1st
#pragma omp parallel for simd
	for(kk = 0; kk < base; kk++)
	{
		V start;
		start = ptrs[kk];

		std::copy(cols[kk].data(), cols[kk].data() + diffs[kk], &inds[start]);
		std::copy(in_data[kk].data(), in_data[kk].data() + diffs[kk], &out_data[start]);

	}

#pragma omp parallel for schedule(dynamic)
	for(kk = 0; kk < base; kk++)
	{
		// dealloc after each inner vector is copied into main CSR
		// structure. ``cols[kk]`` and ``in_data[kk]`` are not used
		// after this.
		std::vector<U>().swap(cols[kk]);
		std::vector<T>().swap(in_data[kk]);
	}

	/// 2
#pragma omp parallel for simd
	for(kk = base; kk < (2 * base); kk++)
	{
		V start;
		start = ptrs[kk];

		std::copy(cols[kk].data(), cols[kk].data() + diffs[kk], &inds[start]);
		std::copy(in_data[kk].data(), in_data[kk].data() + diffs[kk], &out_data[start]);

	}

#pragma omp parallel for schedule(dynamic)
	for(kk = base; kk < (2 * base); kk++)
	{
		std::vector<U>().swap(cols[kk]);
		std::vector<T>().swap(in_data[kk]);
	}

	/// 3
#pragma omp parallel for simd
	for(kk = 2 * base; kk < (3 * base); kk++)
	{
		V start;
		start = ptrs[kk];

		std::copy(cols[kk].data(), cols[kk].data() + diffs[kk], &inds[start]);
		std::copy(in_data[kk].data(), in_data[kk].data() + diffs[kk], &out_data[start]);

	}

#pragma omp parallel for schedule(dynamic)
	for(kk = 2 * base; kk < (3 * base); kk++)
	{
		std::vector<U>().swap(cols[kk]);
		std::vector<T>().swap(in_data[kk]);
	}

	/// 4
#pragma omp parallel for simd
	for(kk = 3 * base; kk < (4 * base); kk++)
	{
		V start;
		start = ptrs[kk];

		std::copy(cols[kk].data(), cols[kk].data() + diffs[kk], &inds[start]);
		std::copy(in_data[kk].data(), in_data[kk].data() + diffs[kk], &out_data[start]);

	}

#pragma omp parallel for schedule(dynamic)
	for(kk = 3 * base; kk < (4 * base); kk++)
	{
		std::vector<U>().swap(cols[kk]);
		std::vector<T>().swap(in_data[kk]);
	}

	/// 5
#pragma omp parallel for simd
	for(kk = 4 * base; kk < num_rows; kk++)
	{
		V start;
		start = ptrs[kk];

		std::copy(cols[kk].data(), cols[kk].data() + diffs[kk], &inds[start]);
		std::copy(in_data[kk].data(), in_data[kk].data() + diffs[kk], &out_data[start]);

	}

#pragma omp parallel for schedule(dynamic)
	for(kk = 4 * base; kk < num_rows; kk++)
	{
		std::vector<U>().swap(cols[kk]);
		std::vector<T>().swap(in_data[kk]);
	}

	auto toc = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic);

    std::cout << "Set CSR time: " << duration.count() << " milliseconds" << std::endl;
}


template <typename T, typename U>
void csrlike_spmv(const std::vector<std::vector<T>>& data,
                  const std::vector<std::vector<U>>& cols,
                  const T* __restrict vec,
                  T* __restrict out,
                  U dim)
{
    U row;
#pragma omp parallel for if(dim > 128) schedule(dynamic)
    for(row = 0; row < dim; row++)
    {
        T dot = 0.0;
        std::size_t jj, row_end;
        const T* row_data = data[row].data();
        const U* row_cols = cols[row].data();
        row_end = data[row].size();
        for(jj = 0; jj < row_end; jj++)
        {
            dot += row_data[jj] * vec[row_cols[jj]];
        }
        out[row] += dot;
    }
}

void clear_csrlike_data(std::vector<std::vector<int>>& data_d32_cols,
                        std::vector<std::vector<double>>& data_d32_data,
                        std::vector<std::vector<long long>>& data_d64_cols,
                        std::vector<std::vector<double>>& data_d64_data,
                        std::vector<std::vector<int>>& data_z32_cols,
                        std::vector<std::vector<std::complex<double>>>& data_z32_data,
                        std::vector<std::vector<long long>>& data_z64_cols,
                        std::vector<std::vector<std::complex<double>>>& data_z64_data)
{
    std::vector<std::vector<int>>().swap(data_d32_cols);
    std::vector<std::vector<double>>().swap(data_d32_data);
    std::vector<std::vector<long long>>().swap(data_d64_cols);
    std::vector<std::vector<double>>().swap(data_d64_data);
    std::vector<std::vector<int>>().swap(data_z32_cols);
    std::vector<std::vector<std::complex<double>>>().swap(data_z32_data);
    std::vector<std::vector<long long>>().swap(data_z64_cols);
    std::vector<std::vector<std::complex<double>>>().swap(data_z64_data);
}
