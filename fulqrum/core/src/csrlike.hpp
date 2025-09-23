/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
#include <cstring>
#include <vector>
#include <complex>


// Data types for CSR-like matrix structure

typedef struct RowData_Real64{
    std::vector<long long> cols;
    std::vector<double> data;
} RowData_Real64_t;


typedef struct RowData_Real32{
    std::vector<int> cols;
    std::vector<double> data;
} RowData_Real32_t;


typedef struct RowData_Complex64{
    std::vector<long long> cols;
    std::vector<std::complex<double> > data;
} RowData_Complex64_t;


typedef struct RowData_Complex32{
    std::vector<int> cols;
    std::vector<std::complex<double> > data;
} RowData_Complex32_t;


/**
 * Set the row-pointers array data from the data for a CSRLike struct
 *
 * @param row_data The data for each row in the matrix
 * @param ptrs A pointer to the indptr array for a CSR matrix of length num_rows+1
 * 
 */
template <typename T, typename U>
void set_csr_ptr(T& row_data, U * __restrict ptrs)
{
    std::size_t num_rows = row_data.size();
    std::size_t kk;
    std::size_t temp, _sum = 0;
    for(kk=0; kk < num_rows; kk++)
    {
        ptrs[kk] = row_data[kk].cols.size();
        temp = _sum + ptrs[kk];
        ptrs[kk] = _sum;
        _sum = temp;
    }
    ptrs[num_rows] = _sum;
}


/**
 * Set the indices and data arrays for a CSR matrix from a CSRLike struct
 *
 * @param row_data The data for each row in the matrix
 * @param ptrs A pointer to the indptr array for a CSR matrix of length num_rows+1
 * @param inds A pointer to the indices array for a CSR matrix of length nnz
 * @param data A pointer to the data array for a CSR matrix of length nnz
 * 
 */
template <typename T, typename U, typename V>
void set_csr_data(T& row_data, U * __restrict ptrs, U * __restrict inds, V * __restrict data)
{
    std::size_t num_rows = row_data.size();
    std::size_t kk;
    #pragma parallel omp for schedule(dynamic)
    for(kk=0; kk < num_rows; kk++)
    {
        U start, stop;
        start = ptrs[kk];
        stop = ptrs[kk+1];
        std::memcpy(&inds[start], row_data[kk].cols.data(), (stop-start)*sizeof(U));
        std::memcpy(&data[start], row_data[kk].data.data(), (stop-start)*sizeof(V));
    }
}