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
void set_csr_ptr(const T& row_data, U * __restrict ptrs)
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
void set_csr_data(const T& row_data, U * __restrict ptrs, U * __restrict inds, V * __restrict data)
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



template <typename T, typename U>
void dcsrlike_spmv(const T& row_data, const double *__restrict vec, double *__restrict out, U dim)
{
    U row;
    #pragma omp parallel for if(dim > 128) schedule(dynamic)
    for (row = 0; row < dim; row++)
    {
        const double * data = row_data[row].data.data();
        const U * cols = row_data[row].cols.data();
        double dot = 0.0;
        std::size_t jj, row_end;
        row_end = row_data[row].cols.size();
        for (jj = 0; jj < row_end; jj++)
        {
            dot += data[jj] * vec[cols[jj]];
        }
        out[row] += dot;
    }
}


template <typename T, typename U>
void zcsrlike_spmv(const T& row_data, const std::complex<double> *__restrict vec, std::complex<double> *__restrict out, U dim)
{
    U row;
    #pragma omp parallel for if(dim > 128) schedule(dynamic)
    for (row = 0; row < dim; row++)
    {
        const std::complex<double> * data = row_data[row].data.data();
        const U * cols = row_data[row].cols.data();
        std::complex<double> dot = 0.0;
        std::size_t jj, row_end;
        row_end = row_data[row].cols.size();
        for (jj = 0; jj < row_end; jj++)
        {
            dot += data[jj] * vec[cols[jj]];
        }
        out[row] += dot;
    }
}