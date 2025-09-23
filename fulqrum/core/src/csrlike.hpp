/**
 * Fulqrum
 * Copyright (C) 2024, IBM
 */
#pragma once
#include <cstdlib>
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
 * @param ptrs A pointer to the indptr array for a CSR matrix of length nnz+1
 * 
 */
template <typename T, typename U>
void set_csr_ptr(T& row_data, U * ptrs)
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