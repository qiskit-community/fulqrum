# Fulqrum
# Copyright (C) 2024, IBM


cdef extern from "../src/bitstrings.hpp":
    void string_to_vec(const char * in_string, 
                       unsigned char * out_string, 
                       size_t num_qubits) nogil


    int bin_width_to_int(const unsigned char * vec,
                         size_t num_qubits,
                         size_t bin_width) nogil


    size_t col_index(size_t start, size_t stop,
                     const unsigned char * col, 
                     const unsigned char * subspace,
                     size_t num_qubits) nogil


    void get_column_vec(const unsigned char * row,
                        unsigned char * col,
                        size_t bit_len,
                        const size_t * pos,
                        const char * val,
                        size_t N) nogil

    const size_t MAX_SIZE_T