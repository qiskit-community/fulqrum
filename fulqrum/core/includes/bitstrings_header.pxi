# Fulqrum
# Copyright (C) 2024, IBM


cdef extern from "../src/bitstrings.hpp":
    void string_to_vec(const char * in_string, 
                       unsigned char * out_string, 
                       unsigned int num_qubits) nogil


    int bin_width_to_int(const unsigned char * vec,
                         unsigned int num_qubits,
                         unsigned int bin_width) nogil


    size_t col_index(size_t start, size_t stop,
                     const unsigned char * col, 
                     const unsigned char * subspace,
                     size_t num_qubits) nogil


    void get_column_vec(const unsigned char * row,
                        unsigned char * col,
                        unsigned int bit_len,
                        const unsigned int * pos,
                        const unsigned char * val,
                        size_t N) nogil

    const size_t MAX_SIZE_T