# cython: c_string_type=unicode, c_string_encoding=UTF-8
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "<boost/dynamic_bitset.hpp>" namespace "boost":

    cdef cppclass dynamic_bitset[T]:
        dynamic_bitset()
        dynamic_bitset(size_t)
        dynamic_bitset(size_t, size_t)
        dynamic_bitset(string, size_t, size_t)

        vector[T] m_bits

        bint operator==(const dynamic_bitset&, const dynamic_bitset&)
        bint operator!=(const dynamic_bitset&, const dynamic_bitset&)
        bint operator[](size_t)

        void resize(size_t)
        void set(size_t)
        void reset(size_t)
        void flip(size_t)
        void clear()

        size_t size()
        size_t num_blocks()
        bint test(size_t)
        bint empty()
        bint all()
        bint any()
        bint none()
        size_t count()
        void push_back(bool)

    # Need to specify type here instead of T
    cdef void to_string(dynamic_bitset[size_t]&, string&)

# typedef the kind we want which is always size_t
ctypedef dynamic_bitset[size_t] bitset_t



cdef class Bitset:
    cdef bitset_t bits