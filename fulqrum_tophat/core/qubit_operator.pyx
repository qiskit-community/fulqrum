# Fulqrum - Top Hat
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
from cython.operator cimport dereference
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.algorithm cimport sort as stdsort
from cython.operator cimport dereference, preincrement
from fulqrum_tophat.exceptions import FulqrumError
from fulqrum_tophat.core.base cimport OperatorTerm, size_uchar_pair, diagonal_term

import numbers
import numpy as np
cimport numpy as np


cdef char[6] diag_oper_elems = [1, -1,   # Z
                                1, 0,    # 0
                                0, 1     # 1
                                ]

cdef unordered_map[string, size_t] STR_TO_IND = {'Z': 0, '0': 1, '1': 2, 'X': 3, 'Y': 4,
                                                 'D': 5, 'U': 6}

cdef unordered_map[size_t, string] IND_TO_STR = {0: 'Z', 1: '0', 2: '1', 3: 'X', 4: 'Y',
                                                 5: 'D', 6: 'U'}


cdef const OperatorTerm EmptyOperatorTerm


cdef inline unsigned int int_min(unsigned int x, unsigned int y) nogil:
    if x < y:
        return x
    return y


cdef class QubitOperator():
    """Operator class for qubit terms consisting of Pauli
    operators,projection operators, and ladder operators
    """
    #public size_t width
    #public size_t num_qubits
    #vector[QubitTerm] terms
    #public bool sorted
    
    def __cinit__(self, size_t num_qubits,
                  object operators=None,
                  double complex coeff=1.0):
        self.width = num_qubits
        self.num_qubits = num_qubits
        self.sorted = False
        cdef object item
        cdef OperatorTerm term
        if operators is not None:
            term = EmptyOperatorTerm
            term.coeff = coeff
            for item in operators:
                if any(item):
                    if item[0] != 'I':
                        if item[1] > self.width - 1:
                            raise FulqrumError(f'Index {item[1]} is out of range for width={self.width}')
                        term.operators.push_back(size_uchar_pair(item[1], STR_TO_IND[item[0]]))
            self.terms.push_back(term)

    def __len__(self):
        return self.terms.size()

    @property
    def num_terms(self):
        """Return the number of terms in the operator

        Returns:
            int: Number of terms in operator
        """
        return self.terms.size()
    
    @cython.boundscheck(False)
    def split_diagonal(self):
        """Spit an operator into diagonal and non-diagonal components

        Returns:
            tuple: QubitOperators for diagonal and non-diagonal parts
        """
        cdef size_t kk
        cdef QubitOperator diag = QubitOperator(self.width)
        cdef QubitOperator offdiag = QubitOperator(self.width)
        cdef OperatorTerm term
        for kk in range(self.terms.size()):
            term = self.terms[kk]
            if diagonal_term(&term):
                diag.terms.push_back(term)
            else:
                offdiag.terms.push_back(term)
        return diag, offdiag

    @property
    def coeff(self):
        """Return the coeff for a single term or empty operator
        """
        cdef size_t kk, jj
        cdef OperatorTerm * term
        cdef list out = []
        if self.num_terms > 2:
            raise FulqrumError('Can only grab coeff from operators with < 2 terms')
        elif self.num_terms == 0:
            return 0+0j
        return self.terms[0].coeff

    @property
    def operators(self):
        """Return the operators for a single term or empty operator

        Returns:
            list or None : List of operator index tuples, if any, else None
        """
        cdef size_t kk, jj
        cdef OperatorTerm * term
        cdef list out = []
        if self.num_terms > 1:
            raise FulqrumError('Can only grab operators from operators with < 2 terms')
        elif self.num_terms == 0:
            return None
        else:
            for kk in range(self.terms.size()):
                term = &self.terms[kk]
                for jj in range(term.operators.size()):
                    out.append((IND_TO_STR[term.operators[jj].second], 
                                term.operators[jj].first))
            return out

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, key):
        """Impliments indexing and slicing of terms

        Parameters:
            key (integral or slice or list or tuple): Indices

        Returns:
            QubitOperator: Indexed terms
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.num_qubits)
        if isinstance(key, numbers.Integral):
            kk = <size_t>key
            if kk > self.terms.size() - 1:
                raise FulqrumError(f"Index {kk} is out of range for operator"
                                   f" of length {self.terms.size()}")
            out.terms.push_back(self.terms[kk])
        elif isinstance(key, slice):
            for kk in range(*key.indices(self.terms.size())):
                out.terms.push_back(self.terms[kk])
        elif isinstance(key, (list, tuple)):
            for idx in key:
                kk = <size_t>idx
                out.terms.push_back(self.terms[kk])
        else:
            raise FulqrumError(f"Cannot get operator terms using {type(key)}")
        return out

    def __iadd__(self, QubitOperator other):
        """Inplace addition of QubitOperators
        """
        self.append(other)
        return self
    
    def __imul__(self, double complex other):
        """Inplace multiplication of QubitOperators
        """
        cdef size_t kk
        for kk in range(self.terms.size()):
            self.terms[kk].coeff *= other
        return self

    def __add__(self, QubitOperator other):
        """Addition of QubitOperators with copy
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.width)
        out.terms = self.terms
        for kk in range(other.terms.size()):
            out.terms.push_back(other.terms[kk])
        return out

    def __mul__(self, double complex other):
        """Multiplication of QubitOperators on left with copy
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.width)
        out.terms = self.terms
        for kk in range(out.terms.size()):
            out.terms[kk].coeff *= other
        return out

    def __rmul__(self, double complex other):
        """Multiplication of QubitOperators on right with copy
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.width)
        out.terms = self.terms
        for kk in range(out.terms.size()):
            out.terms[kk].coeff *= other
        return out

    @cython.boundscheck(False)
    def __repr__(self):
        cdef unsigned int idx 
        cdef list out = []
        cdef str temp_str
        cdef OperatorTerm term
        cdef size_t kk
        for idx in range(self.terms.size()):
            temp_str = ''
            term = self.terms[idx]
            for kk in range(term.operators.size()):
                if kk:
                    temp_str += ' '
                temp_str += IND_TO_STR[term.operators[kk].second]
                temp_str += str(term.operators[kk].first)
            out.append([temp_str, term.coeff])

        out_strs = ', '.join(str(kk) for kk in out)
        return f"QubitOperator<[{out_strs}], num_qubits={self.num_qubits}>"
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void append(self, QubitOperator other):
        cdef size_t kk
        if self.num_qubits != other.num_qubits:
            raise FulqrumError('Appending number of qubits does not match current number')
        for kk in range(other.terms.size()):
            self.terms.push_back(other.terms[kk])

    @cython.boundscheck(False)
    def weight(self):
        """Weight of each term in the operator

        Returns:
            ndarray: Array of operator weights
        """
        cdef size_t[::1] out = np.zeros(self.terms.size(), dtype=np.uint64)
        cdef size_t kk
        for kk in range(self.terms.size()):
            out[kk] = self.terms[kk].operators.size()
        return np.asarray(out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef bool is_diagonal(self):
        """Check if operator is diagonal in computational basis

        Returns:
            bool: True if diagonal, False otherwise
        """
        cdef size_t kk, jj
        for kk in range(self.terms.size()):
            for jj in range(self.terms[kk].operators.size()):
                if self.terms[kk].operators[jj].second > 2:
                    return False
        return True
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def expval(self, object dist):
        """Expectation value of operator with respect to a given
        input distribution of bit-strings.

        Parameters:
            dist (dict): Input dictionary of bit-strings and corresponding
                         probabilities or counts
        Returns:
            double complex: Expectation value

        Notes:
            The return value is complex as the operator need not be
            Hermitian
        """
        cdef double shots = 0
        cdef unordered_map[string, double] dist_map = dist
        cdef unordered_map[string, double].iterator it
        cdef unordered_map[string, double].iterator end = dist_map.end()
        cdef double temp_sum
        cdef double complex out = 0
        cdef size_t kk, jj,
        cdef OperatorTerm * term
        cdef const char * temp_str
        cdef char prod

        with nogil:
            #compute shots
            it = dist_map.begin()
            while it != end:
                shots += dereference(it).second
                preincrement(it)
            
            for kk in range(self.terms.size()):
                it = dist_map.begin()
                term = &self.terms[kk]
                temp_sum = 0
                while it != end:
                    temp_str = dereference(it).first.c_str()
                    prod = 1
                    for jj in range(term.operators.size()):
                        prod *= diag_oper_elems[2*term.operators[jj].second+(<size_t>temp_str[self.num_qubits-term.operators[jj].first-1]-48)]
                    temp_sum += prod * dereference(it).second
                    preincrement(it)
                out += term.coeff * temp_sum / shots
        return out
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_operator(self, size_t qubit, str operator, bool overwrite=False):
        cdef size_t kk
        cdef OperatorTerm temp_term 
        if self.terms.size() == 0:
            temp_term = EmptyOperatorTerm
            temp_term.coeff = 1.0
            self.terms.push_back(temp_term)
        if self.terms.size() > 1:
            raise FulqrumError('Can only add operators to single-term QubitOperators.')
        if qubit >= self.num_qubits:
            raise FulqrumError(f"qubit number {qubit} out of range")
        # Check if element already exists if overwrite=False
        if not overwrite:
            for kk in range(self.terms[0].operators.size()):
                if self.terms[0].operators[kk].first == qubit:
                    raise FulqrumError(f"Operator {IND_TO_STR[self.terms[0].operators[kk].second]} already exists at qubit {qubit}")
        if operator != 'I':
            self.terms[0].operators.push_back(size_uchar_pair(qubit, STR_TO_IND[operator]))
        self.sorted = False

    @cython.boundscheck(False)
    cpdef double complex identity_terms_sum(self):
        """Sum of identity terms coefficients.

        Returns:
            double complex: Sum of identities
        """
        cdef size_t kk
        cdef OperatorTerm * term_ptr
        cdef double complex out = 0
        for kk in range(self.terms.size()):
            term_ptr = &self.terms[kk]
            if term_ptr.operators.size() == 0:
                out += term_ptr.coeff
        return out

    @cython.boundscheck(False)
    def remove_identity_terms(self, bool return_value=False):
        """Remove identity terms from operator, optionally
        returning the sum of the coefficients

        Parameters:
            return_value (bool): Return the sum of identity coefficients
        
        Returns:
            QubitOperator: Operator with no identity terms
            complex: Sum of identity coefficients, if `return_value=True`
        """
        cdef size_t kk
        cdef OperatorTerm * term_ptr
        cdef double complex val = 0
        cdef QubitOperator out = QubitOperator(self.width)
        for kk in range(self.terms.size()):
            term_ptr = &self.terms[kk]
            if term_ptr.operators.size() != 0:
                out.terms.push_back(dereference(term_ptr))
            else:
                val += term_ptr.coeff
        if return_value:
            return out, val
        return out

    @cython.boundscheck(False)
    cpdef void sort_indices(self):
        """Inplace sorting of term indices
        """
        cdef size_t kk
        cdef OperatorTerm * term_ptr
        for kk in range(self.terms.size()):
            term_ptr = &self.terms[kk]
            stdsort(term_ptr.operators.begin(), term_ptr.operators.end())

    @cython.boundscheck(False)
    cpdef QubitOperator sorted_indices(self):
        """Return a copy of the operator with sorted term indices

        Returns:
            QubitOperator: Operator with sorted term indices
        """
        cdef size_t kk
        cdef OperatorTerm term
        cdef QubitOperator out = QubitOperator(self.width)
        for kk in range(self.terms.size()):
            #make a copy of the term
            term = self.terms[kk]
            stdsort(term.operators.begin(), term.operators.end())
            out.terms.push_back(term)
        return out
        
    @classmethod
    def from_dicts(self, unsigned int num_qubits, object terms=[]):
        cdef double complex coeff
        cdef list item
        cdef unsigned int key
        cdef dict dic
        cdef OperatorTerm term
        cdef QubitOperator out = QubitOperator(num_qubits)
        for item in terms: 
            dic = item[0]
            coeff = item[1]
            term = EmptyOperatorTerm
            term.coeff = coeff
            for key, val in dic.items():
                if val != 'I':
                    term.operators.push_back(size_uchar_pair(key, STR_TO_IND[val]))
            out.terms.push_back(term)
        return out
    
    
    @classmethod
    def from_lists(self, unsigned int num_qubits, object terms=[]):
        cdef object item
        cdef object indices, opers
        cdef size_t ind
        cdef OperatorTerm term
        cdef QubitOperator out = QubitOperator(num_qubits)
        for item in terms:
            term = EmptyOperatorTerm
            term.coeff = 1.0
            if len(item):
                tuples = item[0]
                if len(item) > 1:
                    term.coeff = item[1] 
                for tup in tuples:
                    if tup[0] != 'I':
                        term.operators.push_back(size_uchar_pair(tup[1], STR_TO_IND[tup[0]]))
            out.terms.push_back(term)
        return out
    
    
    @classmethod
    def from_strings(self, unsigned int num_qubits, object terms=[]):
        cdef double complex coeff
        cdef list item
        cdef size_t idx
        cdef str string, s
        cdef unsigned int key
        cdef OperatorTerm term
        cdef QubitOperator out = QubitOperator(num_qubits)
        for item in terms:
            string = item[0]
            coeff = item[1] 
            term = EmptyOperatorTerm
            term.coeff = coeff
            idx = 0
            for s in string[::-1]:
                if s != 'I':
                    term.operators.push_back(size_uchar_pair(idx, STR_TO_IND[s]))
                idx += 1
            out.terms.push_back(term)
        return out
