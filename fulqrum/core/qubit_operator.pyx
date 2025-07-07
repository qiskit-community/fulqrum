# Fulqrum
# Copyright (C) 2024, IBM
# cython: c_string_type=unicode, c_string_encoding=UTF-8

cimport cython
from cython.operator cimport dereference
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libcpp.map cimport map
from libcpp.algorithm cimport sort as stdsort
from cython.operator cimport dereference, preincrement
from fulqrum.exceptions import FulqrumError

from collections.abc import Iterable
from pathlib import Path
import numbers
import orjson
import numpy as np
cimport numpy as np

include "includes/base_header.pxi"
include "includes/elements_header.pxi"
include "includes/bitset_utils_header.pxi"
include "includes/operators_header.pxi"
include "includes/grouping_header.pxi"
include "includes/converters.pxi"
include "includes/io.pxi"

cdef const OperatorTerm_t EmptyOperatorTerm


@cython.boundscheck(False)
cdef inline int diagonal_term(OperatorTerm_t * term):
    """Check if term is diagonal in computational basis

    Returns:
        bool: True if diagonal
    """
    return term.offdiag_weight == 0


@cython.boundscheck(False)
cdef void set_proj_indices(OperatorTerm_t& term):
    cdef size_t kk
    cdef unsigned char val
    term.proj_indices.resize(0)
    for kk in range(term.values.size()):
        val = term.values[kk]
        if val == 1 or val == 2:
            term.proj_indices.push_back(term.indices[kk])



cdef class QubitOperator():
    """Operator class for qubit terms consisting of Pauli
    operators,projection operators, and ladder operators
    """
    #QubitOperator oper

    def __cinit__(self, unsigned int num_qubits,
                  object operators=None):
        self.oper.width = num_qubits
        self.oper.sorted = False
        cdef object item
        cdef OperatorTerm_t term
        cdef string op_str
        cdef double complex coeff
        cdef object inds
        cdef size_t kk
        cdef unsigned char op, ind
        if operators is not None:
            for item in operators:
                term = EmptyOperatorTerm
                term.offdiag_weight = 0
                if any(item):
                    if len(item) == 1:
                        term.coeff = item[0]
                    else:
                        op_str = (<string>item[0]).c_str()
                        inds = item[1] if isinstance(item[1], Iterable) else [item[1]] 
                        coeff = item[2]
                        for kk in range(<size_t>len(item[0])):
                            if inds[kk] > (self.oper.width - 1):
                                raise FulqrumError(f'Index {item[1]} is out of range for width={self.oper.width}')
                            if op_str[kk] != 73:
                                term.indices.push_back(inds[kk])
                                ind = STR_TO_IND[op_str[kk]]
                                term.values.push_back(ind)
                        term.coeff = coeff
                else:
                    term.coeff = 1
                sort_term_data(term.indices, term.values)
                set_offdiag_weight(term)
                set_proj_indices(term)
                set_extended_flag(term)
                self.oper.terms.push_back(term)


    def __dealloc__(self):
        # Clear vectors upon deallocation of class
        self.oper.terms = vector[OperatorTerm_t]()
    
    @classmethod
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def from_label(self, string label, double complex coeff = 1.0):
        cdef unsigned int num_qubits = label.size()
        cdef OperatorTerm_t term = EmptyOperatorTerm
        cdef QubitOperator out = QubitOperator(num_qubits)
        cdef unsigned int kk
        cdef const char * labels = label.c_str()
        cdef char s, ind
        for kk in range(num_qubits):
            s = labels[num_qubits-kk-1]
            if s != 73:
                term.indices.push_back(kk)
                ind = STR_TO_IND[s]
                term.values.push_back(ind)
                term.offdiag_weight += (ind > 2)
        term.coeff = coeff
        sort_term_data(term.indices, term.values)
        set_offdiag_weight(term)
        set_proj_indices(term)
        set_extended_flag(term)
        out.oper.terms.push_back(term)
        return out

    @classmethod
    def from_constant(self, unsigned int width, double complex coeff):
        """Generate a constant Hamiltonian term

        Parameters:
            width (unsigned int): Width of operator
            coeff (complex): Operator coefficient
        
        Returns:
            QubitOperator: Constant operator
        """
        cdef QubitOperator out = QubitOperator(width, [()])
        out.oper.terms[0].coeff = coeff
        return out

    def __len__(self):
        return self.oper.terms.size()

    def set_type(self, unsigned int value):
        if (value != 1 and value != 2):
            raise FulqrumError("Type of operator must be '1' or '2'")
        self.oper.type = value
    
    @property
    def type(self):
        """ Type of QubitOperator

        Type `1` is standard Qubit systems, e.g. Paulis and projectors
        Type `2` is for systems derived from Fermionic systems via extended JW

        Returns:
            int: Type of operator
        """
        return self.oper.type

    @property
    def num_terms(self):
        """Return the number of terms in the operator

        Returns:
            int: Number of terms in operator
        """
        return self.oper.terms.size()
    
    @property
    def width(self):
        """Width (number of qubits) of the operator

        Returns:
            int: Width of operator
        """
        return self.oper.width

    @property
    def sorted(self):
        """Is the operator sorted by off-diagonal structure

        Returns:
            bool: Operator is sorted
        """
        return self.oper.sorted

    @property
    def weight_sorted(self):
        """Is the operator sorted by full operator weight

        Returns:
            bool: Operator is sorted by full weight
        """
        return self.oper.weight_sorted

    @property
    def off_weight_sorted(self):
        """Is the operator sorted by off-diagonal weight

        Returns:
            bool: Operator is sorted by off-diagonal weight
        """
        return self.oper.off_weight_sorted
    
    @property
    def num_groups(self):
        """Number of off-diagonal groupings

        Returns:
            int : Number of groups in operator
        """
        if self.num_terms == 0:
            return 0
        if not self.oper.sorted:
            self.offdiag_term_grouping()
        return (self.oper.terms[self.oper.terms.size()-1].group - self.oper.terms[0].group) + 1
    
    def copy(self):
        """Copy QubitOperator

        Returns:
            QubitOperator
        """
        cdef QubitOperator out = QubitOperator(self.width)
        out.oper.terms = self.oper.terms
        out.oper.sorted = self.oper.sorted
        out.oper.weight_sorted = self.oper.weight_sorted
        out.oper.off_weight_sorted = self.oper.off_weight_sorted
        return out
    
    @cython.boundscheck(False)
    def split_diagonal(self):
        """Spit an operator into diagonal and non-diagonal components

        Returns:
            tuple: QubitOperators for diagonal and non-diagonal parts
        """
        cdef size_t kk
        cdef QubitOperator diag = QubitOperator(self.oper.width)
        cdef QubitOperator offdiag = QubitOperator(self.oper.width)
        cdef OperatorTerm_t term
        for kk in range(self.oper.terms.size()):
            term = self.oper.terms[kk]
            if diagonal_term(&term):
                diag.oper.terms.push_back(term)
            else:
                offdiag.oper.terms.push_back(term)
        # set sorted flag
        diag.oper.sorted = self.oper.sorted
        offdiag.oper.sorted = self.oper.sorted
        return diag, offdiag

    @property
    def coeff(self):
        """Return the coeff for a single term or empty operator
        """
        cdef size_t kk, jj
        cdef OperatorTerm_t * term
        cdef list out = []
        if self.oper.terms.size() > 2:
            raise FulqrumError('Can only grab coeff from operators with < 2 terms')
        elif self.oper.terms.size() == 0:
            return 0+0j
        return self.oper.terms[0].coeff

    @property
    def operators(self):
        """Return the operators for a single term or empty operator

        Returns:
            list or None : List of operator index tuples, if any, else None

        Notes:
            This returns a list of tuples to allow for multiple indices in
            Fermionic operators
        """
        cdef size_t kk, jj
        cdef OperatorTerm_t * term
        cdef list out = []
        if self.num_terms > 1:
            raise FulqrumError('Can only grab operators from operators with < 2 terms')
        elif self.num_terms == 0:
            return None
        else:
            for kk in range(self.oper.terms.size()):
                term = &self.oper.terms[kk]
                for jj in range(term.indices.size()):
                    out.append((IND_TO_STR[term.values[jj]], term.indices[jj]))
            return out

    @property
    def proj_indices(self):
        """Return the projector indices for a single term or empty operator
        """
        if self.oper.terms.size() > 2:
            raise FulqrumError('Can only grab projector indices from operators with < 2 terms')
        cdef size_t kk
        cdef unsigned int[::1] out
        if self.oper.terms.size() == 0:
            return np.array([], dtype=np.uint32)
        out = np.zeros(self.oper.terms[0].proj_indices.size(), dtype=np.uint32)
        for kk in range(self.oper.terms[0].proj_indices.size()):
            out[kk] = self.oper.terms[0].proj_indices[kk]
        return np.asarray(out)

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
        cdef QubitOperator out = QubitOperator(self.oper.width)
        if isinstance(key, numbers.Integral):
            if key < 0:
                 key = self.oper.terms.size() + key 
            kk = <size_t>key
            if kk > self.oper.terms.size() - 1:
                raise FulqrumError(f"Index {kk} is out of range for operator"
                                   f" of length {self.oper.terms.size()}")
            out.oper.terms.push_back(self.oper.terms[kk])
        elif isinstance(key, slice):
            for kk in range(*key.indices(self.oper.terms.size())):
                out.oper.terms.push_back(self.oper.terms[kk])
        elif isinstance(key, (list, tuple)):
            for idx in key:
                kk = <size_t>idx
                out.oper.terms.push_back(self.oper.terms[kk])
        else:
            raise FulqrumError(f"Cannot get operator terms using {type(key)}")
        return out

    def __iadd__(self, QubitOperator other):
        """Inplace addition of QubitOperators
        """
        self.append(other)
        return self
    
    @cython.boundscheck(False)
    def __imul__(self, double complex other):
        """Inplace multiplication of QubitOperators
        """
        cdef size_t kk
        for kk in range(self.oper.terms.size()):
            self.oper.terms[kk].coeff *= other
        return self

    @cython.boundscheck(False)
    def __add__(self, QubitOperator other):
        """Addition of QubitOperators with copy

        Returns:
            QubitOperator
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.oper.width)
        out.oper.terms = self.oper.terms
        for kk in range(other.oper.terms.size()):
            out.oper.terms.push_back(other.oper.terms[kk])
        return out

    @cython.boundscheck(False)
    def __sub__(self, QubitOperator other):
        """Subtraction of QubitOperators with copy

        Returns:
            QubitOperator
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.oper.width)
        cdef OperatorTerm_t term
        out.oper.terms = self.oper.terms
        for kk in range(other.oper.terms.size()):
            term = other.oper.terms[kk]
            term.coeff *= -1
            out.oper.terms.push_back(term)
        return out
    
    @cython.boundscheck(False)
    def __mul__(self, double complex other):
        """Multiplication of QubitOperators on left with copy

        Returns:
            QubitOperator
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.oper.width)
        out.oper.terms = self.oper.terms
        for kk in range(out.oper.terms.size()):
            out.oper.terms[kk].coeff *= other
        return out

    @cython.boundscheck(False)
    def __rmul__(self, double complex other):
        """Multiplication of QubitOperators on right with copy

        Returns:
            QubitOperator
        """
        cdef size_t kk
        cdef QubitOperator out = QubitOperator(self.oper.width)
        out.oper.terms = self.oper.terms
        for kk in range(out.oper.terms.size()):
            out.oper.terms[kk].coeff *= other
        return out

    @cython.boundscheck(False)
    def __repr__(self):
        cdef size_t idx 
        cdef list out = []
        cdef str temp_str
        cdef OperatorTerm_t term
        cdef size_t kk
        cdef size_t num_terms = self.oper.terms.size()
        cdef size_t total_terms = num_terms
        cdef int too_many_terms = 0
        if num_terms > 100:
            too_many_terms = 1
            num_terms = 100
        for idx in range(num_terms):
            temp_str = ''
            term = self.oper.terms[idx]
            for kk in range(term.indices.size()):
                if kk:
                    temp_str += ' '
                temp_str += IND_TO_STR[term.values[kk]] + ':'
                temp_str += str(term.indices[kk])
            out.append((temp_str, term.coeff))

        out_strs = ', '.join(str(kk) for kk in out)
        if too_many_terms:
            out_strs += f' + {total_terms-100} more terms'
        add_str = ''
        if num_terms == 1:
            add_str = f", extended={self.oper.terms[0].extended}, group={self.oper.terms[0].group}"
        return f"<QubitOperator[{out_strs}], width={self.oper.width}{add_str}>"
    
    @cython.boundscheck(False)
    cpdef void append(self, QubitOperator other):
        cdef size_t kk
        if self.oper.width != other.oper.width:
            raise FulqrumError('Appending number of qubits does not match current number')
        for kk in range(other.oper.terms.size()):
            self.oper.terms.push_back(other.oper.terms[kk])
        self.oper.sorted = 0

    @cython.boundscheck(False)
    def weights(self):
        """Weight of each term in the operator

        Returns:
            ndarray: Array of operator weights
        """
        cdef unsigned int[::1] out = np.zeros(self.oper.terms.size(), dtype=np.uint32)
        cdef size_t kk
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].values.size()
        return np.asarray(out)

    @cython.boundscheck(False)
    def offdiag_weights(self):
        """Off-diagonal weight of each term in the operator

        Returns:
            ndarray: Array of operator off-diagonal weights
        """
        cdef unsigned int[::1] out = np.zeros(self.oper.terms.size(), dtype=np.uint32)
        cdef size_t kk
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].offdiag_weight
        return np.asarray(out)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef int is_diagonal(self):
        """Check if operator is diagonal in computational basis

        Returns:
            int: True if diagonal, False otherwise
        """
        cdef size_t kk, jj
        for kk in range(self.oper.terms.size()):
            for jj in range(self.oper.terms[kk].values.size()):
                if self.oper.terms[kk].values[jj] > 2:
                    return 0
        return 1

    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def add_operator(self, unsigned int qubit, str operator, bool overwrite=False):
        cdef size_t kk
        cdef OperatorTerm_t temp_term 
        if self.oper.terms.size() == 0:
            temp_term = EmptyOperatorTerm
            temp_term.coeff = 1.0
            self.oper.terms.push_back(temp_term)
        if self.oper.terms.size() > 1:
            raise FulqrumError('Can only add operators to single-term QubitOperators.')
        if qubit >= self.oper.width:
            raise FulqrumError(f"qubit number {qubit} out of range")
        # Check if element already exists if overwrite=False
        if not overwrite:
            for kk in range(self.oper.terms[0].indices.size()):
                if self.oper.terms[0].indices[kk] == qubit:
                    raise FulqrumError(f"Operator {IND_TO_STR[self.oper.terms[0].indices[kk]]} already exists at qubit {qubit}")
        if operator != 'I':
            self.oper.terms[0].indices.push_back(qubit)
            self.oper.terms[0].values.push_back(STR_TO_IND[operator])
        self.oper.sorted = 0

    @cython.boundscheck(False)
    cpdef double complex sum_identity_terms(self):
        """Sum of identity terms coefficients.

        Returns:
            double complex: Sum of identities
        """
        cdef size_t kk
        cdef OperatorTerm_t * term_ptr
        cdef double complex out = 0
        for kk in range(self.oper.terms.size()):
            term_ptr = &self.oper.terms[kk]
            if term_ptr.indices.size() == 0:
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
        cdef OperatorTerm_t * term_ptr
        cdef double complex val = 0
        cdef QubitOperator out = QubitOperator(self.oper.width)
        for kk in range(self.oper.terms.size()):
            term_ptr = &self.oper.terms[kk]
            if term_ptr.indices.size() != 0:
                out.oper.terms.push_back(dereference(term_ptr))
            else:
                val += term_ptr.coeff
        if return_value:
            return out, val
        return out
    
    @cython.boundscheck(False)
    def matrix_element(self, object row, object col):
        """Compute matrix element value at given row and column

        Parameters:
            row (str or int): Row index
            col (str or int): Column index

        Returns:
            complex: Element value at H[row, col]
        """
        cdef string row_str, col_str
        if isinstance(row, numbers.Integral):
            row_str = bin(row)[2:].zfill(self.oper.width)
        elif isinstance(row, str):
            row_str = row
        else:
            raise Exception('bad row input')
        if isinstance(col, numbers.Integral):
            col_str = bin(col)[2:].zfill(self.oper.width)
        elif isinstance(col, str):
            col_str = col
        else:
            raise Exception('bad col input')
        if row_str.size() != col_str.size():
            raise Exception('String lengths differ')
        cdef unsigned int bit_len = row_str.size()

        cdef bitset_t row_vec, col_vec, nonzero_vec

        # convert strings to bit arrays
        row_vec = bitset_t(row_str, 0, row_str.size())
        col_vec = bitset_t(col_str, 0, col_str.size())

        if bit_len != self.oper.width:
            raise Exception('Operator width does not match string length')
        cdef size_t num_terms = self.oper.terms.size()
        cdef size_t kk, weight
        cdef double complex out = 0.0
        cdef double complex temp
        cdef OperatorTerm_t * term
        for kk in range(num_terms):
            term = &self.oper.terms[kk]
            weight = term.indices.size()
            nonzero_vec = row_vec #copy row vec into nonzero vec
            get_column_bitset(nonzero_vec, &term.indices[0], &term.values[0], weight)
            # Input col string matches that of nonzero column
            if col_vec == nonzero_vec:
                accum_element(row_vec, nonzero_vec,
                              &term.indices[0], &term.values[0], term.coeff, weight, out)
        return out


    def groups(self):
        """Off-diagonal group structure of terms in operator

        Returns:
            ndarray: Array of ints indicating group of each term
        """
        if not self.oper.sorted:
            self.offdiag_term_grouping()
        cdef size_t kk
        cdef int[::1] out = np.zeros(self.oper.terms.size(), dtype=np.int32)
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].group
        return np.asarray(out)

    @cython.boundscheck(False)
    def group_ptrs(self):
        """Get pointers to start and stop indices for off-diagonal grouping

        Returns:
            ndarray: Array of ints giving pointers to group starts and stops
        """
        if self.num_terms == 0:
            return np.zeros(0, dtype=np.uintp)
        if not self.oper.sorted:
            self.offdiag_term_grouping()
        cdef OperatorTerm_t * terms = &self.oper.terms[0]
        cdef int idx = 0
        cdef size_t kk
        cdef size_t num_groups = 1
        cdef int val = terms[0].group
        for kk in range(self.oper.terms.size()):
            if terms[kk].group > val:
                num_groups += 1
                val = terms[kk].group
        cdef size_t[::1] ptrs = np.zeros(num_groups+1, dtype=np.uintp)
        val = terms[0].group
        for kk in range(self.oper.terms.size()):
            if terms[kk].group > val:
                ptrs[idx+1] = kk
                idx += 1
                val += 1
        ptrs[idx+1] = self.oper.terms.size()
        return np.asarray(ptrs)
    
    def extended(self):
        """Extended element flag for each term

        Returns:
            ndarray: Array of ints indicating if terms are extended or not
        """
        cdef size_t kk
        cdef int[::1] out = np.zeros(self.oper.terms.size(), dtype=np.int32)
        for kk in range(self.oper.terms.size()):
            out[kk] = self.oper.terms[kk].extended
        return np.asarray(out)

    def offdiag_term_grouping(self):
        """Inplace sorting of operator terms according to off-diagonal
        structure.
        """
        offdiag_weight_sort(self.oper)
        offdiag_term_sort(self.oper)

    def offdiag_weight_sort(self):
        """In-place sort terms by their off-diagonal weight
        """
        offdiag_weight_sort(self.oper)

    def weight_sort(self):
        """In-place sort terms by their standard weight
        """
        weight_sort(self.oper)

    def offdiag_weight_ptrs(self):
        """Off-diagonal weight pointers for the operator

        Returns:
            ndarray: Array of off-diagonal weight pointers
        """
        if not self.oper.off_weight_sorted:
            self.offdiag_weight_sort()
        cdef vector[size_t] vec
        set_offdiag_weight_ptrs(self.oper.terms, vec)
        cdef size_t[::1] out = np.empty(vec.size(), dtype=np.uintp)
        cdef size_t kk
        for kk in range(vec.size()):
            out[kk] = vec[kk]
        return np.asarray(out)

    def max_offdiag_ptr_size(self):
        """Maximum number of elements in an off-diagonal pointer term

        Returns:
            int: Number of terms
        """
        temp = self.offdiag_weight_ptrs()
        if temp.shape[0] == 0:
            return 0
        cdef size_t[::1] out = temp
        return max_offdiag_ptr_size(&out[0], out.shape[0])
    
    def combine_repeated_terms(self, double atol=1e-12):
        """Combine repeated terms that represent same
        operators, dropping terms smaller than requested tolerance.

        Parameters:
            atol (double): Tolerance for dropping terms, default=1e-12

        Returns:
            QubitOperator: Operator with repeat terms combined
        """
        cdef QubitOperator out = QubitOperator(self.width)
        cdef size_t num_terms = self.oper.terms.size()
        cdef unsigned int[::1] touched = np.zeros(num_terms, dtype=np.uint32)
        if not self.oper.weight_sorted:
            self.weight_sort()
        combine_qubit_terms(self.oper.terms, out.oper.terms,
                            &touched[0], atol)
        return out

    def ladder_ints(self, unsigned int ladder_width=3):
        """Compute the ladder operator integer for each term

        If no ladder ops present then default int is max(uint32)

        Parameters:
            ladder_width (int): Number of ladder terms to consider, default = 3

        Returns:
            ndarray: Array of uint32 integers 
        """
        cdef unsigned int[::1] out = np.zeros(self.oper.terms.size(), dtype=np.uint32)
        cdef size_t kk
        for kk in range(self.oper.terms.size()):
            out[kk] = term_ladder_int(self.oper.terms[kk], ladder_width)
        return np.asarray(out)
    
    def group_ladder_indices(self, unsigned int ladder_width=3):
        if not self.oper.sorted:
            raise FulqrumError("Operator must be group sorted first")
        if not self.oper.type == 2:
            raise FulqrumError("Operator must be type=2")
        cdef size_t[::1] group_ptrs = self.group_ptrs()
        cdef size_t kk, jj
        cdef list out = []
        cdef unsigned int[::1] temp_inds
        for kk in range(group_ptrs.shape[0]-1):
            if not self.oper.terms[group_ptrs[kk]].group: # diagonal group so return nothing
                temp_inds = np.zeros(0, dtype=np.uint32)
            else:
                temp_inds = np.zeros(min(self.oper.terms[group_ptrs[kk]].offdiag_weight, ladder_width), dtype=np.uint32)
                compute_term_ladder_inds(self.oper.terms[group_ptrs[kk]], &temp_inds[0], ladder_width)
            out.append(np.asarray(temp_inds))
        return out

    def group_term_sort_by_ladder_int(self, unsigned int ladder_width=3):
        if not self.oper.type == 2:
            raise FulqrumError("Operator must be type=2")
        if not self.oper.sorted:
            raise FulqrumError('Operator must be group sorted')
        cdef size_t[::1] group_ptrs = self.group_ptrs()
        sort_groups_by_ladder_int(self.oper, &group_ptrs[0], group_ptrs.shape[0]-1, ladder_width)

    def group_ladder_bin_starts(self, unsigned int ladder_width=3):
        if not self.oper.type == 2:
            raise FulqrumError("Operator must be type=2")
        cdef size_t[::1] group_ptrs = self.group_ptrs()
        cdef unsigned int num_bins = 2**ladder_width
        cdef unsigned int ptr_size = num_bins + 1
        cdef unsigned int num_groups = group_ptrs.shape[0] - 1
        cdef unsigned int[::1] group_counts
        cdef unsigned int[::1] group_ranges
        group_counts = np.zeros(num_bins*num_groups, dtype=np.uint32)
        group_ranges = np.zeros(ptr_size*num_groups, dtype=np.uint32)
        ladder_bin_starts(&self.oper.terms[0], &group_ptrs[0], &group_counts[0], &group_ranges[0],
                        num_groups, num_bins, ladder_width)
        
        return np.asarray(group_ranges)

    @cython.boundscheck(False)
    def to_dict(self):
        """Dictionary represenation of QubitOperator
        
        Returns:
            dict: Dictionary representation of QubitOperator
        """
        cdef dict out = {'operator-type': 'qubit',
                        'format-version': FORMAT_VERSION,
                        'fulqrum-version': fversion,
                        'width': self.width
                        }
        cdef OperatorTerm_t * term
        cdef size_t kk, jj
        cdef list terms = []
        cdef list temp_inds
        cdef str temp_vals
        for kk in range(self.oper.terms.size()):
            term = &self.oper.terms[kk]
            temp_inds = []
            temp_vals = ''
            for jj in range(term.indices.size()):
                temp_inds.append(term.indices[jj])
                temp_vals += IND_TO_STR[term.values[jj]]
            terms.append([temp_vals, temp_inds, (term.coeff.real, term.coeff.imag)])
        out['terms'] = terms
        return out
    

    @classmethod
    def from_dict(self, dict dic):
        """QubitOperator from dictionary

        Parameters:
            dic(dict): Dictionary representation of operator
        
        Returns:
            QubitOperator
        """
        if dic['operator-type'] != 'qubit':
            raise FulqrumError("Dictionary operator-type is not 'qubit'")
        cdef size_t width = dic['width']
        cdef QubitOperator out = QubitOperator(width)
        for term in dic['terms']:
            out += QubitOperator(width, [(term[0], term[1], complex(*term[2]))])
        return out
    

    def to_json(self, filename, overwrite=False):
        """Save operator to a JSON file.

        Parameters:
            filename (str): File to store to
            overwrite (bool): Overwrite file if it exits, default=False
        """
        file = Path(filename)
        if file.is_file() and not overwrite:
            raise Exception("File already exists, set overwrite=True")
        dic = self.to_dict()
        with open(filename, "wb") as fd:
            fd.write(orjson.dumps(dic))


    @classmethod
    def from_json(self, filename):
        """Load operator from a JSON file.

        Parameters:
            filename (str): File to load from

        Returns:
            QubitOperator
        """
        with open(filename, "r", encoding="utf-8") as fd:
            dic = orjson.loads(fd.read())
        out = QubitOperator.from_dict(dic)
        return out
