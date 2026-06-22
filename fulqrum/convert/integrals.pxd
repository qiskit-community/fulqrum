# This code is a part of Fulqrum.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


include "../core/includes/base_header.pxi"

cdef extern from "./src/integrals.hpp":
    
    inline FermionicOperator_t pyscf_integrals_to_fermionic(double * one_body_integrals,
                                                            double * two_body_integrals,
                                                            unsigned int ob_arr_len, unsigned int tb_arr_len,
                                                            double constant, double EQ_TOLERANCE)
