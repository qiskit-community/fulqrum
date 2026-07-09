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

"""
Utilities
=========

IO
--

.. autosummary::
   :toctree: ../stubs/

   dict_to_json
   json_to_dict

Spin operators
--------------

.. autosummary::
   :toctree: ../stubs/

   s_z_fermionic_op
   s_plus_fermionic_op
   s_minus_fermionic_op
   s_squared_fermionic_op
   s_squared_qubit_op
   assert_s2_type1
   use_quadratic_penalty
   make_spin_penalized_fermionic_op
   make_spin_penalized_operator
   make_spin_penalized_csr

SQD helpers
-----------

.. autosummary::
   :toctree: ../stubs/

   split_alpha_beta
   build_spin_separated_half_strings

"""

from .matrix import kron_str, qubitoperator_to_matrix
from .io import dict_to_json, json_to_dict
from .spin_operators import (
    s_z_fermionic_op,
    s_plus_fermionic_op,
    s_minus_fermionic_op,
    s_squared_fermionic_op,
    s_squared_qubit_op,
    assert_s2_type1,
    use_quadratic_penalty,
    make_spin_penalized_fermionic_op,
    make_spin_penalized_operator,
    make_spin_penalized_csr,
)

from .sqd_helpers import split_alpha_beta, build_spin_separated_half_strings

__all__ = [
    "kron_str",
    "qubitoperator_to_matrix",
    "dict_to_json",
    "json_to_dict",
    "s_z_fermionic_op",
    "s_plus_fermionic_op",
    "s_minus_fermionic_op",
    "s_squared_fermionic_op",
    "s_squared_qubit_op",
    "assert_s2_type1",
    "use_quadratic_penalty",
    "make_spin_penalized_fermionic_op",
    "make_spin_penalized_operator",
    "make_spin_penalized_csr",
    "split_alpha_beta",
    "build_spin_separated_half_strings",
]
