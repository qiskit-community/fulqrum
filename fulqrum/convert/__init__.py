# Fulqrum
# Copyright (C) 2024, IBM

"""Conversion utilities"""


from .openfermion import (
    openfermion_fermi_op_to_fulqrum,
    openfermion_qubit_op_to_fulqrum,
)
from .qiskit import (
    sparsepauli_to_fulqrum,
    qiskit_nature_fermi_op_to_fulqrum,
)
from .integrals import (
    integrals_to_fq_fermionic_op,
    fcidump_to_fq_fermionic_op
)
