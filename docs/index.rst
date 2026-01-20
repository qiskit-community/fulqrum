#######
Fulqrum
#######

.. figure:: images/fulqrum_logo.png
    :align: center


Fulqrum is a set of tools for enabling the solution to general large-scale Hamiltonian subspace eigenproblems over subspaces defined by samples returned from executing circuits on quantum devices.  Such methods work if the target eigenstate has compact support over the underlying full Hilbert space; the target state has nonzero components in a polynomial number of bit-strings.
Fulqrum works over an extended alphabet of operators that includes projector and ladder operators in addition to the usually Pauli set, and because of this, Fulqrum is able to solve both fermionic and spin systems using the same code base.  Moreover, there is no intrinsic limit on the number of qubits that Fulqrum can handle, allowing users to go up to the full-scale of current and future quantum systems.


.. toctree::
    :maxdepth: 1
    :hidden:

    self
    Installation <install>
    Citing Fulqrum <citing>

.. toctree::
    :maxdepth: 1
    :caption: User guide
    :hidden:
    
    Getting started  <started>
    Using QubitOperators  <qubit_operators.ipynb>
    Using FermionicOperators  <fermi_operators.ipynb>
    Defining Subspaces <subspaces.ipynb>
    Operator IO <operator_io.ipynb>

.. toctree::
    :maxdepth: 2
    :caption: Tutorials
    :hidden:
    :glob:

    tutorials/*


.. toctree::
    :maxdepth: 1
    :caption: API Documentation
    :hidden:
    
    Core <apidocs/core>
