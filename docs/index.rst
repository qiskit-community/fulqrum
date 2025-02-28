#######
Fulqrum
#######

.. figure:: images/fulqrum.png
    :align: center


Fulqrum is a set of tools for enabling the solution to large-scale Hamiltonian subpspace eigenproblems over extended alphabets for those of us without access to high-performance computing (HPC) resources.  To accomplish this, Fulqrum utilizes a novel matrix-free method for performing the matrix-vector computation that is at the core of all sparse eigensolving methods.

Working over extended (i.e. non-Pauli) alphabets allows Fulqrum to be applicable for both Bosonic and Fermionic problems.  Fermionic problems can be cast into Bosonic ones in a one-to-one manner using an extended Jordan-Wigner transformation, and the properties of extended operators can be used to further reduce the computational costs.


.. toctree::
    :maxdepth: 1
    :hidden:

    self
    How it works <how_it_works>
    Installation <install>