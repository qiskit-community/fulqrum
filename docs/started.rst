.. _started:

###############
Getting started
###############

Simple example
##############

Here we demonstrate the standard workflow using a simple example solving a Heisenberg problem on a 5x5 grid.  

To begin, we import the required libraries, and build our Hamiltonian as a ``QubitOperator``:

.. jupyter-execute::

    import scipy.sparse.linalg as spla
    from qiskit.transpiler import CouplingMap
    import fulqrum as fq

    # Build 25-qubit coupling map
    cmap = CouplingMap.from_grid(5, 5)
    num_qubits = cmap.size()

    # Generate Hamiltonian
    H = fq.QubitOperator(num_qubits, [])
    touched_edges = set({})
    coeffs = [1/2, 1/2, 1]
    for edge in cmap.get_edges():
        if edge[::-1] not in touched_edges:
            H += fq.QubitOperator(num_qubits, [("XX", edge, coeffs[0]), 
                                            ("YY", edge, coeffs[1]), 
                                            ("ZZ", edge, coeffs[2])])
            touched_edges.add(edge)


The subspace in which our Hamiltonian will be defined is derived from the counts sampled from a quantum computer
and stored as a ``Subspace`` object:

.. jupyter-execute::

    #100k pseudo counts
    counts = {}
    for kk in range(int(1e5)):
        counts[bin(kk)[2:].zfill(num_qubits)] = 1

    S = fq.Subspace(counts)

Right now we have a Hamiltonian and a subspace, but they know nothing about each other.  We can combine
them into a ``SubspaceHamiltonian`` object, which is where all the magic happens, and is what we pass on
to the eigensolver:


.. jupyter-execute::

    Hsub = fq.SubspaceHamiltonian(H, S)
    evals, evecs = spla.eigsh(Hsub, k=1, which='SA')


The found eigenenergy is:

.. jupyter-execute::

    print("Eigenvalue:", evals[0])


there is of course an associated eigenvector, but because of the way Fulqrum solves the problem, it is not
immediately usable.  Instead, we can use ``SubspaceHamiltonian.interpret_vector()`` to cast the solution
as a dictionary with complex values:

.. jupyter-execute::

    Hsub.interpret_vector(evecs)

