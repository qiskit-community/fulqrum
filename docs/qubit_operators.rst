.. _qubitops:

#####################
Using QubitOperators
#####################

One of the two core data structures used by Fulqrum is the ``QubitOperator`` that represents Hamiltonians defined
over qubit subsystems.  Here we will go through examples of the core ``QubitOperator`` functionality that is pertinent
to most use cases.  To begin, we will import fulqrum:  

.. jupyter-execute::

    import fulqrum as fq


Single-term operators
#####################

Create an empty operator over `5` qubits:

.. jupyter-execute::

    fq.QubitOperator(5)


Create a `N` qubit single-term operator from a dense (including identity operators) string,
where `N` is equal to the length of the string:

.. jupyter-execute::

    fq.QubitOperator.from_label('IIXYI')


Create a `5` qubit, single-term operator using sparse notion specifying an `X` operator on qubit `0` 
and a `1` operator on qubit `3`, and with a coefficient of `2.2`:

.. jupyter-execute::

    H = fq.QubitOperator(5, [("X1", [0, 3], 2.2)])
    H

Check the properties of the ``QubitOperator``.  Number of qubits:

.. jupyter-execute::

    H.width


or, number of terms:

.. jupyter-execute::

    H.num_terms

The number of terms can also be found via the `len()`:

.. jupyter-execute::

    len(H)


Multi-term operators
####################

Create a multi-term ``QubitOperator`` using the sparse notion:

.. jupyter-execute::

    H = fq.QubitOperator(5, [("X1", [0, 3], 2.2), ("ZZY", [0, 1, 4], -1)])
    H


.. jupyter-execute::

    H.num_terms


Create a multi-term operator by adding one operator to another, returning a new operator:


.. jupyter-execute::

    H1 = fq.QubitOperator.from_label('ZIXYI', 2)
    H2 = fq.QubitOperator(5, [('ZZIZZ', range(5), -1.0)])
    H1 + H2


Create a multi-term operator by adding an operator in-place to an existing operator:


.. jupyter-execute::

    H1 += H2
    H1