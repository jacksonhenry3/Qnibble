from itertools import permutations, groupby
from collections import Counter
import qutip as q
from utils import random_complex

def _generating_sequences(nqubit: int, energy: int) -> list:
    """creates multi sets of abcd sequences to be used as generators.
    Works by recursivly generating abcd sequences for lower energy subspaces and adding one element that will increase the energy by one.

     **a**: **0**->**0**
     **b**: **1**->**0**
     **c**: **0**->**1**
     **d**: **1**->**1**

     a sequence of all "a"s is in energy subspace zero.
     By replacing one "a" with a "d" we increase the energy subspace by one.
     By replacing two "a"s with a "b" and a "c" it increases the energy subspace by one.

    """

    # cant have more energy than qubits
    assert energy <= nqubit, f"there are {nqubit} qubits with {energy} energy, can't have more energy than qubits"

    # this is the recursive base case
    if energy == 0:
        return [Counter(a=nqubit)]

    result = []
    # loop through each generating sequence of one energy level lower
    for sequence in _generating_sequences(nqubit, energy - 1):
        s1, s2 = sequence.copy(), sequence.copy()

        # if there is at least one "a" in the sequence
        if s1['a'] > 0:
            # remove one a, add one d
            s1['a'], s1['d'] = s1['a'] - 1, s1['d'] + 1

            # add generator to result
            result += [s1]
        # if there is at least two "a"s in the sequence
        if s2['a'] > 1:
            # reduce the number of "a"s by two and add a "b" and a "c"
            s2['a'], s2['b'], s2['c'] = s2['a'] - 2, s2['b'] + 1, s2['c'] + 1

            # add generator to result
            result += [s2]

    # the +is a special counter thing that drops zeros.
    return list(+result for result, _ in groupby(result))


def random_hamiltonian(nqubit: int, energy: int) -> q.Qobj:
    """Generates a random hamiltonian of nqubits in energy subspace energy"""

    # Becouse Qobj is not hashable the generating sequences are made from charachters and then convert to qobj via this dict
    letter_to_Qobj = {'a': q.Qobj([[1, 0], [0, 0]]), 'b': q.Qobj([[0, 1], [0, 0]]), 'c': q.Qobj([[0, 0], [1, 0]]), 'd': q.Qobj([[0, 0], [0, 1]])}

    generators = _generating_sequences(nqubit, energy)

    # this creates a list of all abcd sequences in the energy subspace
    sequences_by_generator = [list(permutations(generator.elements())) for generator in generators if generator['b'] > 0]

    # using a set removes duplicates
    # flattens the list so it is just a list of sequences rather than a list of list of sequences (organized by which generator they are from)
    sequences = set([sequence for sequences in sequences_by_generator for sequence in sequences])

    # take the tensor product of each abcd sequence scaled by a random complex number
    non_hermitian_transformation = sum([random_complex() * q.tensor(*[letter_to_Qobj[letter] for letter in sequence]) for sequence in sequences])

    # to enforce hermiticity, add the conjugate transpose to the result
    return (non_hermitian_transformation + non_hermitian_transformation.dag())


def full_hamiltonian(nqubit: int, energy: int) -> q.Qobj:
    """Generates a random hamiltonian of nqubits in energy subspace energy"""

    # Becouse Qobj is not hashable the generating sequences are made from charachters and then convert to qobj via this dict
    letter_to_Qobj = {'a': q.Qobj([[1, 0], [0, 0]]), 'b': q.Qobj([[0, 1], [0, 0]]), 'c': q.Qobj([[0, 0], [1, 0]]), 'd': q.Qobj([[0, 0], [0, 1]])}

    generators = _generating_sequences(nqubit, energy)

    # this creates a list of all abcd sequences in the energy subspace
    sequences_by_generator = [list(permutations(generator.elements())) for generator in generators if generator['b'] > 0]

    # using a set removes duplicates
    # flattens the list so it is just a list of sequences rather than a list of list of sequences (organized by which generator they are from)
    sequences = set([sequence for sequences in sequences_by_generator for sequence in sequences])

    # take the tensor product of each abcd sequence scaled by a random complex number
    non_hermitian_transformation = sum([q.tensor(*[letter_to_Qobj[letter] for letter in sequence]) for sequence in sequences])

    # to enforce hermiticity, add the conjugate transpose to the result
    return (non_hermitian_transformation + non_hermitian_transformation.dag())
