import src.density_matrix as DM
import src.orders as orders
from src import measurements as measure
import numpy as np
import random

def disorder_random(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
    pops = list(pops.values())
    num_qubits = len(pops)
    if num_qubits == 8:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.5, 0., 0.5]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 10:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.5, 0.5]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 12:
        list_of_lists = [
        [[0,1], [2, 3], [4, 5], [6, 7],[8,9],[10,11]],
        [[1,2], [3, 4], [5, 6], [7, 8],[9,10],[0,11]]]

    # Probabilities for each list, should sum to 1
        probabilities = [0.5, 0.5]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 14:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    return current_order

def disorder_greedy(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
    pops = list(pops.values())
    num_qubits = len(pops)
    if num_qubits == 8:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 10:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 12:
        list_of_lists = [
        [[0,1], [2, 3], [4, 5], [6, 7],[8,9],[10,11]],
        [[1,2], [3, 4], [5, 6], [7, 8],[9,10],[0,11]]]

    # Probabilities for each list, should sum to 1
        probabilities = [0.49, 0.51]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 14:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    return current_order

def disorder_mimic(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
    pops = list(pops.values())
    num_qubits = len(pops)
    if num_qubits == 8:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 10:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 12:
        list_of_lists = [
        [[0,1], [2, 3], [4, 5], [6, 7],[8,9],[10,11]],
        [[1,2], [3, 4], [5, 6], [7, 8],[9,10],[0,11]]]

    # Probabilities for each list, should sum to 1
        probabilities = [0.47, 0.53]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 14:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    return current_order

def disorder_landscape_maximizes(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
    pops = list(pops.values())
    num_qubits = len(pops)
    if num_qubits == 8:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 10:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 12:
        list_of_lists = [
        [[0,1], [2, 3], [4, 5], [6, 7],[8,9],[10,11]],
        [[1,2], [3, 4], [5, 6], [7, 8],[9,10],[0,11]]]

    # Probabilities for each list, should sum to 1
        probabilities = [0.35, 0.65]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 14:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    return current_order

def disorder_strongest_maximizes(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
    pops = list(pops.values())
    num_qubits = len(pops)
    if num_qubits == 8:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 10:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 12:
        list_of_lists = [
        [[0,1], [2, 3], [4, 5], [6, 7],[8,9],[10,11]],
        [[1,2], [3, 4], [5, 6], [7, 8],[9,10],[0,11]]]

    # Probabilities for each list, should sum to 1
        probabilities = [0.66, 0.34]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    if num_qubits == 14:
        list_of_lists = [
        [[4, 6], [1, 7], [0, 3], [2, 5]],
        [[4, 6], [1, 2], [0, 3], [5, 7]],
        [[1, 7], [2, 5], [3, 4], [0, 6]],
    ]

    # Probabilities for each list, should sum to 1
        probabilities = [0.9, 0., 0.1]  # Example: probabilities for each list

    # Ensure that probabilities sum to 1 (or close enough)
        assert abs(sum(probabilities) - 1) < 1e-6, "Probabilities must sum to 1"
        current_order = random.choices(list_of_lists, weights=probabilities, k=1)[0]
    return current_order

