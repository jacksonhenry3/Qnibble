import src.density_matrix as DM
import src.orders as orders
from src import measurements as measure
import numpy as np


def random(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
    """
    Args:
        two_qubit_dms_current: the current two qubit density matrices
        two_qubit_dms_previous: the previous two qubit density matrices
        pops: the current one qubit populations
        prev_pops: the previous one qubit populations
        past_order: the previous order
        connectivity: a string representing the connectivity of the qubits
    """

    num_qubits = len(pops)
    chunk_size = 2

    match connectivity:
        case 'c5':
            order = orders.n_random_c5_orders(num_qbits=num_qubits, chunk_size=chunk_size, n=1)[0]
        case 'c6':
            order = orders.n_random_c6_orders(num_qbits=num_qubits, chunk_size=chunk_size, n=1)[0]
        case 'c7':
            order = orders.n_random_c7_orders(num_qbits=num_qubits, chunk_size=chunk_size, n=1)[0]
        case 'gas':
            order = orders.n_random_gas_orders(num_qbits=num_qubits, chunk_size=chunk_size, n=1)[0]
        case _:
            raise ValueError(f"connectivity {connectivity} not recognized")
    return order


def greedy(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
    """
    Args:
        two_qubit_dms_current: the current two qubit density matrices
        two_qubit_dms_previous: the previous two qubit density matrices
        pops: the current one qubit populations
        prev_pops: the previous one qubit populations
        past_order: the previous order
        connectivity: a string representing the connectivity of the qubits
        sub_unitary: the unitary that operates on a group
        dm: the current density matrix
    """

    num_qubits = len(pops)
    chunk_size = 2

    # this is inefficient, dont need to recalculate every time
    match connectivity:
        case 'c5':
            all_orders = orders.all_c5_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c6':
            all_orders = orders.all_c6_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c7':
            all_orders = orders.all_c7_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case _:
            raise ValueError(f"connectivity {connectivity} not recognized")

    all_qubits = set([i for i in range(num_qubits)])

    qpopth = 0.225
    score_board = []
    for order in all_orders:
        dist = []
        pops_of_updated_sub_dm = []
        chunked_dms = [dm.ptrace(tuple(all_qubits - set(chunk))) for chunk in order]
        for sub_dm in chunked_dms:
            sub_dm.change_to_energy_basis()
            updated_sub_dm = sub_unitary * sub_dm * sub_unitary.H
            pops_of_updated_sub_dm.append(measure.pops(updated_sub_dm))
        pops_of_updated_sub_dm = np.array(pops_of_updated_sub_dm).flatten()
        for qpop in pops_of_updated_sub_dm:
            dist.append(abs(qpop - qpopth))
        score_card = [order, sum(dist)]
        score_board.append(score_card)

    current_max_score = 0
    current_order = None
    for order, score in score_board:
        if score > current_max_score:
            current_max_score = score
            current_order = order
    return current_order


def therm(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
    """
    Args:
        two_qubit_dms_current: the current two qubit density matrices
        two_qubit_dms_previous: the previous two qubit density matrices
        pops: the current one qubit populations
        prev_pops: the previous one qubit populations
        past_order: the previous order
        connectivity: a string representing the connectivity of the qubits
        sub_unitary: the unitary that operates on a group
        dm: the current density matrix
    """

    num_qubits = len(pops)
    chunk_size = 2

    # this is inefficient, dont need to recalculate every time
    match connectivity:
        case 'c5':
            all_orders = orders.all_c5_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c6':
            all_orders = orders.all_c6_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c7':
            all_orders = orders.all_c7_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case _:
            raise ValueError(f"connectivity {connectivity} not recognized")

    all_qubits = set([i for i in range(num_qubits)])

    qpopth = 0.225
    score_board = []
    for order in all_orders:
        dist = []
        pops_of_updated_sub_dm = []
        chunked_dms = [dm.ptrace(tuple(all_qubits - set(chunk))) for chunk in order]
        for sub_dm in chunked_dms:
            sub_dm.change_to_energy_basis()
            updated_sub_dm = sub_unitary * sub_dm * sub_unitary.H
            pops_of_updated_sub_dm.append(measure.pops(updated_sub_dm))
        pops_of_updated_sub_dm = np.array(pops_of_updated_sub_dm).flatten()
        for qpop in pops_of_updated_sub_dm:
            dist.append(abs(qpop - qpopth))
        score_card = [order, sum(dist)]
        score_board.append(score_card)

    current_min_score = 1.4
    current_order = None
    for order, score in score_board:
        if score <= current_min_score:
            current_min_score = score
            current_order = order
    return current_order


def mimic(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
    """
    Args:
        two_qubit_dms_current: the current two qubit density matrices
        two_qubit_dms_previous: the previous two qubit density matrices
        pops: the current one qubit populations
        prev_pops: the previous one qubit populations
        past_order: the previous order
        connectivity: a string representing the connectivity of the qubits
        sub_unitary: the unitary that operates on a group
        dm: the current density matrix
    """

    num_qubits = len(pops)
    chunk_size = 2

    # this is inefficient, dont need to recalculate every time
    match connectivity:
        case 'c5':
            all_orders = orders.all_c5_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c6':
            all_orders = orders.all_c6_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c7':
            all_orders = orders.all_c7_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case _:
            raise ValueError(f"connectivity {connectivity} not recognized")

    all_qubits = set([i for i in range(num_qubits)])

    qpopth = 0.225
    score_board = []
    for order in all_orders:
        dist = []
        pops_of_updated_sub_dm = []
        chunked_dms = [dm.ptrace(tuple(all_qubits - set(chunk))) for chunk in order]
        for sub_dm in chunked_dms:
            sub_dm.change_to_energy_basis()
            updated_sub_dm = sub_unitary * sub_dm * sub_unitary.H
            pops_of_updated_sub_dm.append(measure.pops(updated_sub_dm))
        pops_of_updated_sub_dm = np.array(pops_of_updated_sub_dm).flatten()
        for qpop in pops_of_updated_sub_dm:
            dist.append(abs(qpop - qpopth))
        score_card = [order, sum(dist)]
        score_board.append(score_card)

    current_max_score = 0
    current_order = None
    for order, score in score_board:
        if score > current_max_score:
            current_max_score = score
            current_order = order
    return current_order
