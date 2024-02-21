import src.density_matrix as DM
import src.orders as orders


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
    chunk_size = 4

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
    chunk_size = 4

    # this is innefecient, dont need to recalculate every time
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

    for order in all_orders:
        chunked_dms = [dm.ptrace(tuple(all_qubits - set(chunk))) for chunk in order]
        score = 0
        for sub_dm in chunked_dms:
            updated_sub_dm = sub_unitary * sub_dm * sub_unitary.H




    return order
