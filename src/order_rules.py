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

    num_qbits = len(pops)
    chunk_size = 4

    #match connectivity:
        #case 'c5':
         #   order = orders.n_random_c5_orders(num_qbits=num_qubits, chunk_size=chunk_size, n=1)[0]
        #case 'c6':
         #   order = orders.n_random_c6_orders(num_qbits=num_qubits, chunk_size=chunk_size, n=1)[0]
        #case 'c7':
         #   order = orders.n_random_c7_orders(num_qbits=num_qubits, chunk_size=chunk_size, n=1)[0]
        #case 'gas':
         #   order = orders.n_random_gas_orders(num_qbits=num_qubits, chunk_size=chunk_size, n=1)[0]
        #case _:
         #   raise ValueError(f"connectivity {connectivity} not recognized")
    #return order
    match connectivity:
        case 'c2_2local':
            order = orders.n_random_c2_2local_orders(num_qbits=num_qbits, chunk_size=chunk_size)
        case 'c4_2local':
            order = orders.n_random_c4_2local_orders(num_qbits=num_qbits, chunk_size=chunk_size)
        case 'c5':
            order = orders.n_random_c5_orders(num_qbits=num_qbits, chunk_size=chunk_size, n=1)[0]
        case 'c5_2local':
             order = orders.n_random_c5_2local_orders(num_qbits=num_qbits, chunk_size=chunk_size)
        case 'c6_2local':
            order = orders.n_random_c6_2local_orders(num_qbits=num_qbits, chunk_size=chunk_size)
        case 'cN_2local':
            order = orders.n_random_cN_2local_orders(num_qbits=num_qbits, chunk_size=chunk_size)
        case 'c6':
            order = orders.n_random_c6_orders(num_qbits=num_qbits, chunk_size=chunk_size, n=1)[0]
        case 'c7':
            order = orders.n_random_c7_orders(num_qbits=num_qbits, chunk_size=chunk_size, n=1)[0]
        case 'gas':
            order = orders.n_random_gas_orders(num_qbits=num_qbits, chunk_size=chunk_size, n=1, seed=unitary_rng)[0]
        case _:
                # throw an explanatory error
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
    #several comments are present in the code as a way to debug in case the rule is misbehaving

    #convert pops to a list otherwise it is a dictionary
    pops = list(pops.values())
    num_qubits = len(pops)
    chunk_size = 2

    # this is inefficient, dont need to recalculate every time
    match connectivity:
        case'c2_2local':
            all_orders = orders.all_c2_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'c4_2local':
            all_orders = orders.all_c4_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'c5_2local':
            all_orders = orders.all_c5_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'c6_2local':
            all_orders = orders.all_c6_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'cN_2local':
            all_orders = orders.all_cN_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c5':
            all_orders = orders.all_c5_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c6':
            all_orders = orders.all_c6_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c7':
            all_orders = orders.all_c7_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case _:
            raise ValueError(f"connectivity {connectivity} not recognized")

    all_qubits = set([i for i in range(num_qubits)])

    #print(f"all_orders: {all_orders}")  # Check if all_orders is populated

    qpopth = (1/num_qubits)*sum(pops)
    #print(f"thermal Q pop: {qpopth}")  # Check if all_orders is populated

    score_board = []
    for order in all_orders:
        #print(f"Processing order: {order}")
        dist = []
        pops_of_updated_sub_dm = []
        order=order.tolist()
        chunked_dms = [dm.ptrace(tuple(all_qubits - set(chunk))) for chunk in order]
        order=np.array(order)
        for sub_dm in chunked_dms:
            sub_dm.change_to_energy_basis()
            updated_sub_dm = sub_unitary * sub_dm * sub_unitary.H
            pops_of_updated_sub_dm.append(measure.pops(updated_sub_dm))
        pops_of_updated_sub_dm = np.array(pops_of_updated_sub_dm).flatten()
        #print(f"Current populations: {pops}")
        #print(f"Updated populations: {pops_of_updated_sub_dm}")
        for qpop in pops_of_updated_sub_dm:
            dist.append(abs(qpop - qpopth))
        score_card = [order, sum(dist)]
        #print(f"Adding score card: {score_card}")
        score_board.append(score_card)
        #print(f"Final score card: {score_card}")

    current_max_score = 0
    current_order = None
    for order, score in score_board:
        if score > current_max_score:
            current_max_score = score
            current_order = order
    if current_order is None:
        raise ValueError("score_board was empty; no order selected")

    #print(f"Final Order: {current_order}")

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


def weakest_maximizes(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
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
        case'c2_2local':
            all_orders = orders.all_c2_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'c4_2local':
            all_orders = orders.all_c4_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'c5_2local':
            all_orders = orders.all_c5_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'c6_2local':
            all_orders = orders.all_c6_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'cN_2local':
            all_orders = orders.all_cN_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
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
    extractable_work_i0 = np.array(
        measure.extractable_work_of_each_qubit_from_pops(prev_pops))
    extractable_work_i1 = np.array(
        measure.extractable_work_of_each_qubit_from_pops(pops))
    change_in_ex_work = extractable_work_i1-extractable_work_i0
    decider_Q_index = np.argmin(change_in_ex_work)

    score_board = []
    for order in all_orders:
        change_in_ex_work_decider_Q=0
        #change_in_ex_work_decider_Q = []
        pops_of_updated_sub_dm = []
        chunked_dms = [dm.ptrace(tuple(all_qubits - set(chunk))) for chunk in order]
        for sub_dm in chunked_dms:
            sub_dm.change_to_energy_basis()
            updated_sub_dm = sub_unitary * sub_dm * sub_unitary.H
            pops_of_updated_sub_dm.append(measure.pops(updated_sub_dm))
        pops_of_updated_sub_dm = np.array(pops_of_updated_sub_dm).flatten()
        extractable_work_trial_0 = np.array(
            measure.extractable_work_of_each_qubit_from_pops(pops))
        extractable_work_trial_1 = np.array(
            measure.extractable_work_of_each_qubit_from_pops(pops_of_updated_sub_dm))
        change_in_ex_work = extractable_work_trial_1 - extractable_work_trial_0
        change_in_ex_work_decider_Q=change_in_ex_work[decider_Q_index]
        score_card = [order, change_in_ex_work_decider_Q]
        score_board.append(score_card)

    max_order = None
    max_change = float('-inf')
    current_order = past_order
    # Iterate through each order and its associated change value
    for order, change in score_board:
        # Check if the current change value is greater than the maximum found so far
        if change >= max_change:
            # If it is, update the maximum change value and the corresponding order
            max_change = change
            current_order = order
    return current_order


def landscape_maximizes(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
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

    #several comments are present in the code as a way to debug in case the rule is misbehaving

    chunk_size = 2
    #convert pops into a list. They way its stored, pops is a dicitonary!!
    pops = list(pops.values())
    pops = [max(pop, 0) for pop in pops]

    num_qubits = len(pops)
    #print(f"Length of pops: {pops}")

    match connectivity:
        case'c2_2local':
            all_orders = orders.all_c2_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'c4_2local':
            all_orders = orders.all_c4_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'c5_2local':
            all_orders = orders.all_c5_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'c6_2local':
            all_orders = orders.all_c6_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case'cN_2local':
            all_orders = orders.all_cN_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c5':
            all_orders = orders.all_c5_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c6':
            all_orders = orders.all_c6_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case 'c7':
            all_orders = orders.all_c7_orders(num_qbits=num_qubits, chunk_size=chunk_size)
        case _:
            raise ValueError(f"connectivity {connectivity} not recognized")

    all_qubits = set([i for i in range(num_qubits)])

    #extractable_work_i0 = np.array(
     #   measure.extractable_work_of_each_qubit_from_pops(prev_pops))
    #extractable_work_i1 = np.array(
     #   measure.extractable_work_of_each_qubit_from_pops(pops))
    #change_in_ex_work = extractable_work_i1-extractable_work_i0
    #decider_Q_index = np.argmin(change_in_ex_work)
    #print(f"all_orders: {all_orders}")  # Check if all_orders is populated
    score_board = []
    for order in all_orders:
        #print(f"Processing order: {order}")
        # Rest of the code for processing orders...
        pops_of_updated_sub_dm = []
        total_change_in_ex_work = []
        order=order.tolist()
        chunked_dms = [dm.ptrace(tuple(all_qubits - set(chunk))) for chunk in order]
        order = np.array(order)
        for sub_dm in chunked_dms:
            sub_dm.change_to_energy_basis()
            updated_sub_dm = sub_unitary * sub_dm * sub_unitary.H
            pops_of_updated_sub_dm.append(measure.pops(updated_sub_dm))
        pops_of_updated_sub_dm = np.array(pops_of_updated_sub_dm).flatten()
        pops_of_updated_sub_dm = pops_of_updated_sub_dm.tolist()
        #print(f"Current populations: {pops}")
        #print(f"Updated populations: {pops_of_updated_sub_dm}")
        if np.any(np.isnan(pops_of_updated_sub_dm)):
            #print("NaN values found in updated populations.")
            continue

        extractable_work_trial_0 = np.array(
            measure.extractable_work_of_each_qubit_from_pops(pops))
        extractable_work_trial_1 = np.array(
            measure.extractable_work_of_each_qubit_from_pops(pops_of_updated_sub_dm))

        #print(f"Extractable work trial 0: {extractable_work_trial_0}")
        #print(f"Extractable work trial 1: {extractable_work_trial_1}")

        change_in_ex_work = extractable_work_trial_1 - extractable_work_trial_0

        #print(f"Change in extractable work: {change_in_ex_work}")

        total_change_in_ex_work = sum(change_in_ex_work)

        #print(f"Total change in extractable work: {total_change_in_ex_work}")

        score_card = [order, total_change_in_ex_work]
        #print(f"Adding score_card: {score_card}")

        score_board.append(score_card)
        #print(f"Final score_board: {score_board}")

    max_change = float('-inf')
    current_order = None
    # Iterate through each order and its associated change value
    for order, change in score_board:
        # Check if the current change value is greater than the maximum found so far
        if change >= max_change:
            # If it is, update the maximum change value and the corresponding order
            max_change = change
            current_order = order
    #print(f"Final Order: {current_order}")

    if current_order is None:
        raise ValueError("score_board was empty; no order selected")
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
        # several comments are present in the code as a way to debug in case the rule is misbehaving

        # convert pops into a list. They way its stored, pops is a dicitonary!!
        pops = list(pops.values())
        prev_pops = list(prev_pops.values())
        #print(f"Current pops: {pops}")
        #print(f"Prev pops: {prev_pops}")

        num_qubits = len(pops)
        chunk_size = 2

        # this is inefficient, dont need to recalculate every time
        match connectivity:
            case 'c2_2local':
                all_orders = orders.all_c2_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
            case 'c4_2local':
                all_orders = orders.all_c4_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
            case 'c5_2local':
                all_orders = orders.all_c5_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
            case 'c6_2local':
                all_orders = orders.all_c6_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
            case 'cN_2local':
                all_orders = orders.all_cN_2local_orders(num_qbits=num_qubits, chunk_size=chunk_size)
            case 'c5':
                all_orders = orders.all_c5_orders(num_qbits=num_qubits, chunk_size=chunk_size)
            case 'c6':
                all_orders = orders.all_c6_orders(num_qbits=num_qubits, chunk_size=chunk_size)
            case 'c7':
                all_orders = orders.all_c7_orders(num_qbits=num_qubits, chunk_size=chunk_size)
            case _:
                raise ValueError(f"connectivity {connectivity} not recognized")

        all_qubits = set([i for i in range(num_qubits)])

        # find all neighbours of a qubit
        def find_shared_elements(list_of_lists_of_lists, target_element):
            shared_elements = set()  # Use a set to store unique elements
            for sublist in list_of_lists_of_lists:
                for subsublist in sublist:
                    if target_element in subsublist:
                        shared_elements.update([elem for elem in subsublist if elem != target_element])
            return list(shared_elements)

        #Gives a list of lists of lists. Each list inside the list is the ordered according q index and corresponds to the lists of neighbours that qubit index has
        # list of lists where each list is the set of neighbours of the qubit with that list index. ie the nth list is a list of neighbours of the nth qubit on landscape
        neighbours_qubit_index = []
        for id in range(num_qubits):
            neighbours_qubit_index.append(find_shared_elements(all_orders, id))
        #print(f"List of neighbours per Q: {neighbours_qubit_index}")


        extractable_work_i0 = np.array(
            measure.extractable_work_of_each_qubit_from_pops(prev_pops))
        extractable_work_i1 = np.array(
            measure.extractable_work_of_each_qubit_from_pops(pops))

        #print(f"Ext Work prev step: {extractable_work_i0}")
        #print(f"Current Ext Work: {extractable_work_i1}")

        change_in_ex_work_prev_step = extractable_work_i1 - extractable_work_i0
        #print(f"Change in ext work at current step: {change_in_ex_work_prev_step}")


        # code to find the qubit the target qubit was paired in the previous order
        # Returns a qubit index corresponding to the  past orders groupings
        def paired_element(given_element, past_order):
            for pair in past_order:
                # Convert the NumPy array to a list
                pair_list = pair.tolist()
                if given_element in pair_list:
                    # Return the other element in the pair
                    return pair_list[1 - pair_list.index(given_element)]
            # If the given element is not found in any pair, return None
            return None

        # code to find the qubit indices with change in extractable work in ascending order. These will help find the decider qubits
        # Returns a list of indices who have the corresponding value given in the argument in ascending order
        def find_indices_in_ascending_order(values):
            # Enumerate the values to get (index, value) pairs
            indexed_values = list(enumerate(values))

            # Sort the indexed values based on the values (the second element of each tuple)
            sorted_indices = sorted(indexed_values, key=lambda x: x[1])

            # Extract the indices from the sorted list of (index, value) pairs
            indices_in_ascending_order = [index for index, _ in sorted_indices]

            return indices_in_ascending_order

        decider_Q_index = find_indices_in_ascending_order(change_in_ex_work_prev_step)
        #print(f"Decider Q to start mimic: {decider_Q_index}")

        # code to find allowed neighbourhoods after some have been fixed

        current_order = []
        order = []

        def find_lists_with_sublist(list_of_lists_of_lists, sublist_of_lists_to_match):
            matching_lists = []
            scoreboard = []
            for lists_of_lists in list_of_lists_of_lists:
                score = 0
                for pair in sublist_of_lists_to_match:
                    for ordered_pair in lists_of_lists:
                        if np.array_equal(ordered_pair, [pair[0], pair[1]]) or np.array_equal(ordered_pair,
                                                                                              [pair[1], pair[0]]):
                            score = score + 1
                list_score = [lists_of_lists, score]
                scoreboard.append(list_score)
            # return scoreboard
            for lists, scores in scoreboard:
                if scores == len(sublist_of_lists_to_match):
                    matching_lists.append(lists)
            return matching_lists

        for qubit_id in decider_Q_index:
            #print(f"Qubit id in loop: {qubit_id}")
            # we start with the qubits in ascending order
            # we check that the qubit in loop does not already exist in the pairings made previously; this would not matter for the first qubit but can hinder later on
            fullorder = np.array(order).flatten()
            #print(f"Full order: {fullorder}")
            if qubit_id not in fullorder:
                #print(f"Qubit id not in loop: {qubit_id}")
                # we proceed only if the qubit id does not already exists in the ordering
                # print(qubit_id)
                sub_order = []
                max_D_W_ex = float('-inf')
                # to check the possible neighbours the qubit has, we need to make sure that we are looking wihin the subset of lists that obey the already found orders
                if len(order) > 0:
                    order_array = np.array(order)
                    match = find_lists_with_sublist(all_orders, order_array)
                    # finds the subset of full network ordering for which the subset of pairs match
                    # print(match)
                    allowed_orders = match
                    #print(f"allowed orders: {allowed_orders}")
                else:
                    allowed_orders = all_orders
                if len(allowed_orders) == 1:
                    #print(f"Was length equal to 1 yes!")
                    current_order = allowed_orders[0]
                    # print("this happened")
                    # if only one ordering is allowed then no need to do further optimization. This is the order of choice set by other qubits and connectivity
                    #print(f"Current order : {current_order}")
                    #return current_order
                else:
                    # for id in range(num_qubits):
                    # neighbours_qubit_index.append(find_shared_elements(allowed_orders, id))
                    neighbours = find_shared_elements(allowed_orders, qubit_id)
                    #print(f"Neighbours : {neighbours}")
                    # neighbours = neighbours_qubit_index[qubit_id]
                    # neighbours = [x for x in neighbours if x not in np.array(order).flatten()]
                    # print(neighbours)
                    pop_diff_to_mimic = 0
                    for neighbour in neighbours:
                        #print(f"neighbour currently checking : {neighbour}")
                        if max_D_W_ex <= change_in_ex_work_prev_step[neighbour]:
                            max_D_W_ex = change_in_ex_work_prev_step[neighbour]
                            neighbour_to_mimic = neighbour
                            pop_diff_to_mimic = prev_pops[neighbour_to_mimic] - prev_pops[
                                paired_element(neighbour_to_mimic, past_order)]
                    #print(f"neighbour to mimic: {neighbour_to_mimic}")
                    diff = 0
                    min_diff = float('inf')
                    for neighbour in neighbours:
                        pop_diff = pops[qubit_id] - pops[neighbour]
                        diff_with_n = pop_diff - pop_diff_to_mimic
                        #print(f"pop diff: {pop_diff}")
                        #print(f"compared to pop diff to mimic: {diff_with_n}")
                        if diff_with_n <= min_diff:
                            Q_pair = neighbour
                            min_diff=diff_with_n
                    #print(f"Neighbour getting paired with: {Q_pair}")
                    sub_order = [qubit_id, Q_pair]
                    #print(f"Sub order : {sub_order}")
                    order.append(sub_order)
                    # print(order)
                    current_order = order

        #print(f"Current order : {current_order}")
        return current_order



def thermodynamic(past_order, prev_pops, pops, two_qubit_dms_previous, two_qubit_dms_current, connectivity, sub_unitary, dm):
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
    extractable_work_i0 = np.array(
        measure.extractable_work_of_each_qubit_from_pops(prev_pops))
    extractable_work_i1 = np.array(
        measure.extractable_work_of_each_qubit_from_pops(pops))
    change_in_ex_work = extractable_work_i1-extractable_work_i0
    decider_Q_index = np.argmin(change_in_ex_work)

    score_board = []
    for order in all_orders:
        change_in_ex_work_decider_Q=0
        #change_in_ex_work_decider_Q = []
        pops_of_updated_sub_dm = []
        chunked_dms = [dm.ptrace(tuple(all_qubits - set(chunk))) for chunk in order]
        for sub_dm in chunked_dms:
            sub_dm.change_to_energy_basis()
            updated_sub_dm = sub_unitary * sub_dm * sub_unitary.H
            pops_of_updated_sub_dm.append(measure.pops(updated_sub_dm))
        pops_of_updated_sub_dm = np.array(pops_of_updated_sub_dm).flatten()
        extractable_work_trial_0 = np.array(
            measure.extractable_work_of_each_qubit_from_pops(pops))
        extractable_work_trial_1 = np.array(
            measure.extractable_work_of_each_qubit_from_pops(pops_of_updated_sub_dm))
        change_in_ex_work = extractable_work_trial_1 - extractable_work_trial_0
        change_in_ex_work_decider_Q=change_in_ex_work[decider_Q_index]
        score_card = [order, change_in_ex_work_decider_Q]
        score_board.append(score_card)

    max_order = None
    max_change = float('-inf')
    current_order = past_order
    # Iterate through each order and its associated change value
    for order, change in score_board:
        # Check if the current change value is greater than the maximum found so far
        if change >= max_change:
            # If it is, update the maximum change value and the corresponding order
            max_change = change
            current_order = order
    return current_order