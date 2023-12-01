import numpy as np
import numpy.random

rng = np.random.default_rng()


def n_random_c5_orders(num_qbits: int, n: int) -> list[np.ndarray]:
    assert num_qbits % 4 == 0, "n must be divisible by 4"
    assert num_qbits >= 8, "n must be at least 8"
    a = np.arange(num_qbits).reshape(2, num_qbits // 2)
    possible_order_1 = np.array([(a[:, i: i + 2]).flatten() for i in range(0, num_qbits // 2, 2)])
    a = np.roll(a, 1, axis=1)
    possible_order_2 = np.array([(a[:, i: i + 2]).flatten() for i in range(0, num_qbits // 2, 2)])
    orders = [possible_order_1, possible_order_2]
    result = []
    for _ in range(n):
        order = rng.choice(orders)
        result.append(np.array([rng.permuted(chunk) for chunk in order]))

    return result


def n_random_c6_orders(num_qbits: int, n: int) -> list[np.ndarray]:
    assert num_qbits % 4 == 0, "n must be divisible by 4"
    split_indices = [4 * i for i in range(1, num_qbits // 4)]

    result = []
    for _ in range(n):
        order = np.split(np.roll(np.arange(num_qbits), np.random.randint(num_qbits)), split_indices)
        permuted_order = [np.random.permutation(chunk) for chunk in order]
        result.append(np.array(permuted_order))

    return result


def n_random_c7_orders(num_qbits: int, n: int) -> list[np.ndarray]:
    assert num_qbits % 4 == 0, "n must be divisible by 4"
    assert num_qbits >= 8, "n must be at least 12"

    if num_qbits == 8:
        # groups = [vv, H, _H]
        groups = [[[0, 2, 1, 3], [4, 6, 5, 7]], [[0, 2, 4, 6], [1, 3, 5, 7]], [[4, 6, 1, 3], [0, 2, 5, 7]]]
    elif num_qbits == 12:
        # groups = [vvv, vH, _vH, Hv] This system doesnt include cross horiozontal terms, ive added them on the end
        groups = [[[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 11, 9]],
                  [[0, 2, 1, 3], [4, 6, 8, 10], [5, 7, 9, 11]],
                  [[0, 2, 9, 11], [1, 3, 8, 10], [4, 6, 5, 7]],
                  [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 9, 11]],
                  [[1, 3, 5, 7], [4, 6, 8, 10], [9, 11, 0, 2]],
                  [[0, 2, 4, 6], [5, 7, 9, 11], [8, 10, 1, 3]]
                  ]
    elif num_qbits == 16:
        # groups = [vvvv, vvH, vHv, Hvv, _vvH, HH, _HH]
        groups = [[[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 9, 11], [12, 14, 13, 15]],
                  [[0, 2, 1, 3], [4, 6, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]],
                  [[0, 2, 1, 3], [4, 6, 8, 10], [5, 7, 9, 11], [12, 14, 13, 15]],
                  [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 9, 11], [12, 14, 13, 15]],
                  [[0, 2, 13, 15], [1, 3, 12, 14], [4, 6, 5, 7], [8, 10, 9, 11]],
                  [[0, 2, 4, 6], [1, 3, 5, 7], [8, 10, 12, 14], [9, 11, 13, 15]],
                  [[0, 2, 13, 15], [1, 3, 12, 14], [4, 6, 8, 10], [5, 7, 9, 11]]
                  ]
    result = []
    for _ in range(n):
        order = np.array([np.random.permutation(chunk) for chunk in rng.choice(groups)])
        result.append(order)

    return result


def n_random_gas_orders(num_qbits: int, n: int) -> list[np.ndarray]:
    assert num_qbits % 4 == 0, "n must be divisible by 4"
    split_indices = [4 * i for i in range(1, num_qbits // 4)]
    return [np.split(rng.permutation(num_qbits), split_indices) for _ in range(n)]


def n_alternating_messenger_orders(num_qbits: int, n: int) -> list[np.ndarray]:
    """
    The first and last qbit will be the messenger qbits
    """

    assert num_qbits % 4 == 0, "n must be a multiple of 4"
    first_order = np.arange(num_qbits).reshape((num_qbits // 4, 4))
    second_order = np.arange(num_qbits)
    second_order[[0, -1]] = second_order[[-1, 0]]
    second_order = second_order.reshape((num_qbits // 4, 4))

    result = []
    for _ in range(n):
        order = rng.choice([first_order, second_order])
        result.append(np.array([rng.permutation(chunk) for chunk in order]))

    return result
