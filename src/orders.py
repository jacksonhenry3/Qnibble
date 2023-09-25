import numpy as np

rng = np.random.default_rng()


def n_random_line_orders(num_qbits: int, chunk_sizes: list[int], n: int) -> list[np.ndarray]:
    assert sum(chunk_sizes) == num_qbits, "chunk sizes must add up to the number of qbits"
    split_indices = np.cumsum(chunk_sizes)[:-1]
    return [np.split(np.roll(np.arange(num_qbits), np.random.random_integers(num_qbits)), split_indices) for _ in range(n)]


def n_line_orders(num_qbits: int, chunk_sizes: list[int], n: int) -> list[np.ndarray]:
    assert sum(chunk_sizes) == num_qbits, "chunk sizes must add up to the number of qbits"
    split_indices = np.cumsum(chunk_sizes)[:-1]
    return [np.split(np.roll(np.arange(num_qbits), i), split_indices) for i in range(n)]


def n_random_gas_orders(num_qbits: int, chunk_sizes: list[int], n: int) -> list[np.ndarray]:
    assert sum(chunk_sizes) == num_qbits, "chunk sizes must add up to the number of qbits"
    split_indices = np.cumsum(chunk_sizes)[:-1]
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

    return [first_order if i % 2 == 0 else second_order for i in range(n)]


def n_alternating_c5_orders(num_qbits: int, n: int) -> list[np.ndarray]:
    assert num_qbits % 2 == 0, "n must be even"
    a = np.arange(num_qbits).reshape(2, num_qbits // 2)
    possible_order_1 = np.array([(a[:, i: i + 2]).flatten() for i in range(0, num_qbits // 2, 2)])
    a = np.roll(a, 1, axis=1)
    possible_order_2 = np.array([(a[:, i: i + 2]).flatten() for i in range(0, num_qbits // 2, 2)])
    return [possible_order_1 if i % 2 == 0 else possible_order_2 for i in range(n)]
