import numpy as np

rng = np.random.default_rng()


def n_random_line_orders(num_qbits: int, chunk_sizes: list[int], n: int) -> list[np.ndarray]:
    assert sum(chunk_sizes) == num_qbits, "chunk sizes must add up to the number of qbits"
    split_indices = np.cumsum(chunk_sizes)[:-1]
    return [np.split(np.roll(np.arange(num_qbits), np.random.random_integers(num_qbits)), split_indices) for _ in range(n)]


def n_random_gas_orders(num_qbits: int, chunk_sizes: list[int], n: int) -> list[np.ndarray]:
    assert sum(chunk_sizes) == num_qbits, "chunk sizes must add up to the number of qbits"
    split_indices = np.cumsum(chunk_sizes)[:-1]
    return [np.split(rng.permutation(num_qbits), split_indices) for _ in range(n)]


def n_alternating_messenger_orders(num_qbits: int, n: int) -> list[np.ndarray]:
    """
    The first and last qbit will be the messenger qbits
    """
    first_order = np.split(np.arange(num_qbits), [num_qbits // 2])
    second_order = np.arange(num_qbits)
    second_order[[0, -1]] = second_order[[-1, 0]]
    second_order = np.split(second_order, [num_qbits // 2])

    return [first_order if i % 2 == 0 else second_order for i in range(n)]


def n_8qbit_c5_orders(n: int) -> list[np.ndarray]:
    possible_order_1 = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    possible_order_2 = np.array([[0, 1, 6, 7], [2, 3, 4, 5]])
    return [possible_order_1 if i % 2 == 0 else possible_order_2 for i in range(n)]
