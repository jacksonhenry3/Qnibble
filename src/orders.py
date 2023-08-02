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
