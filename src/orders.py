import numpy as np

rng = np.random.default_rng()


def n_random_line_orders(line_length: int, n: int) -> list[list[int]]:
    return [np.roll(np.arange(line_length), np.random.random_integers(line_length)) for _ in range(n)]


def n_random_gas_orders(num_qbits: int, n: int) -> list[np.ndarray]:
    return [rng.permutation(num_qbits) for _ in range(n)]
