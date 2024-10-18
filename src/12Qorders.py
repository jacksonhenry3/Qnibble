import numpy as np
import numpy.random
from functools import cache
import random

def generate_unique_pairs(num_qbits):
    # Create a list of numbers from 0 to 11
    numbers = list(range(num_qbits))
    # Shuffle the list to randomize the order
    random.shuffle(numbers)
    # Create pairs from the shuffled list
    pairs = [[numbers[i], numbers[i + 1]] for i in range(0, num_qbits, 2)]
    return np.array(pairs)

@cache
def all_c2_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    orders = []
    if num_qbits == 12:
        orders = [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]),
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 0]])]
    return orders


def n_random_c2_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    rand_order = 0
    if num_qbits == 12:
        rand_order = random.choice(
            [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]),
             np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 0]])])
    return rand_order

def first_10_orders_C2_2local(num_qbits):
    first_10_orders = []
    order=0
    for n in range(10):
        order = n_random_c2_2local_orders(num_qbits,2)
        first_10_orders.append(order)
    return first_10_orders

#c4_2local

@cache
def all_c4_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    orders = []
    if num_qbits == 12:
        orders = [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]),
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [0, 13]])]
    return orders


def n_random_c4_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    rand_order = 0
    if num_qbits == 12:
        rand_order = random.choice(
            [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]),
             np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [0, 13]])])
    return rand_order

def first_10_orders_C4_2local(num_qbits):
    first_10_orders = []
    order=0
    for n in range(10):
        order = n_random_c4_2local_orders(num_qbits,2)
        first_10_orders.append(order)
    return first_10_orders

#c5_2local

@cache
def all_c5_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    orders = []
    if num_qbits == 12:
        orders = [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]),
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [0, 13]])]
    return orders


def n_random_c5_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    rand_order = 0
    if num_qbits == 12:
        rand_order = random.choice(
            [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]),
             np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [0, 13]])])
    return rand_order

def first_10_orders_C5_2local(num_qbits):
    first_10_orders = []
    order=0
    for n in range(10):
        order = n_random_c5_2local_orders(num_qbits,2)
        first_10_orders.append(order)
    return first_10_orders

#c6_2local

@cache
def all_c6_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    orders = []
    if num_qbits == 12:
        orders = [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]),
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [0, 13]])]
    return orders


def n_random_c6_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    rand_order = 0
    if num_qbits == 12:
        rand_order = random.choice(
            [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]),
             np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [0, 13]])])
    return rand_order

def first_10_orders_C6_2local(num_qbits):
    first_10_orders = []
    order=0
    for n in range(10):
        order = n_random_c6_2local_orders(num_qbits,2)
        first_10_orders.append(order)
    return first_10_orders


#cN_2local
@cache
def all_cN_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    orders = []
    if num_qbits == 12:
        orders = [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]),
                  np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [0, 13]])]
    return orders


def n_random_cN_2local_orders(num_qbits: int, chunk_size=2) -> list[np.array]:
    rand_order = 0
    if num_qbits == 12:
        rand_order = random.choice(
            [np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]]),
             np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [0, 13]])])
    return rand_order

def first_10_orders_CN_2local(num_qbits):
    first_10_orders = []
    order=0
    for n in range(10):
        order = n_random_cN_2local_orders(num_qbits,2)
        first_10_orders.append(order)
    return first_10_orders