from dataclasses import dataclass
import numpy as np
import pylab as pl
from matplotlib import collections as mc
import matplotlib.pyplot as plt
import itertools

# lines = [[(0, 1), (1, 1)], [(2, 3), (3, 3)], [(1, 2), (1, 3)]]
# c = np.array([(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)])
#
# lc = mc.LineCollection(lines, colors=c, linewidths=2)
# fig, ax = pl.subplots()
# ax.add_collection(lc)
# ax.autoscale()
# ax.margins(0.1)


@dataclass
class Graph:
    nodes: list
    edges: list

    def add_node(self):
        if len(self.nodes) == 0:
            self.nodes.append(0)
            return 0
        index = max(self.nodes) + 1
        self.nodes.append(index)
        return index

    def add_edge(self, node1, node2):
        self.edges.append([node1, node2])
        return [node1, node2]

    def plot(self):
        coords = {}
        for node in self.nodes:
            theta = 2 * np.pi / len(self.nodes) * node
            coords[node] = [np.cos(theta), np.sin(theta)]
        lines = []
        for edge in self.edges:
            lines.append([coords[n] for n in edge])
        lc = mc.LineCollection(lines, linewidths=.1, alpha=1)
        fig, ax = pl.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.0)

        return fig, ax


def generate_regular_graph(n, m):
    if n * m % 2 != 0:
        raise ValueError("n * m must be even")

    G = Graph([], [])
    for _ in range(n):
        G.add_node()
    if m % 2 == 0:
        for i in range(n):
            for j in range(-m // 2, m // 2 + 1):
                if j != 0:
                    G.add_edge(i, (i + j) % n)
    else:
        for i in range(n):
            for j in range(-(m - 1) // 2, (m - 1) // 2 + 2):
                if j != 0:
                    G.add_edge(i, (i + j) % n)
    return G


def connections(i, n, m):
    """
    i is the index of the node
    n is the number of nodes
    m is the degree of the graph
    """
    if m % 2 == 0:
        result = list(range((-m // 2 + i), (m // 2 + 1 + i)))
        result.remove(0)
        return result
    else:
        result = list(range((-m // 2 + i), (m // 2 + 2 + i)))
        result.remove(0)
        return result


def all_regular_groups(n, m, g):
    # group differences
    if g > m:
        raise ValueError("g must be less than or equal to m")
    group_differences = []
    if m % 2 == 0:
        for start in range(m//2):
            options = connections(start, n, m)
            # get all groups of size g from options using itertools
            all_group_differences = []
            for group in itertools.combinations(options, g):
                all_group_differences.append(group)
            group_differences.append(all_group_differences)
    else:
        print("buts")
    return group_differences





a = all_regular_groups(8, 6, 4)
print(a)