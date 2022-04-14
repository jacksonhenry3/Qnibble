from dataclasses import dataclass
import numpy as np
import pylab as pl
from matplotlib import collections as mc


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
    """Class for keeping track of an item in inventory."""
    nodes: list
    edges: list

    def add_node(self):
        index = max(self.nodes)
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
        lc = mc.LineCollection(lines, linewidths=2)
        fig, ax = pl.subplots()
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)


a = Graph(edges=[], nodes=[0, 1, 2, 3, 4, 5])

def n_vert_m_valent(N,M):
    g = Graph([],[])
    g.add_node()

for n in a.nodes:
    for e in range(5):
        a.add_edge(n, (n + e) % 5)

a.plot()
pl.show()
