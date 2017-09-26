import networkx as nx
# import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli


N = 20
p = 0.2


def er_graph(N, p):
    """Generate an er graph."""
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for node1 in G.nodes():
        for node2 in G.nodes():
            if node1 < node2 and bernoulli.rvs(p=p):
                G.add_edge(node1, node2)
    return G


def plot_degree_distribution(G):
    plt.hist(list(G.degree().values()), histtype="step")
    plt.xlabel("Degree $k$")
    plt.ylabel("$P(k)$")
    plt.title("Degree distribution")


G = er_graph(500, 0.08)
plot_degree_distribution(G)
G1 = er_graph(500, 0.08)
plot_degree_distribution(G1)
G2 = er_graph(500, 0.08)
plot_degree_distribution(G2)
# nx.draw(er_graph(30, 0.08), node_size=40, node_color="gray")
plt.show()
