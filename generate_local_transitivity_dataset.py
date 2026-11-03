import pickle as pkl
import random

import networkx as nx
import torch
from networkx import disjoint_union
from networkx.classes.digraph import DiGraph


def generate_t_path(n) -> DiGraph:
    g = nx.DiGraph()

    for i in range(n):
        g.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j)

    return g


def is_locally_transitive(g: DiGraph, v: int) -> bool:

    ns = set(g.neighbors(v))

    for u in ns:
        for z in g.neighbors(u):
            if z not in ns:
                return False
    return True


def generate_graphs(save_path):
    random.seed(67)

    num_classes = 1
    num_features = 1

    g = DiGraph()

    for i in range(200):
        g = disjoint_union(g, generate_t_path(20))

    for j in range(4000):
        edge = random.choice(list(g.edges()))
        g.remove_edge(edge[0], edge[1])

    x = torch.full((g.number_of_nodes(), num_features), float(1))

    labels = []
    p = 0
    n = 0
    for v in g.nodes():
        if is_locally_transitive(g, v):
            labels.append(1)
            p += 1
        else:
            labels.append(0)
            n += 1

    y = torch.tensor(labels)

    print(p)
    print(n)

    g.graph["x"] = x
    g.graph["y"] = y
    g.graph["mask"] = torch.ones(size=(g.number_of_nodes(),))

    raw_data = [g]

    with open(save_path, "wb") as f:
        pkl.dump((num_classes, num_features, raw_data), f)


generate_graphs("./data/local-transitivity.pkl")
