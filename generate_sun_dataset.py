import itertools
import pickle as pkl
import random

import torch
from networkx.classes.digraph import DiGraph


def add_node(g: DiGraph) -> int:
    i = g.number_of_nodes()
    g.add_node(i)
    return i


def generate_sunu(g: DiGraph, nds: int, seq: tuple[int, ...]) -> list[int]:
    assert len(seq) == 6
    s = [nds] + [add_node(g) for i in range(5)]
    for i in range(6):
        sp = add_node(g)
        g.add_edge(s[i], s[(i + 1) % 6])
        g.add_edge(s[(i + 1) % 6], s[i])

        g.add_edge(s[i], sp)
        g.add_edge(sp, s[i])
        for j in range(seq[i] - 1):
            nb = add_node(g)
            g.add_edge(sp, nb)
            g.add_edge(nb, sp)
    return s


def generate_suns(save_path: str):
    random.seed(67)

    num_classes = 1
    num_features = 1

    g: DiGraph = DiGraph()

    positive_dsuns = 100
    negative_dsuns = 100

    negative_patterns = list(itertools.product([1, 2, 3, 4, 5, 6, 7, 8], repeat=6))
    negative_patterns.remove((1, 1, 1, 1, 1, 1))

    pos = set()
    centers = set()
    for p in range(positive_dsuns):
        center_node = add_node(g)
        cns = generate_sunu(g, center_node, (1, 1, 1, 1, 1, 1))

        for c in cns:
            pos.add(c)
            centers.add(c)

        for i in range(6):
            ptrn = random.choice(negative_patterns)
            generate_sunu(g, cns[i], ptrn)

    for p in range(negative_dsuns):
        center_node = add_node(g)

        k = random.choice([2, 3, 4, 5, 6, 7, 8])
        k2 = random.choice([2, 3, 4, 5, 6, 7, 8])
        ptrn = random.choice([(1, 1, 1, 1, 1, k), (1, 1, k, 1, 1, k2)])

        cns = generate_sunu(g, center_node, ptrn)
        for c in cns:
            centers.add(c)
        for i in range(6):
            np = random.choice(negative_patterns)
            while np[0] == 1:
                np = random.choice(negative_patterns)
            generate_sunu(g, cns[i], np)

    x = torch.full((g.number_of_nodes(), num_features), float(1))
    labels = []
    ctrs = []
    for v in g.nodes():
        if v in pos:
            labels.append(1)
        else:
            labels.append(0)
        if v in centers:
            ctrs.append(True)
        else:
            ctrs.append(False)

    y = torch.tensor(labels)
    mask = torch.tensor(ctrs)

    g.graph["x"] = x
    g.graph["y"] = y
    g.graph["mask"] = mask

    raw_data = [g]

    with open(save_path, "wb") as f:
        pkl.dump((num_classes, num_features, raw_data), f)


generate_suns("./data/suns.pkl")
