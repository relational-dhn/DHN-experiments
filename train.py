import pickle as pkl
import random

from copy import deepcopy

import numpy
import torch
import torch.nn.functional as F
from networkx.classes.digraph import DiGraph
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data.data import Data
from torch_geometric.nn.models import GIN
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.utils import from_networkx
from tqdm import tqdm

PATTERNS: dict[str, DiGraph] = {
    "p1": DiGraph([(0, 1)]),
    "ip1": DiGraph([(1, 0)]),
    "p2": DiGraph([(0, 1), (1, 2)]),
    "ip2": DiGraph([(1, 0), (2, 1)]),
    "v2-1": DiGraph([(1, 0), (0, 2)]),
    "v2-2": DiGraph([(1, 0), (2, 0)]),
    "v2-3": DiGraph([(0, 1), (2, 1)]),
    "v2-4": DiGraph([(1, 0), (1, 2)]),
    "v2-5": DiGraph([(0, 1), (0, 2)]),
    "t2-1": DiGraph([(0, 1), (1, 2), (0, 2)]),
    "t2-2": DiGraph([(0, 1), (2, 1), (2, 0)]),
    "t2-3": DiGraph([(1, 0), (2, 1), (2, 0)]),
    "c2": DiGraph([(0, 1), (1, 2), (2, 0)]),
    "c6": DiGraph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]),
}


def all_homomorphisms_int(pattern: DiGraph, nxg: DiGraph, h: list[int]):
    if len(h) == pattern.number_of_nodes():
        for u, v in pattern.edges():
            if (h[u], h[v]) not in nxg.edges():
                return
        yield h
        return

    v = len(h)

    for u in pattern.predecessors(v):
        if u < len(h):
            for z in nxg.successors(h[u]):
                if z not in h:
                    yield from all_homomorphisms_int(pattern, nxg, h + [z])
            return

    for u in pattern.successors(v):
        if u < len(h):
            for z in nxg.predecessors(h[u]):
                if z not in h:
                    yield from all_homomorphisms_int(pattern, nxg, h + [z])
            return

    # Not connected to a mapped node
    for z in nxg.nodes():
        if z not in h:
            yield from all_homomorphisms_int(pattern, nxg, h + [z])


def all_injective_homomorphisms(pattern: DiGraph, nxg: DiGraph):
    for v in nxg.nodes():
        yield from all_homomorphisms_int(pattern, nxg, [v])


def build_mapping(nxg: DiGraph) -> dict[str, torch.Tensor]:
    result = {}

    for name, pattern in PATTERNS.items():
        with tqdm(desc=f"  {name}") as match_pbar:
            matches = []
            for match in all_injective_homomorphisms(pattern, nxg):
                matches.append(match)
                match_pbar.update(1)
            if matches:
                result[name] = torch.tensor(numpy.vstack(matches)).long()
            else:
                result[name] = torch.zeros(size=[0, pattern.number_of_nodes()]).long()
    return result


def seed_everything(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


# Custom Data class that tells pytorch geometric how to handle the custom
# homomorphism match data when batching
class HomData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key.startswith("homs"):
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key.startswith("homs"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class Mu(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, simple: bool):
        super().__init__()
        if simple:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(negative_slope=0.1),
                # nn.Linear(out_dim, out_dim),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(out_dim, out_dim),
            )

    def forward(self, x):
        return self.net(x)


class HomQuery(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pattern_name: str,
        aggregation: str,
        simple: bool = False,
    ):
        super().__init__()
        self.aggregation = aggregation
        self.pattern_name = pattern_name
        self.out_channels = out_channels

        pattern_size = PATTERNS[self.pattern_name].number_of_nodes()

        self.mus = nn.ModuleList(
            Mu(in_channels, out_channels, simple) for i in range(pattern_size)
        )

    def forward(self, x: torch.Tensor, homs: dict[str, torch.Tensor]):
        phoms = homs[self.pattern_name]

        product = torch.ones(size=[phoms.size(0), self.out_channels], device=x.device)
        for i in range(phoms.size(dim=1)):
            product *= self.mus[i](x[phoms[:, i]])

        output = torch.zeros(size=[x.size(dim=0), product.size(dim=1)], device=x.device)

        indexes = phoms[:, 0].unsqueeze(1).expand(-1, product.size(1))

        output.scatter_reduce_(
            0, indexes, product, reduce=self.aggregation, include_self=False
        )

        return output


class DHNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        patterns: list[str],
        aggregation: str = "sum",
        simple: bool = False,
    ):
        super().__init__()

        self.queries = nn.ModuleList(
            HomQuery(in_channels, hidden_channels, pattern, aggregation, simple)
            for pattern in patterns
        )

        if simple:
            self.net = nn.Sequential(
                nn.Linear(
                    in_channels + hidden_channels * len(patterns),
                    out_channels,
                ),
                nn.LeakyReLU(),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(
                    in_channels + hidden_channels * len(patterns),
                    hidden_channels,
                ),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(hidden_channels, hidden_channels),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(hidden_channels, out_channels),
            )

    def forward(self, x, homs):
        xs = [query(x, homs) for query in self.queries]
        x = torch.cat([x] + xs, dim=1)
        return self.net(x)


class DHN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        patterns: list[list[str]],
        aggregation: str = "sum",
        simple: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(
            DHNLayer(
                in_channels,
                hidden_channels,
                hidden_channels,
                patterns[0],
                aggregation,
                simple,
            )
        )

        for ptrns in patterns[1:]:
            self.norms.append(LayerNorm(hidden_channels, mode="node"))
            self.layers.append(
                DHNLayer(
                    hidden_channels,
                    hidden_channels,
                    hidden_channels,
                    ptrns,
                    aggregation,
                    simple,
                )
            )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x, homs):

        for i, layer in enumerate(self.layers):
            if i > 0:
                x = self.norms[i - 1](x)
            x = layer(x, homs)

        node_scores = self.node_mlp(x)  # [num_nodes, 1]

        return node_scores


class HomCountGIN(nn.Module):
    def __init__(self, hidden_channels: int, patterns: list[str], layers: int):
        super().__init__()
        self.patterns = patterns
        self.gin = GIN(
            len(patterns),
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=layers,
            norm="layer_norm",
            train_eps=True,
            flow="target_to_source",
            act="LeakyReLU",
            act_kwargs={"negative_slope": 0.1}
        )

    def forward(self, x, edge_index, homs):
        x = torch.zeros(x.size(0), len(self.patterns), device=x.device)

        for i, p in enumerate(self.patterns):
            matches = homs[p]
            if matches.size(0) > 0:
                # bincount is fastest for pure counting
                x[:, i] = torch.bincount(matches[:, 0], minlength=x.size(0)).float()

        return self.gin(x, edge_index)


def extract_homs(data):
    return {k[5:]: v for k, v in data if k.startswith("homs_")}


def forward_pass(model, data):
    if isinstance(model, DHN):
        homs = extract_homs(data)
        return model(data.x, homs=homs).squeeze(1)
    elif isinstance(model, HomCountGIN):
        homs = extract_homs(data)
        return model(data.x, data.edge_index, homs=homs).squeeze(1)
    else:
        return model(data.x, data.edge_index).squeeze(1)


def evaluate(model: nn.Module, data: HomData, train_mask, valid_mask, pos_weight):
    model.eval()

    with torch.no_grad():
        out = forward_pass(model, data)

        probs = torch.sigmoid(out).cpu()
        pred = (probs > 0.5).long().cpu()
        labels = data.y.cpu()
        results = {}
        for split, mask in [("train", train_mask), ("valid", valid_mask)]:
            results[split] = {
                "loss": F.binary_cross_entropy_with_logits(
                    out[mask].float(),
                    labels[mask].float(),
                    reduction="mean",
                    pos_weight=pos_weight,
                ),
                "acc": (pred[mask] == labels[mask]).float().mean().item(),
                "f1": f1_score(labels[mask], pred[mask], zero_division=0),
                "auroc": roc_auc_score(labels[mask], probs[mask]),
            }

    return results


def experiment(
    dataset: str, model: nn.Module, epochs=1000, lr=0.0001
) -> tuple[float, float, float, float, int]:
    with open(dataset, mode="rb") as f:
        num_classes, num_features, raw_data = pkl.load(f)

    da = []
    for G in raw_data:
        if isinstance(model, DHN) or isinstance(model, HomCountGIN):
            homs = build_mapping(G)
        else:
            homs = {}
        data = from_networkx(G)

        homdata = HomData(
            x=data.x,
            edge_index=data.edge_index,
            y=data.y,
            mask=data.mask,
            **{f"homs_{k}": v for k, v in homs.items()},
        )

        da.append(homdata)

    data = da[0]

    valid_indices = data.mask.nonzero(as_tuple=True)[0]

    train_indices, temp_indices = train_test_split(valid_indices, test_size=0.4, random_state = 67)
    valid_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state = 67)

    train_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)
    valid_mask = torch.zeros(data.x.shape[0], dtype=torch.bool)

    # 60% Train, 20% Test, 20% Valid
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    valid_mask[valid_indices] = True

    # Compute pos_weight = num_negatives / num_positives
    num_pos = data.y[train_mask].sum().float()
    num_neg = train_mask.sum() - num_pos
    pos_weight = num_neg / num_pos

    device = torch.device("cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_valid_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = forward_pass(model, data)
        loss = F.binary_cross_entropy_with_logits(
            out[train_mask].float(), data.y[train_mask].float(), pos_weight=pos_weight
        )
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs -1:
            results = evaluate(model, data, train_mask, valid_mask, pos_weight)

            valid_loss = results["valid"]["loss"].item()
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch + 1
                best_state = deepcopy(model.state_dict())

            print(
                f"Epoch {epoch:4d} | "
                f"Train loss: {results['train']['loss']:.4f}, acc: {results['train']['acc']:.4f} | "
                f"Validation loss: {results['valid']['loss']:.4f}, acc: {results['valid']['acc']:.4f}, f1: {results['valid']['f1']:.4f}, auroc: {results['valid']['auroc']:.4f}"
            )

            # We stop if we reach f1 of 100% on the training and validation sets
            if results["valid"]["f1"] >= 1.000 and results["train"]["f1"] >= 1.000:
                break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was selected.")

    # Evaluate the model selected using validation loss.
    model.load_state_dict(best_state)

    # Evaluate on the test set
    valid_results = evaluate(model, data, train_mask, valid_mask, pos_weight)
    test_results = evaluate(model, data, train_mask, test_mask, pos_weight)
    return (
        valid_results["valid"]["f1"],
        valid_results["valid"]["auroc"],
        test_results["valid"]["f1"],
        test_results["valid"]["auroc"],
        best_epoch
    )
