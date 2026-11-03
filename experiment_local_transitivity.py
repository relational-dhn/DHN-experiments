import csv
import time

from torch_geometric.nn.models import GCN, GIN, GraphSAGE

from train import DHN, experiment, seed_everything

dataset = "data/local-transitivity.pkl"

outfile = "results_lt_experiment.csv"
repeats = 3
layers = 3
epochs = 1000
hidden_channels = 32


def output(f, writer, m, l, r, epoch, vauroc, vf1, tauroc, tf1, ds):
    print(f"{m} {l} Layers {epoch} {vauroc:.3f} auroc {vf1:.3f} f1 {tauroc:.3f} auroc {tf1} f1")
    writer.writerow(
        [
            m,
            l,
            r,
            epoch,
            vf1,
            vauroc,
            tf1,
            tauroc,
            ds,
        ]
    )
    f.flush()


with open(outfile, mode="x", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "model",
            "layers",
            "run",
            "epoch",
            "valid_f1",
            "valid_auroc",
            "test_f1",
            "test_auroc",
            "duration_s",
        ]
    )

    for repeat in range(repeats):
        seed_everything(67 + repeat)
        start = time.perf_counter()
        ptrns = [
            "p1",
            "ip1",
            "p2",
            "ip2",
            "v2-1",
            "v2-2",
            "v2-3",
            "v2-4",
            "v2-5",
            "t2-1",
            "t2-2",
            "t2-3",
            "c2",
        ]
        model = DHN(1, hidden_channels, [ptrns] * layers, "sum")
        name = "DHN-all-sum"
        valid_f1, valid_auroc, test_f1, test_auroc, best_epoch = experiment(
            dataset, model, epochs=epochs, lr=0.0003
        )
        end = time.perf_counter()
        output(
            f,
            writer,
            name,
            layers,
            repeat + 1,
            best_epoch,
            valid_auroc,
            valid_f1,
            test_auroc,
            test_f1,
            end - start,
        )

    for repeat in range(repeats):
        seed_everything(67 + repeat)
        start = time.perf_counter()
        ptrns = [
            "p1",
            "ip1",
            "p2",
            "ip2",
            "v2-1",
            "v2-2",
            "v2-3",
            "v2-4",
            "v2-5",
            "t2-1",
            "t2-2",
            "t2-3",
            "c2",
        ]
        model = DHN(1, hidden_channels, [ptrns] * layers, "amax")
        name = "DHN-all-max"
        valid_f1, valid_auroc, test_f1, test_auroc, best_epoch = experiment(
            dataset, model, epochs=epochs, lr=0.0003
        )
        end = time.perf_counter()
        output(
            f,
            writer,
            name,
            layers,
            repeat + 1,
            best_epoch,
            valid_auroc,
            valid_f1,
            test_auroc,
            test_f1,
            end - start,
        )

    for repeat in range(repeats):
        seed_everything(67 + repeat)
        start = time.perf_counter()
        name = "GraphSAGE"
        model = GraphSAGE(
            1,
            hidden_channels,
            layers,
            1,
            flow="target_to_source",
            norm="layer_norm",
            act="LeakyRelu",
            act_kwargs={"negative_slope": 0.1}
        )
        valid_f1, valid_auroc, test_f1, test_auroc, best_epoch = experiment(
            dataset, model, epochs=epochs * 10, lr=0.001
        )
        end = time.perf_counter()
        output(
            f,
            writer,
            name,
            layers,
            repeat + 1,
            best_epoch,
            valid_auroc,
            valid_f1,
            test_auroc,
            test_f1,
            end - start,
        )

    for repeat in range(repeats):
        seed_everything(67 + repeat)

        start = time.perf_counter()
        name = "GCN"
        model = GCN(
            1,
            hidden_channels,
            layers,
            1,
            flow="target_to_source",
            norm="layer_norm",
            act="LeakyRelu",
            act_kwargs={"negative_slope": 0.1}
        )
        valid_f1, valid_auroc, test_f1, test_auroc, best_epoch = experiment(
            dataset, model, epochs=epochs * 10, lr=0.001
        )
        end = time.perf_counter()
        output(
            f,
            writer,
            name,
            layers,
            repeat + 1,
            best_epoch,
            valid_auroc,
            valid_f1,
            test_auroc,
            test_f1,
            end - start,
        )

    for repeat in range(repeats):
        seed_everything(67 + repeat)

        start = time.perf_counter()
        name = "GIN"
        model = GIN(
            1,
            hidden_channels,
            layers,
            1,
            norm="layer_norm",
            train_eps=True,
            flow="target_to_source",
            act="LeakyRelu",
            act_kwargs={"negative_slope": 0.1}
        )
        valid_f1, valid_auroc, test_f1, test_auroc, best_epoch = experiment(
            dataset, model, epochs=epochs * 10, lr=0.001
        )
        end = time.perf_counter()
        output(
            f,
            writer,
            name,
            layers,
            repeat + 1,
            best_epoch,
            valid_auroc,
            valid_f1,
            test_auroc,
            test_f1,
            end - start,
        )
