import csv
import time

from torch_geometric.nn.models import GCN, GIN, GraphSAGE

from train import DHN, HomCountGIN, experiment, seed_everything

dataset = "data/suns.pkl"

outfile = "results_sp_experiment.csv"
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
            "best_epoch",
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
    
        model = DHN(1, hidden_channels, [["p1", "c6"]] * layers, "sum")
        name = "DHN-p1-c6-sum"
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
            act="LeakyReLU",
            act_kwargs={"negative_slope": 0.1}
        )
        valid_f1, valid_auroc, test_f1, test_auroc, best_epoch = experiment(
            dataset, model, epochs=epochs, lr=0.001
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
            act="LeakyReLU",
            act_kwargs={"negative_slope": 0.1}
        )
        valid_f1, valid_auroc, test_f1, test_auroc, best_epoch = experiment(
            dataset, model, epochs=epochs, lr=0.001
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
            act="LeakyReLU",
            act_kwargs={"negative_slope": 0.1}
        )
        valid_f1, valid_auroc, test_f1, test_auroc, best_epoch = experiment(
            dataset, model, epochs=epochs, lr=0.001
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
        name = "HomCount GIN"
        model = HomCountGIN(hidden_channels, ["c6", "p1"], layers)
        valid_f1, valid_auroc, test_f1, test_auroc, best_epoch = experiment(
            dataset, model, epochs=epochs, lr=0.001
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
