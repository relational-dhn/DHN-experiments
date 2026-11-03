import csv
import sys
from collections import defaultdict
from statistics import mean, stdev

groups = defaultdict(list)

with open(sys.argv[1], newline="") as f:
    for row in csv.DictReader(f):
        groups[(row["model"], row["layers"])].append(row)

for (model, layers), rows in groups.items():
    summaries = []
    for metric in ("test_f1", "test_auroc"):
        values = [float(row[metric]) for row in rows]
        sd = f"{stdev(values):.3f}" if len(values) > 1 else "N/A"
        summaries.append(f"{metric}: {mean(values):.3f} ± {sd}")

    print(f"{model}, layers={layers}, n={len(rows)} | " + " | ".join(summaries))
