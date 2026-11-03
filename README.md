# Experiments for Deep Homomorphism Networks

This repository contains the code and synthetic datasets for the experiments
accompanying the paper “Expressive Power of Deep Homomorphism Networks over Relational
Databases” (accepted at NeurIPS 2026). The two tasks are binary node
classification for local transitivity and the sun property.

## Files

* `train.py`: implementation of DHNs, GINs with homomorphism counts, training
   and evaluation.

* `experiment_local_transitivity.py`: local transitivity experiment.

* `experiment_sun.py`: sun property experiment.

* `process_table.py` summary statistics.

* `generate_sun_dataset.py` and `generate_local_transitivity_dataset.py` scripts
  to generate the synthetic datasets.

* `data/local-transitivity.pkl` and `data/suns.pkl`: the synthetic datasets used
  for the reported experiments.

## Environment

Experiments were performed using python 3.12, pytorch 2.8.0, pytorch-geometric
2.7.0 and scikit-learn 1.6.1. See `pyproject.toml`. Experiments are executed
only on the CPU.

## Reproducing the Experiments

Simply run
```
uv run experiment_local_transitivity.py
```
and
```
uv run experiment_sun.py
```
The scripts output detailed training progress to stdout and record the final
results in `results_lt_experiment.csv` and `results_sp_experiment.csv`
respectively. Then use
```
uv run process_table.py results_lt_experiment.csv 
```
and
```
uv run process_table.py results_sp_experiment.csv 
```
to summarize the data.

# Datasets

| Dataset | Nodes | Edges | Eligible nodes | Eligible positives / negatives |
|---|---:|---:|---:|---:|
| Local transitivity | 4,000 | 34,000 | 4,000 | 1,899 / 2,101 |
| Sun | 41,400 | 85,200 | 1,200 | 600 / 600 |