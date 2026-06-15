#!/usr/bin/env python3
"""Recall benchmark using synthetic random vectors (no dataset download needed).

Ground truth is brute-force exact inner-product search. Measures recall@k
for bit_width in {2, 3, 4} across a range of dims.
"""
import json
import os

import numpy as np

from turbovec import TurboQuantIndex

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
N = 50_000
N_QUERIES = 500
K = 64
K_VALUES = [1, 2, 4, 8, 16, 32, 64]
SEED = 42


def synthetic_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def recall_at_1_at_k(true_top1: np.ndarray, predicted: np.ndarray, k: int) -> float:
    return float(np.mean([true_top1[i] in predicted[i, :k] for i in range(len(true_top1))]))


def run(dim: int, bit_width: int, database: np.ndarray, queries: np.ndarray) -> dict:
    true_top1 = np.argmax(queries @ database.T, axis=1)

    index = TurboQuantIndex(dim, bit_width=bit_width)
    index.add(database)
    _, tq_ids = index.search(queries, k=K)
    tq_ids = np.array(tq_ids)

    recalls = {str(k): round(recall_at_1_at_k(true_top1, tq_ids, k), 4) for k in K_VALUES}
    return recalls


def main():
    configs = [
        ("synthetic_d200",  200),
        ("synthetic_d1536", 1536),
    ]

    results = {}
    for name, dim in configs:
        print(f"\nGenerating vectors (dim={dim}, N={N:,}, queries={N_QUERIES})...")
        all_vecs = synthetic_vectors(N + N_QUERIES, dim, SEED)
        database = all_vecs[:N]
        queries  = all_vecs[N:]

        for bit_width in [2, 3, 4]:
            key = f"{name}_{bit_width}bit"
            print(f"  {key}...", end=" ", flush=True)
            recalls = run(dim, bit_width, database, queries)
            results[key] = {
                "n": N,
                "n_queries": N_QUERIES,
                "dim": dim,
                "bit_width": bit_width,
                "recalls": recalls,
            }
            print(f"recall@1={recalls['1']:.4f}  recall@4={recalls['4']:.4f}  recall@16={recalls['16']:.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "recall_synthetic.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
