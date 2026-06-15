#!/usr/bin/env python3
"""Compression benchmark using synthetic random vectors (no dataset download needed).

Measures TQ index file size vs raw FP32 across dim/bit_width combinations.
Compression ratio is geometry-driven (n × dim × bit_width / 8) so synthetic
vectors give the same ratios as real embeddings.
"""
import json
import os
import tempfile

import numpy as np

from turbovec import TurboQuantIndex

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
N = 100_000
SEED = 42


def measure_index_size(database: np.ndarray, dim: int, bit_width: int) -> int:
    index = TurboQuantIndex(dim, bit_width=bit_width)
    index.add(database)
    with tempfile.NamedTemporaryFile(suffix=".tv", delete=False) as tmp:
        path = tmp.name
    try:
        index.write(path)
        return os.path.getsize(path)
    finally:
        os.remove(path)


def synthetic_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def main():
    configs = [
        ("synthetic_d200",  200),
        ("synthetic_d1536", 1536),
        ("synthetic_d3072", 3072),
    ]

    results = {}
    for name, dim in configs:
        print(f"\nGenerating {N:,} synthetic vectors (dim={dim})...")
        database = synthetic_vectors(N, dim, SEED)
        fp32_mb = N * dim * 4 / (1024 * 1024)

        for bit_width in [2, 3, 4]:
            key = f"{name}_{bit_width}bit"
            print(f"  {key}...", end=" ", flush=True)
            index_bytes = measure_index_size(database, dim, bit_width)
            index_mb = index_bytes / (1024 * 1024)
            ratio = fp32_mb / index_mb
            results[key] = {
                "n": N,
                "dim": dim,
                "bit_width": bit_width,
                "fp32_mb": round(fp32_mb, 1),
                "index_mb": round(index_mb, 1),
                "ratio": round(ratio, 1),
            }
            print(f"{fp32_mb:.1f} MB → {index_mb:.1f} MB ({ratio:.1f}x)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "compression_synthetic.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
