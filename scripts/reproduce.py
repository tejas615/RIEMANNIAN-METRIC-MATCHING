"""Run a full, commented RMM-style reproduction experiment.

This script generates synthetic source/target distribution families, learns a map,
and saves metrics to results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.rmm import fit_metric_map, generate_synthetic_pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Riemannian Metric Matching experiment")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-dists", type=int, default=64)
    parser.add_argument("--dim", type=int, default=6)
    parser.add_argument("--iters", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.03)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1) Create paired synthetic distribution families.
    means_s, covs_s, means_t, covs_t, true_map = generate_synthetic_pair(
        n_dists=args.n_dists,
        dim=args.dim,
        rng=rng,
    )

    # 2) Learn a map that aligns source geometry to target geometry.
    learned_map, history = fit_metric_map(
        means_src=means_s,
        covs_src=covs_s,
        means_tgt=means_t,
        covs_tgt=covs_t,
        iters=args.iters,
        lr=args.lr,
    )

    # 3) Collect easy-to-read metrics.
    map_error = float(np.linalg.norm(learned_map - true_map, ord="fro"))
    metrics = {
        "seed": args.seed,
        "n_dists": args.n_dists,
        "dim": args.dim,
        "iters": args.iters,
        "lr": args.lr,
        "initial_loss": float(history[0]),
        "final_loss": float(history[-1]),
        "map_frobenius_error": map_error,
        "loss_history": [float(x) for x in history],
    }

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "map.npy", learned_map)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Reproduction complete.")
    print(json.dumps({k: v for k, v in metrics.items() if k != "loss_history"}, indent=2))


if __name__ == "__main__":
    main()
