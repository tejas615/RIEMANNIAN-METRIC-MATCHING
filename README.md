# RIEMANNIAN-METRIC-MATCHING

Reproduction scaffold for the paper:
**"Riemannian Metric Matching for Scalable Geometric Modelling of Distributions"**.

This repository now contains a fully commented, end-to-end implementation for a **toy but faithful reproduction pipeline**:

1. Construct synthetic source/target distributions (Gaussian families).
2. Represent each distribution with a mean vector and SPD covariance matrix.
3. Learn a linear map that aligns source and target distributions by minimizing a geometry-aware objective:
   - Euclidean mean mismatch
   - Affine-invariant Riemannian distance between covariance matrices
4. Report convergence statistics and save artifacts for inspection.

> Why this approach?
> The original paper's large-scale datasets/proprietary setup are not included in this repository. This implementation reconstructs the core geometric idea and makes every step transparent and executable.

## Project structure

- `src/rmm.py` — core math and optimization utilities.
- `scripts/reproduce.py` — executable experiment script that regenerates results.
- `results/` — generated metrics and learned map (created after running).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/reproduce.py --seed 7 --n-dists 64 --dim 6 --iters 250
```

## Output

Running the reproduction script writes:

- `results/metrics.json` — optimization trace and summary numbers.
- `results/map.npy` — learned linear transformation matrix.

## Notes

- The code is intentionally heavily commented for learning and auditability.
- You can increase `--n-dists`, `--dim`, and `--iters` to stress-test scalability.
