# RIGEL: Reliability-Informed Graph Engine for Trustworthy Learning

This repository contains the implementation, configuration, and experiment suite for the paper:

> **RIGEL: Reliability-Informed Graph Engine for Trustworthy Learning**
>
> Submitted to IEEE ICDM 2026 (Anonymous Submission)

RIGEL decomposes GNN prediction uncertainty on streaming cryptocurrency transaction graphs into four independently bounded components — structural (missing edges), temporal (data staleness), feature (measurement noise), and interaction (cross-source coupling) — and derives a per-node reliability score enabling three-tier decision routing.

---

## Repository Structure

```
RIGEL/
├── models.py            # Lipschitz-constrained GNN architecture
│                        #   CayleyLinear, GroupSort, LipschitzMessagePassing,
│                        #   RIGELNet, LipschitzRegistry, margin computation
│
├── uncertainty.py       # Uncertainty decomposition framework
│                        #   StructuralUncertaintyAnalyzer  (Theorem 2)
│                        #   TemporalUncertaintyAnalyzer    (Theorem 3)
│                        #   FeatureUncertaintyAnalyzer     (Proposition 1)
│                        #   UncertaintyInteractionAnalyzer (Theorem 4)
│                        #   ReliabilityScorer              (Theorem 5)
│                        #   StreamingReliabilityTracker     (Algorithm 1)
│
├── engine.py            # Streaming engine and data loading
│                        #   StreamingReliabilityEngine, SlidingWindowBuffer,
│                        #   IncrementalAdjacency, KHopNeighborhoodCache,
│                        #   RIGELDataLoader, baseline UQ implementations
│
├── experiments.py       # All 8 experiments + ablation study
│                        #   Experiment 1: Decomposition validation (R²)
│                        #   Experiment 2: Per-node reliability heterogeneity
│                        #   Experiment 3: Reliability-accuracy Pareto frontier
│                        #   Experiment 4: Streaming reliability dynamics
│                        #   Experiment 5: Incompleteness regimes
│                        #   Experiment 6: Three-tier decision routing
│                        #   Experiment 7: UQ method comparison
│                        #   Experiment 8: Scalability analysis
│                        #   Experiment 9: Ablation study
│
├── run.py               # Single reproducible entry point (all modes)
├── config.yaml          # Single source of truth for all hyperparameters
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Requirements

**Hardware used in the paper:**
Four NVIDIA RTX 3090 GPUs (24 GB each), 64-core Intel Xeon Silver 4314 CPU, 384 GB DDR4 RAM, CentOS 7.

**Minimum hardware for reproduction:**
One NVIDIA GPU with at least 12 GB VRAM (for Eth-S and Eth-P). BTC-L requires at least 18 GB VRAM per GPU and 64 GB system RAM.

**Software:**

```bash
# Python 3.9+ required
conda create -n rigel python=3.10 -y
conda activate rigel
pip install -r requirements.txt
```

## Quick Start

**Run all unit tests (no GPU required, ~30 seconds):**

```bash
python run.py --mode test
```

This executes 25 built-in tests across `models.py` (8 tests), `uncertainty.py` (9 tests), and `engine.py` (8 tests), verifying Cayley orthogonality, Lipschitz bounds, GroupSort properties, per-edge Weyl bounds, temporal staleness, feature uncertainty, interaction computation, streaming invariants, K-hop neighborhoods, and window expiration.

**Train RIGEL on a single dataset:**

```bash
python run.py --mode train --dataset eth_s --gpus 0
```

**Train on all four datasets (multi-GPU):**

```bash
python run.py --mode train --dataset all --gpus 0,1,2,3
```

**Run a specific experiment:**

```bash
python run.py --mode experiment --exp 1 --dataset eth_s
```

**Run all experiments and generate paper artifacts:**

```bash
python run.py --mode full --gpus 0,1,2,3
```

**Evaluate a saved checkpoint:**

```bash
python run.py --mode evaluate --dataset eth_s --checkpoint ./checkpoints/best.pt
```

## Datasets

The four cryptocurrency transaction datasets are sourced from Ding et al. (CIKM 2024):

| Dataset | Nodes | Edges | Features | Illicit | Licit | Ratio | Period |
|---------|-------|-------|----------|---------|-------|-------|--------|
| Eth-S | 1.33M | 6.79M | 2 | 1,700 | 1,700 | 1:1.02 | 2020–2023 |
| Eth-P | 2.97M | 13.55M | 2 | 1,200 | 3,400 | 1:2.93 | 2020–2023 |
| BTC-M | 2.51M | 14.18M | 8 | 46.9K | 213K | 1:4.54 | 2012–2018 |
| BTC-L | 20.1M | 203.4M | 8 | 362K | 1.27M | 1:3.51 | 2012–2018 |

Place raw data under `./data/{eth_s,eth_p,btc_m,btc_l}/` with adjacency and feature files. All datasets use temporal splits (70%/15%/15%) to prevent information leakage from future transactions into training data. DeFi exploits, flash loan attacks, and cross-chain attacks are not represented.

## Experiments

All experiments report mean ± standard deviation over five random seeds (42, 43, 44, 45, 46). Deterministic execution is enforced via `torch.backends.cudnn.deterministic=True` and `torch.backends.cudnn.benchmark=False`.

| # | Experiment | Section | Key Metric |
|---|-----------|---------|------------|
| 1 | Decomposition validation | V-C | Structural R² > 0.87 |
| 2 | Per-node heterogeneity | V-D | Spearman ρ = 0.387–0.438 |
| 3 | Reliability-accuracy tradeoff | V-E | ΔF1 = +3.18 to +4.72 |
| 4 | Streaming dynamics | V-F | Reliability-error corr. = −0.847 |
| 5 | Incompleteness regimes | V-G | Monotonic degradation confirmed |
| 6 | Decision routing | V-H | AUTO tier F1 = 97.12 (61.2% cov.) |
| 7 | UQ method comparison | V-I | Best ECE (0.019) in single pass |
| 8 | Scalability analysis | V-J | 4,218 edges/s streaming throughput |
| 9 | Ablation study | V-K | R(v) routing > softmax (p < 0.001) |

To reproduce all results from the paper:

```bash
# Full pipeline: train all datasets, run all 9 experiments, generate figures
python run.py --mode full --gpus 0,1,2,3

# Results are saved to ./results/ with per-experiment JSON files
# Figures are saved to ./figures/ at 300 DPI (IEEE-compatible)
```

## Model Configuration

All hyperparameters are defined in `config.yaml`. Key settings:

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Hidden dimension | 256 | Balances capacity with Lipschitz constraint |
| GNN layers (K) | 3 | Standard depth; L = (α+1)^K = 8.0 |
| Self-loop weight (α) | 1.0 | Ensures ‖αI + Â‖₂ ≤ 2.0 per layer |
| Total Lipschitz constant | 8.0 (exact) | Product of per-layer bounds |
| Activation | GroupSort (group size 2) | Exact 1-Lipschitz, gradient-preserving |
| Weight parameterization | Cayley (square) + Householder (non-square) | Exact orthogonality |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) | Standard for GNNs |
| Loss | Focal CE (γ=2) + 0.1·Margin + 0.01·Lipschitz | Addresses imbalance + certification |
| Batch size | 8,192 (512 × 4 GPU × 4 accum) | Fits in 24 GB with FP16 |
| Early stopping | Patience 30 on val F1_illicit | Prevents overfitting on majority class |
| Routing thresholds | AUTO ≥ 0.8, DEFER < 0.3 | Application-configurable |
| Streaming τ_max | 60 s | Maximum staleness before invalidation |
| Streaming τ_deq | 30 s | Dequeue trigger threshold |

## Theoretical Results

The implementation maps directly to the paper's theoretical framework:

**Theorem 1 (Uncertainty Decomposition):** `uncertainty.py` — `UncertaintyDecomposer.compute_total()`. Total uncertainty decomposes as U_total = U_struct + U_temp + U_feat + U_inter, where each component is a single-source bound computed independently.

**Theorem 2 (Structural Uncertainty):** `uncertainty.py` — `StructuralUncertaintyAnalyzer`. Per-edge Weyl bounds via Eq. 6, aggregated by the triangle inequality, normalized by L.

**Theorem 3 (Temporal Staleness):** `uncertainty.py` — `TemporalUncertaintyAnalyzer`. Poisson-rate exponential amplification Γ(Δt, λ_v) = 1 − exp(−λ_v Δt).

**Proposition 1 (Feature Uncertainty):** `uncertainty.py` — `FeatureUncertaintyAnalyzer`. Lipschitz bound U_feat = L · σ · √d.

**Theorem 4 (Interaction):** `uncertainty.py` — `UncertaintyInteractionAnalyzer`. Cauchy-Schwarz cross-terms weighted by empirical source correlations.

**Theorem 5 (Reliability Score):** `uncertainty.py` — `ReliabilityScorer`. R(v) = m(v) / (m(v) + √2 · U_total(v)).

**Algorithm 1 (Streaming Maintenance):** `uncertainty.py` — `StreamingReliabilityTracker` + `engine.py` — `StreamingReliabilityEngine`.

## Testing

Each module contains a comprehensive test suite invoked via `run.py --mode test`:

```
models.py       — 8 tests: orthogonality, spectral norm, GroupSort,
                   Lipschitz composition, margin, certificate radius,
                   registry consistency, full forward pass
uncertainty.py  — 9 tests: per-edge Weyl, multi-edge aggregation,
                   staleness amplification, feature bound, interaction,
                   reliability score properties, streaming invariants,
                   batch computation, edge cases
engine.py       — 8 tests: sliding window, adjacency updates, K-hop
                   cache, degree tracking, staleness enforcement,
                   incompleteness injection, baseline interfaces,
                   data loader
```

## Reproducibility Checklist

| Item | Status |
|------|--------|
| All hyperparameters specified | ✓ (config.yaml) |
| Random seeds reported | ✓ (42, 43, 44, 45, 46) |
| Deterministic execution | ✓ (cudnn.deterministic=True) |
| Hardware specified | ✓ (4× RTX 3090, Xeon 4314, 384 GB) |
| Statistical significance | ✓ (paired t-test, p-values reported) |
| Baseline tuning | ✓ (all use focal CE γ=2; same data splits) |
| Temporal train/val/test split | ✓ (70/15/15, chronological) |
| Complete algorithm pseudocode | ✓ (Algorithm 1 in paper) |
| Unit tests | ✓ (25 tests across 3 modules) |
| Single entry point | ✓ (run.py) |

## Citation

```bibtex
@inproceedings{anonymous2026rigel,
  title     = {{RIGEL}: Reliability-Informed Graph Engine for Trustworthy Learning},
  author    = {Anonymous},
  booktitle = {Proceedings of the IEEE International Conference on Data Mining (ICDM)},
  year      = {2026},
  note      = {Under review}
}
```

## License

This code is released for academic research purposes under the MIT License. See `LICENSE` for details.
