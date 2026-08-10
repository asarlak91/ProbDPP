# Reliability-Aware Determinantal Point Processes for Robust Informative Data Selection in Large Language Models

A compact research implementation of **ProbDPP** for selecting diverse LLM context under unreliable data access. This repository provides code for two core experiments:

1. **Known-reliability robustness:** heterogeneous source reliabilities under independent and correlated failures.
2. **Time-varying adaptation:** Fixed, all-history KL-UCB, sliding-window KL-UCB, and Oracle ProbDPP under controlled reliability shifts.

## Experiment 1: known-reliability robustness

`probdpp_hotpotqa_offline.py` evaluates `N=10`, `K=3` HotpotQA context selection with a heterogeneous reliability vector.

Failure modes:

- **Independent:** each source is sampled independently using its own marginal reliability.
- **Correlated:** sources 1--5 and 6--10 share group-level latent uniforms, introducing positive within-group dependence while preserving each source's marginal reliability.

## Experiment 2: time-varying reliability adaptation

`probdpp_time_varying.py` evaluates four reliability shifts while preserving average reliability. The default experiment uses `T=600` rounds, a change point at `t=300`, and '30' seeds.

## Installation

Python 3.9+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The evaluation scripts query a local Ollama server through its HTTP API. Make sure Ollama is installed separately, the service is running, and a model is available under the name `llama3` (or pass a different name with `--model-name`).

The default embedding backend is `sentence-transformers/all-MiniLM-L6-v2`, and BERTScore uses `roberta-large`. These model files may be downloaded on first use.

## Download HotpotQA

The dataset itself is intentionally **not committed** to this repository.

```bash
python Download_HotPotQA.py
```

This creates:

```text
hotpot_dev_distractor_v1.json
```

## Run Experiment 1

```bash
python probdpp_hotpotqa_offline.py \
  --data hotpot_dev_distractor_v1.json \
  --output-dir results/offline \
  --num-questions 1000 \
  --num-seeds 20 \
  --epsilon 0.6
```

## Run Experiment 2

```bash
python probdpp_time_varying_4severity_combined.py \
  --data hotpot_dev_distractor_v1.json \
  --output-dir results/time_varying \
  --num-questions 600 \
  --num-seeds 30 \
  --change-point 300 \
  --epsilon 0.6
```

## Reproducibility notes

- Candidate-to-source assignment is controlled by deterministic seeds.
- Competing methods use the same failure realization within each round.
- Exact subset search is used for `N=10`, `K=3`.
- The included results use `epsilon=0.6`.

