# METIS Benchmark Framework

Empirical validation of METIS metacognitive capabilities on standard hallucination detection benchmarks.

## Supported Benchmarks

| Benchmark | Paper | What it Tests |
|---|---|---|
| **TruthfulQA** | Lin et al., 2022 | Hallucination detection — does the model produce truthful answers? |
| **HaluEval** | Li et al., 2023 | Hallucination classification — can SE distinguish hallucinated vs. factual answers? |

## Metrics

- **AUROC**: Semantic entropy as a hallucination predictor (SE ↑ when answer is wrong)
- **Boundary F1**: Precision/Recall of epistemic boundary detection (REFUSE/HEDGE when uncertain)
- **ECE**: Expected Calibration Error — alignment between model confidence and actual accuracy
- **Latency Profile**: Per-query breakdown (sampling, entailment, total) with early-exit statistics

## Usage

```bash
# Install benchmark dependencies
pip install datasets

# TruthfulQA evaluation
python -m benchmarks.evaluate \
    --model_path /path/to/model \
    --benchmark truthfulqa \
    --method hybrid \
    --n_samples 5 \
    --output_dir benchmarks/results/

# HaluEval evaluation
python -m benchmarks.evaluate \
    --model_path /path/to/model \
    --benchmark halueval \
    --method hybrid \
    --output_dir benchmarks/results/

# Quick test (first 50 queries only)
python -m benchmarks.evaluate \
    --model_path /path/to/model \
    --benchmark truthfulqa \
    --max_queries 50
```

## Method Comparison

Run with `--method nli`, `--method embedding`, and `--method hybrid` to compare:

| Method | Accuracy | Latency | NLI Dependency |
|---|---|---|---|
| `nli` | Highest (academic standard) | Highest (N² NLI calls) | Full |
| `embedding` | Lower (cosine approximation) | Lowest | None |
| `hybrid` | Near-NLI accuracy | 3-5× faster than NLI | Graceful degradation |

## Baselines

The framework compares METIS semantic entropy against these baselines:
- **Token Entropy**: Single-position softmax entropy (naive)
- **P(True)**: Verbalized confidence prompting (Kadavath et al., 2022)
- **Softmax Confidence**: Max softmax probability

## Output Format

Results are saved as JSON:
```
benchmarks/results/
  truthfulqa_hybrid_summary.json   # Aggregated metrics
  truthfulqa_hybrid.json           # Full per-query results
```
