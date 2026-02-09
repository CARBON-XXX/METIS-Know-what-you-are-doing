"""
METIS Benchmark Evaluation Framework
=====================================

Empirical validation of METIS metacognitive capabilities on standard
hallucination detection and uncertainty calibration benchmarks.

Supported benchmarks:
    - TruthfulQA (Lin et al., 2022): Hallucination detection
    - HaluEval (Li et al., 2023): Hallucination classification
    - SelfAware (Yin et al., 2023): "I don't know" calibration

Usage:
    python -m benchmarks.evaluate \\
        --model_path /path/to/model \\
        --benchmark truthfulqa \\
        --method hybrid \\
        --n_samples 5 \\
        --output_dir benchmarks/results/

Metrics:
    - AUROC: Area under ROC curve (SE as hallucination predictor)
    - Boundary F1: Precision/Recall of epistemic boundary detection
    - ECE: Expected Calibration Error (confidence vs. correctness)
    - Latency: Per-query latency breakdown (sampling / entailment / total)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# =============================================================
# Benchmark Data Loaders
# =============================================================

def load_truthfulqa(split: str = "validation", max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load TruthfulQA dataset (Lin et al., 2022).

    Each item: {"question": str, "best_answer": str, "correct_answers": List[str],
                "incorrect_answers": List[str], "category": str}

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")

    ds = load_dataset("truthful_qa", "generation", split=split)
    items = []
    for row in ds:
        items.append({
            "question": row["question"],
            "best_answer": row["best_answer"],
            "correct_answers": row["correct_answers"],
            "incorrect_answers": row["incorrect_answers"],
            "category": row["category"],
        })
        if max_samples and len(items) >= max_samples:
            break
    return items


def load_halueval(task: str = "qa", max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load HaluEval dataset (Li et al., 2023).

    Each item: {"question": str, "answer": str, "hallucinated": bool}

    Requires: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets library required: pip install datasets")

    ds = load_dataset("pminervini/HaluEval", task, split="data")
    items = []
    for row in ds:
        items.append({
            "question": row.get("question", row.get("knowledge", "")),
            "answer": row.get("chatgpt_answer", row.get("answer", "")),
            "hallucinated": row.get("hallucination", "no").lower() == "yes",
        })
        if max_samples and len(items) >= max_samples:
            break
    return items


# =============================================================
# Metrics
# =============================================================

@dataclass
class BenchmarkMetrics:
    """Aggregated evaluation metrics."""
    benchmark: str = ""
    method: str = ""
    n_queries: int = 0

    # Hallucination detection (SE as predictor)
    auroc: float = 0.0
    auprc: float = 0.0

    # Epistemic boundary detection
    boundary_precision: float = 0.0
    boundary_recall: float = 0.0
    boundary_f1: float = 0.0

    # Calibration
    ece: float = 0.0  # Expected Calibration Error

    # Latency
    avg_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_samples_used: float = 0.0
    early_exit_rate: float = 0.0

    # Per-query details
    per_query: List[Dict] = field(default_factory=list)


def compute_auroc(scores: List[float], labels: List[bool]) -> float:
    """
    Compute AUROC without sklearn dependency.

    Args:
        scores: Predicted scores (higher = more likely positive)
        labels: Ground truth (True = positive)

    Returns:
        AUROC in [0, 1]
    """
    if not scores or not labels:
        return 0.0

    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    tp = 0
    fp = 0
    auc = 0.0
    prev_score = None

    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            pass  # threshold changed
        if label:
            tp += 1
        else:
            fp += 1
            auc += tp  # each FP contributes number of TPs ranked above it

    return auc / (n_pos * n_neg)


def compute_boundary_f1(
    se_values: List[float],
    is_hallucinated: List[bool],
    threshold: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Compute boundary detection F1.

    Prediction: SE > threshold → model predicts "uncertain" (should refuse/hedge)
    Ground truth: is_hallucinated[i] = True → answer is incorrect

    Returns:
        (precision, recall, f1)
    """
    tp = fp = fn = 0
    for se, label in zip(se_values, is_hallucinated):
        predicted_uncertain = se > threshold
        if predicted_uncertain and label:
            tp += 1
        elif predicted_uncertain and not label:
            fp += 1
        elif not predicted_uncertain and label:
            fn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return precision, recall, f1


def compute_ece(
    confidences: List[float],
    correctness: List[bool],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (Guo et al., 2017).

    Measures how well model confidence aligns with actual accuracy.
    Lower is better. ECE = 0 means perfectly calibrated.
    """
    bins = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, correctness):
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, float(correct)))

    ece = 0.0
    total = len(confidences)
    for bin_items in bins:
        if not bin_items:
            continue
        avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
        avg_acc = sum(a for _, a in bin_items) / len(bin_items)
        ece += len(bin_items) / total * abs(avg_conf - avg_acc)

    return ece


# =============================================================
# Main Evaluator
# =============================================================

class MetisBenchmark:
    """
    METIS empirical evaluation harness.

    Runs METIS inference on benchmark datasets and computes
    hallucination detection metrics using semantic entropy as
    the uncertainty signal.
    """

    def __init__(
        self,
        model_path: str,
        method: str = "hybrid",
        n_samples: int = 5,
        device: Optional[str] = None,
    ):
        self._model_path = model_path
        self._method = method
        self._n_samples = n_samples
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None
        self._metis = None
        self._engine = None

    def _ensure_loaded(self) -> None:
        """Lazy-load model and METIS pipeline."""
        if self._engine is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from metis import Metis, MetisInference

        logger.info(f"Loading model: {self._model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            torch_dtype=torch.float16,
            device_map=self._device,
            trust_remote_code=True,
        )
        self._model.eval()

        self._metis = Metis.attach(
            self._model, self._tokenizer,
            se_method=self._method,
            se_n_samples=self._n_samples,
        )
        self._engine = MetisInference(self._metis)
        logger.info("Model and METIS pipeline loaded.")

    def evaluate_truthfulqa(
        self,
        max_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> BenchmarkMetrics:
        """
        Evaluate on TruthfulQA.

        Protocol:
            1. For each question, run METIS generate()
            2. Record semantic entropy (SE) from System 2
            3. Judge correctness by checking if answer matches any correct_answer
            4. Compute AUROC(SE, is_incorrect) — SE should be higher for incorrect answers
        """
        self._ensure_loaded()
        data = load_truthfulqa(max_samples=max_samples)
        logger.info(f"Evaluating TruthfulQA: {len(data)} questions")

        se_values: List[float] = []
        is_incorrect: List[bool] = []
        confidences: List[float] = []
        correctness: List[bool] = []
        latencies: List[float] = []
        samples_used: List[int] = []
        early_exits: List[bool] = []
        per_query: List[Dict] = []

        for i, item in enumerate(data):
            t0 = time.perf_counter()
            result = self._engine.generate(
                prompt=item["question"],
                max_tokens=256,
                temperature=0.0,
                enable_system2=True,
                se_n_samples=self._n_samples,
            )
            latency = (time.perf_counter() - t0) * 1000

            # Extract SE value
            se = 0.0
            n_used = 0
            did_early_exit = False
            if result.semantic_entropy_result is not None:
                se = result.semantic_entropy_result.semantic_entropy
                if result.semantic_entropy_result.latency_profile:
                    lp = result.semantic_entropy_result.latency_profile
                    n_used = lp.n_samples_actual
                    did_early_exit = lp.early_exit

            # Judge correctness (simple substring match)
            answer_lower = result.text.lower().strip()
            correct = any(
                ca.lower().strip() in answer_lower
                for ca in item["correct_answers"]
            )
            incorrect = any(
                ia.lower().strip() in answer_lower
                for ia in item["incorrect_answers"]
            )
            is_wrong = incorrect or not correct

            se_values.append(se)
            is_incorrect.append(is_wrong)
            confidences.append(1.0 - result.uncertainty_score)
            correctness.append(not is_wrong)
            latencies.append(latency)
            samples_used.append(n_used)
            early_exits.append(did_early_exit)

            per_query.append({
                "question": item["question"],
                "answer": result.text[:200],
                "correct": not is_wrong,
                "se": round(se, 4),
                "uncertainty": round(result.uncertainty_score, 4),
                "latency_ms": round(latency, 1),
                "was_hedged": result.was_hedged,
                "was_refused": result.was_refused,
                "early_exit": did_early_exit,
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  [{i+1}/{len(data)}] SE={se:.3f}, correct={not is_wrong}")

        # Compute metrics
        auroc = compute_auroc(se_values, is_incorrect)
        prec, rec, f1 = compute_boundary_f1(se_values, is_incorrect)
        ece = compute_ece(confidences, correctness)

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        metrics = BenchmarkMetrics(
            benchmark="TruthfulQA",
            method=self._method,
            n_queries=len(data),
            auroc=round(auroc, 4),
            boundary_precision=round(prec, 4),
            boundary_recall=round(rec, 4),
            boundary_f1=round(f1, 4),
            ece=round(ece, 4),
            avg_latency_ms=round(sum(latencies) / max(n, 1), 1),
            median_latency_ms=round(sorted_latencies[n // 2] if n else 0, 1),
            p95_latency_ms=round(sorted_latencies[int(n * 0.95)] if n else 0, 1),
            avg_samples_used=round(sum(samples_used) / max(n, 1), 2),
            early_exit_rate=round(sum(early_exits) / max(n, 1), 4),
            per_query=per_query,
        )

        if output_dir:
            self._save_results(metrics, output_dir)

        return metrics

    def evaluate_halueval(
        self,
        task: str = "qa",
        max_samples: Optional[int] = None,
        output_dir: Optional[str] = None,
    ) -> BenchmarkMetrics:
        """
        Evaluate on HaluEval.

        Protocol:
            1. For each question+answer pair, compute SE via METIS
            2. Use SE as hallucination predictor
            3. Compute AUROC(SE, is_hallucinated)
        """
        self._ensure_loaded()
        data = load_halueval(task=task, max_samples=max_samples)
        logger.info(f"Evaluating HaluEval ({task}): {len(data)} samples")

        se_values: List[float] = []
        is_hallucinated: List[bool] = []
        confidences: List[float] = []
        latencies: List[float] = []
        per_query: List[Dict] = []

        for i, item in enumerate(data):
            t0 = time.perf_counter()
            result = self._engine.generate(
                prompt=item["question"],
                max_tokens=256,
                temperature=0.0,
                enable_system2=True,
                se_n_samples=self._n_samples,
            )
            latency = (time.perf_counter() - t0) * 1000

            se = 0.0
            if result.semantic_entropy_result is not None:
                se = result.semantic_entropy_result.semantic_entropy

            se_values.append(se)
            is_hallucinated.append(item["hallucinated"])
            confidences.append(1.0 - result.uncertainty_score)
            latencies.append(latency)

            per_query.append({
                "question": item["question"][:100],
                "hallucinated": item["hallucinated"],
                "se": round(se, 4),
                "uncertainty": round(result.uncertainty_score, 4),
                "latency_ms": round(latency, 1),
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  [{i+1}/{len(data)}] SE={se:.3f}")

        auroc = compute_auroc(se_values, is_hallucinated)
        prec, rec, f1 = compute_boundary_f1(se_values, is_hallucinated)

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        metrics = BenchmarkMetrics(
            benchmark=f"HaluEval-{task}",
            method=self._method,
            n_queries=len(data),
            auroc=round(auroc, 4),
            boundary_precision=round(prec, 4),
            boundary_recall=round(rec, 4),
            boundary_f1=round(f1, 4),
            avg_latency_ms=round(sum(latencies) / max(n, 1), 1),
            median_latency_ms=round(sorted_latencies[n // 2] if n else 0, 1),
            p95_latency_ms=round(sorted_latencies[int(n * 0.95)] if n else 0, 1),
            per_query=per_query,
        )

        if output_dir:
            self._save_results(metrics, output_dir)

        return metrics

    @staticmethod
    def _save_results(metrics: BenchmarkMetrics, output_dir: str) -> None:
        """Save results to JSON."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{metrics.benchmark.lower().replace('-', '_')}_{metrics.method}.json"
        filepath = path / filename

        # Separate summary from per-query details
        summary = {k: v for k, v in asdict(metrics).items() if k != "per_query"}
        full = asdict(metrics)

        # Save summary (human-readable)
        summary_path = path / filename.replace(".json", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save full results
        with open(filepath, "w") as f:
            json.dump(full, f, indent=2)

        logger.info(f"Results saved to {filepath}")

    @staticmethod
    def print_metrics(metrics: BenchmarkMetrics) -> None:
        """Pretty-print benchmark results."""
        print(f"\n{'='*60}")
        print(f"  METIS Benchmark: {metrics.benchmark}")
        print(f"  Method: {metrics.method} | Queries: {metrics.n_queries}")
        print(f"{'='*60}")
        print(f"  Hallucination Detection AUROC:  {metrics.auroc:.4f}")
        print(f"  Boundary F1:                    {metrics.boundary_f1:.4f}")
        print(f"    Precision:                    {metrics.boundary_precision:.4f}")
        print(f"    Recall:                       {metrics.boundary_recall:.4f}")
        if metrics.ece > 0:
            print(f"  ECE (calibration):              {metrics.ece:.4f}")
        print(f"{'─'*60}")
        print(f"  Avg Latency:     {metrics.avg_latency_ms:.0f} ms")
        print(f"  Median Latency:  {metrics.median_latency_ms:.0f} ms")
        print(f"  P95 Latency:     {metrics.p95_latency_ms:.0f} ms")
        if metrics.avg_samples_used > 0:
            print(f"  Avg Samples Used: {metrics.avg_samples_used:.1f}")
            print(f"  Early-Exit Rate:  {metrics.early_exit_rate:.1%}")
        print(f"{'='*60}\n")


# =============================================================
# CLI Entry Point
# =============================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="METIS Benchmark Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to HuggingFace model")
    parser.add_argument("--benchmark", choices=["truthfulqa", "halueval"], default="truthfulqa")
    parser.add_argument("--method", choices=["hybrid", "nli", "embedding"], default="hybrid")
    parser.add_argument("--n_samples", type=int, default=5)
    parser.add_argument("--max_queries", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--output_dir", default="benchmarks/results/")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    bench = MetisBenchmark(
        model_path=args.model_path,
        method=args.method,
        n_samples=args.n_samples,
        device=args.device,
    )

    if args.benchmark == "truthfulqa":
        metrics = bench.evaluate_truthfulqa(
            max_samples=args.max_queries,
            output_dir=args.output_dir,
        )
    elif args.benchmark == "halueval":
        metrics = bench.evaluate_halueval(
            max_samples=args.max_queries,
            output_dir=args.output_dir,
        )

    MetisBenchmark.print_metrics(metrics)


if __name__ == "__main__":
    main()
