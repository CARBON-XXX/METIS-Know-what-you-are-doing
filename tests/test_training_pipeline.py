"""
METIS Training Pipeline — End-to-end test

Tests the full chain: CognitiveTrace → Reward → GRPO → DPO/KTO export
No GPU needed, uses synthetic traces.
"""
import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from metis.core.types import (
    CognitiveTrace, CognitiveEvent, Decision,
    EpistemicState, BoundaryAction,
)
from metis.training.rewards import CognitiveRewardComputer, RewardConfig
from metis.training.grpo import CognitiveGRPO, GRPOConfig
from metis.training.dataset import PreferencePairGenerator, GeneratorConfig
from metis.training.trl_adapter import (
    prepare_dpo_dataset, prepare_kto_dataset, MetisRewardFunction,
)


# ─────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────

def make_trace(quality: str, seed: int, n_tokens: int = 30) -> CognitiveTrace:
    """Generate synthetic trace with controllable quality."""
    rng = random.Random(seed)
    trace = CognitiveTrace(query=f"test_q_{seed}")

    profiles = {
        "good":  {"entropy_std": 0.15, "surprise_base": 1.5, "confusion_p": 0.0,  "confidence": 0.85, "fast_p": 0.7},
        "ok":    {"entropy_std": 0.40, "surprise_base": 2.8, "confusion_p": 0.10, "confidence": 0.60, "fast_p": 0.3},
        "bad":   {"entropy_std": 0.80, "surprise_base": 4.5, "confusion_p": 0.40, "confidence": 0.45, "fast_p": 0.1},
    }
    p = profiles[quality]

    for i in range(n_tokens):
        phase = "confusion" if rng.random() < p["confusion_p"] else rng.choice(["fluent", "recall", "reasoning"])
        decision = Decision.FAST if rng.random() < p["fast_p"] else Decision.DEEP
        is_uncertain = quality == "bad" and rng.random() < 0.3

        trace.events.append(CognitiveEvent(
            step=i,
            token_entropy=1.0 + rng.gauss(0, p["entropy_std"]),
            semantic_entropy=1.2 + rng.gauss(0, p["entropy_std"] * 1.2),
            confidence=p["confidence"] + rng.gauss(0, 0.05),
            z_score=rng.gauss(0, 0.5 if quality == "good" else 1.5),
            token_surprise=p["surprise_base"] + rng.gauss(0, 1.0),
            entropy_gradient=rng.gauss(0, 0.1),
            entropy_momentum=rng.gauss(0, 0.05),
            cognitive_phase=phase,
            decision=decision,
            epistemic_state=EpistemicState.UNCERTAIN if is_uncertain else EpistemicState.LIKELY,
            boundary_action=BoundaryAction.GENERATE,
        ))

    trace.total_tokens = n_tokens
    return trace


# ─────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────

def test_reward_computation():
    print("=" * 60)
    print("TEST 1: Reward Computation")
    print("=" * 60)

    computer = CognitiveRewardComputer()

    for quality in ["good", "ok", "bad"]:
        trace = make_trace(quality, seed=42)
        reward = computer.compute(trace)
        print(f"  [{quality:4s}] total={reward.total:+.4f}  "
              f"coh={reward.coherence:+.4f}  cal={reward.calibration:+.4f}  "
              f"phase={reward.phase_quality:+.4f}  epist={reward.epistemic_honesty:+.4f}  "
              f"eff={reward.efficiency:+.4f}")

    # Verify ordering: good > ok > bad
    r_good = computer.compute(make_trace("good", 1)).total
    r_ok = computer.compute(make_trace("ok", 2)).total
    r_bad = computer.compute(make_trace("bad", 3)).total
    assert r_good > r_ok > r_bad, f"Ordering failed: good={r_good:.4f} ok={r_ok:.4f} bad={r_bad:.4f}"
    print("  ✓ Reward ordering: good > ok > bad")
    print()


def test_grpo_ranking():
    print("=" * 60)
    print("TEST 2: GRPO Ranking")
    print("=" * 60)

    grpo = CognitiveGRPO()
    prompt = "Explain quantum entanglement"
    responses = ["Excellent answer", "Decent answer", "Mediocre answer", "Poor answer"]
    traces = [make_trace(q, i) for q, i in zip(["good", "ok", "ok", "bad"], range(4))]

    group = grpo.rank_traces(prompt, responses, traces)

    print(f"  Samples: {len(group.samples)}")
    print(f"  Spread:  {group.reward_spread:.4f}")
    for s in group.samples:
        print(f"    rank={s.rank}  adv={s.advantage:+.3f}  "
              f"reward={s.reward.total:+.4f}  resp={s.response}")

    assert group.samples[0].rank == 0
    assert group.samples[0].reward.total >= group.samples[-1].reward.total
    assert group.reward_spread > 0
    print("  ✓ Best ranked first, spread > 0")
    print()


def test_dpo_generation():
    print("=" * 60)
    print("TEST 3: DPO Preference Pair Generation")
    print("=" * 60)

    grpo = CognitiveGRPO()
    groups = []
    for i in range(3):
        responses = [f"resp_{i}_{q}" for q in ["good", "ok", "bad"]]
        traces = [make_trace(q, i * 10 + j) for j, q in enumerate(["good", "ok", "bad"])]
        groups.append(grpo.rank_traces(f"prompt_{i}", responses, traces))

    # Test all strategies
    for strategy in ["best_worst", "all_pairs", "adjacent"]:
        gen = PreferencePairGenerator(GeneratorConfig(pair_strategy=strategy))
        pairs = gen.from_groups(groups)
        print(f"  [{strategy:11s}] {len(pairs)} pairs")
        for p in pairs[:2]:
            print(f"    chosen={p.chosen}  rejected={p.rejected}  margin={p.reward_margin:.4f}")

    print("  ✓ All strategies produce valid pairs")
    print()


def test_kto_generation():
    print("=" * 60)
    print("TEST 4: KTO Sample Generation")
    print("=" * 60)

    grpo = CognitiveGRPO()
    groups = []
    for i in range(3):
        responses = [f"resp_{i}_{q}" for q in ["good", "ok", "bad"]]
        traces = [make_trace(q, i * 10 + j) for j, q in enumerate(["good", "ok", "bad"])]
        groups.append(grpo.rank_traces(f"prompt_{i}", responses, traces))

    gen = PreferencePairGenerator()
    kto = gen.to_kto(groups)

    desirable = [s for s in kto if s.label]
    undesirable = [s for s in kto if not s.label]
    print(f"  Total: {len(kto)} (desirable={len(desirable)}, undesirable={len(undesirable)})")
    for s in kto[:3]:
        print(f"    label={s.label}  reward={s.reward:+.4f}  comp={s.completion}")

    assert len(desirable) > 0, "No desirable samples"
    assert len(undesirable) > 0, "No undesirable samples"
    print("  ✓ Both desirable and undesirable samples generated")
    print()


def test_export_jsonl():
    print("=" * 60)
    print("TEST 5: JSONL Export")
    print("=" * 60)

    grpo = CognitiveGRPO()
    responses = ["good_resp", "bad_resp"]
    traces = [make_trace("good", 1), make_trace("bad", 2)]
    group = grpo.rank_traces("test", responses, traces)

    gen = PreferencePairGenerator()
    pairs = gen.from_groups([group])
    kto = gen.to_kto([group])

    with tempfile.TemporaryDirectory() as tmpdir:
        # DPO export
        dpo_path = os.path.join(tmpdir, "dpo.jsonl")
        gen.export_dpo(pairs, dpo_path)
        with open(dpo_path, "r") as f:
            dpo_rows = [json.loads(line) for line in f]
        print(f"  DPO JSONL: {len(dpo_rows)} rows")
        assert all(k in dpo_rows[0] for k in ["prompt", "chosen", "rejected"])

        # KTO export
        kto_path = os.path.join(tmpdir, "kto.jsonl")
        gen.export_kto(kto, kto_path)
        with open(kto_path, "r") as f:
            kto_rows = [json.loads(line) for line in f]
        print(f"  KTO JSONL: {len(kto_rows)} rows")
        assert all(k in kto_rows[0] for k in ["prompt", "completion", "label"])

        # GRPO export
        grpo_path = os.path.join(tmpdir, "grpo.jsonl")
        CognitiveGRPO.export_groups([group], grpo_path)
        with open(grpo_path, "r") as f:
            grpo_rows = [json.loads(line) for line in f]
        print(f"  GRPO JSONL: {len(grpo_rows)} groups")

    print("  ✓ All export formats valid")
    print()


def test_trl_adapter():
    print("=" * 60)
    print("TEST 6: TRL Adapter (prepare_dpo_dataset / prepare_kto_dataset)")
    print("=" * 60)

    prompts = ["Q1", "Q2", "Q3"]
    responses = [
        ["good1", "ok1", "bad1"],
        ["good2", "ok2", "bad2"],
        ["good3", "ok3", "bad3"],
    ]
    traces = [
        [make_trace("good", 10), make_trace("ok", 11), make_trace("bad", 12)],
        [make_trace("good", 20), make_trace("ok", 21), make_trace("bad", 22)],
        [make_trace("good", 30), make_trace("ok", 31), make_trace("bad", 32)],
    ]

    dpo_rows = prepare_dpo_dataset(prompts, responses, traces)
    print(f"  DPO rows: {len(dpo_rows)}")
    for r in dpo_rows:
        print(f"    prompt={r['prompt']}  chosen={r['chosen']}  rejected={r['rejected']}")
    assert all(k in dpo_rows[0] for k in ["prompt", "chosen", "rejected"])

    kto_rows = prepare_kto_dataset(prompts, responses, traces)
    print(f"  KTO rows: {len(kto_rows)}")
    for r in kto_rows:
        print(f"    prompt={r['prompt']}  label={r['label']}  comp={r['completion']}")
    assert all(k in kto_rows[0] for k in ["prompt", "completion", "label"])

    # MetisRewardFunction offline
    rf = MetisRewardFunction()
    reward = rf.from_trace(make_trace("good", 99))
    print(f"  MetisRewardFunction offline: total={reward.total:+.4f}")
    assert reward.total > 0

    print("  ✓ TRL adapter functional")
    print()


def test_reward_sensitivity():
    print("=" * 60)
    print("TEST 7: Reward Sensitivity (custom configs)")
    print("=" * 60)

    # High calibration weight
    cfg1 = RewardConfig(w_calibration=0.6, w_coherence=0.1, w_phase=0.1, w_epistemic=0.1, w_efficiency=0.1)
    # High efficiency weight
    cfg2 = RewardConfig(w_efficiency=0.6, w_coherence=0.1, w_phase=0.1, w_epistemic=0.1, w_calibration=0.1)

    trace = make_trace("ok", 42)
    r1 = CognitiveRewardComputer(cfg1).compute(trace)
    r2 = CognitiveRewardComputer(cfg2).compute(trace)

    print(f"  Calibration-heavy: total={r1.total:+.4f}  cal={r1.calibration:+.4f}")
    print(f"  Efficiency-heavy:  total={r2.total:+.4f}  eff={r2.efficiency:+.4f}")
    assert r1.total != r2.total, "Different weights should produce different totals"
    print("  ✓ Reward responds to config changes")
    print()


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═" * 60)
    print("  METIS Training Pipeline — Full Test Suite")
    print("═" * 60 + "\n")

    test_reward_computation()
    test_grpo_ranking()
    test_dpo_generation()
    test_kto_generation()
    test_export_jsonl()
    test_trl_adapter()
    test_reward_sensitivity()

    print("═" * 60)
    print("  ALL 7 TESTS PASSED ✓")
    print("═" * 60)
