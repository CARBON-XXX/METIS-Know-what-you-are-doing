"""
METIS Preference Pair Generator for DPO/KTO Training

Generates preference pairs from GRPO groups or raw cognitive traces.
Exports in standard formats compatible with TRL, OpenRLHF, and LLaMA-Factory.

Output formats:
═══════════════════════════════════════════════════════════════════
1. DPO pairs:    {"prompt", "chosen", "rejected", "reward_chosen", "reward_rejected"}
2. KTO singles:  {"prompt", "completion", "label"}  (label = True/False)
3. Reward model: {"prompt", "response", "reward"}
═══════════════════════════════════════════════════════════════════

Key insight: cognitive rewards provide CONTINUOUS scores, not binary preferences.
This means we can:
- Filter pairs by reward spread (only train on clear-cut preferences)
- Use reward magnitude as confidence weight in DPO loss
- Generate KTO data by thresholding on absolute reward
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.types import CognitiveTrace
from .rewards import CognitiveRewardComputer, RewardBreakdown, RewardConfig
from .grpo import GRPOGroup, GRPOSample

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single DPO preference pair."""
    prompt: str
    chosen: str
    rejected: str
    reward_chosen: float
    reward_rejected: float
    reward_margin: float            # chosen - rejected (signal strength)
    chosen_breakdown: Dict[str, float] = field(default_factory=dict)
    rejected_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "reward_chosen": round(self.reward_chosen, 4),
            "reward_rejected": round(self.reward_rejected, 4),
            "reward_margin": round(self.reward_margin, 4),
            "chosen_breakdown": self.chosen_breakdown,
            "rejected_breakdown": self.rejected_breakdown,
        }


@dataclass
class KTOSample:
    """A single KTO (Kahneman-Tversky Optimization) sample."""
    prompt: str
    completion: str
    label: bool                     # True = desirable, False = undesirable
    reward: float
    breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "label": self.label,
            "reward": round(self.reward, 4),
            "breakdown": self.breakdown,
        }


@dataclass
class GeneratorConfig:
    """Configuration for preference pair generation."""
    # DPO: minimum reward margin to consider a pair informative
    min_reward_margin: float = 0.05

    # KTO: threshold for desirable / undesirable classification
    kto_desirable_threshold: float = 0.3    # reward > this → desirable
    kto_undesirable_threshold: float = -0.1  # reward < this → undesirable

    # Pair selection strategy
    pair_strategy: str = "best_worst"  # "best_worst" | "all_pairs" | "adjacent"

    # Maximum pairs per group (for "all_pairs" strategy)
    max_pairs_per_group: int = 6


class PreferencePairGenerator:
    """
    Generate DPO preference pairs and KTO samples from GRPO groups.

    Usage:
        gen = PreferencePairGenerator()
        pairs = gen.from_groups(grpo_groups)
        gen.export_dpo(pairs, "dpo_train.jsonl")

        kto_samples = gen.to_kto(grpo_groups)
        gen.export_kto(kto_samples, "kto_train.jsonl")
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self._config = config or GeneratorConfig()

    # ─────────────────────────────────────────────────────
    # DPO Pair Generation
    # ─────────────────────────────────────────────────────

    def from_groups(self, groups: List[GRPOGroup]) -> List[PreferencePair]:
        """
        Generate DPO preference pairs from GRPO groups.

        Args:
            groups: List of ranked GRPO groups

        Returns:
            List of PreferencePair objects
        """
        pairs: List[PreferencePair] = []
        strategy = self._config.pair_strategy

        for group in groups:
            if len(group.samples) < 2:
                continue

            if strategy == "best_worst":
                new_pairs = self._pairs_best_worst(group)
            elif strategy == "all_pairs":
                new_pairs = self._pairs_all(group)
            elif strategy == "adjacent":
                new_pairs = self._pairs_adjacent(group)
            else:
                raise ValueError(f"Unknown pair strategy: {strategy}")

            # Filter by minimum margin
            for p in new_pairs:
                if p.reward_margin >= self._config.min_reward_margin:
                    pairs.append(p)

        logger.info(
            f"[DPO] Generated {len(pairs)} preference pairs "
            f"from {len(groups)} groups (strategy={strategy})"
        )
        return pairs

    def from_traces(
        self,
        prompt: str,
        responses: List[str],
        traces: List[CognitiveTrace],
        reward_config: Optional[RewardConfig] = None,
    ) -> List[PreferencePair]:
        """
        Generate DPO pairs directly from traces (without GRPO groups).

        Computes rewards internally and generates pairs.

        Args:
            prompt: Original prompt
            responses: List of response texts
            traces: List of CognitiveTrace objects
            reward_config: Optional reward configuration

        Returns:
            List of PreferencePair objects
        """
        computer = CognitiveRewardComputer(reward_config)
        scored: List[Tuple[str, RewardBreakdown]] = []

        for resp, trace in zip(responses, traces):
            reward = computer.compute(trace)
            scored.append((resp, reward))

        # Sort by reward (descending)
        scored.sort(key=lambda x: x[1].total, reverse=True)

        pairs: List[PreferencePair] = []
        # Generate best-vs-rest pairs
        best_resp, best_reward = scored[0]
        for resp, reward in scored[1:]:
            margin = best_reward.total - reward.total
            if margin >= self._config.min_reward_margin:
                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=best_resp,
                    rejected=resp,
                    reward_chosen=best_reward.total,
                    reward_rejected=reward.total,
                    reward_margin=margin,
                    chosen_breakdown=best_reward.to_dict(),
                    rejected_breakdown=reward.to_dict(),
                ))

        return pairs

    # ─────────────────────────────────────────────────────
    # KTO Sample Generation
    # ─────────────────────────────────────────────────────

    def to_kto(self, groups: List[GRPOGroup]) -> List[KTOSample]:
        """
        Generate KTO samples from GRPO groups.

        KTO (Kahneman-Tversky Optimization) uses unpaired data:
        each sample is independently labeled as desirable or undesirable.

        Advantage over DPO: doesn't require paired comparisons,
        works with absolute quality thresholds.

        Args:
            groups: List of GRPO groups

        Returns:
            List of KTOSample objects
        """
        samples: List[KTOSample] = []
        cfg = self._config

        for group in groups:
            for s in group.samples:
                r = s.reward.total
                if r >= cfg.kto_desirable_threshold:
                    samples.append(KTOSample(
                        prompt=s.prompt,
                        completion=s.response,
                        label=True,
                        reward=r,
                        breakdown=s.reward.to_dict(),
                    ))
                elif r <= cfg.kto_undesirable_threshold:
                    samples.append(KTOSample(
                        prompt=s.prompt,
                        completion=s.response,
                        label=False,
                        reward=r,
                        breakdown=s.reward.to_dict(),
                    ))
                # Else: ambiguous zone, skip

        desirable = sum(1 for s in samples if s.label)
        undesirable = len(samples) - desirable
        logger.info(
            f"[KTO] Generated {len(samples)} samples "
            f"(desirable={desirable}, undesirable={undesirable})"
        )
        return samples

    # ─────────────────────────────────────────────────────
    # Reward Model Training Data
    # ─────────────────────────────────────────────────────

    def to_reward_model_data(
        self, groups: List[GRPOGroup]
    ) -> List[Dict[str, Any]]:
        """
        Export (prompt, response, reward) triples for reward model training.

        Can be used to distill METIS cognitive rewards into a lightweight
        neural reward model for faster inference-time scoring.

        Args:
            groups: List of GRPO groups

        Returns:
            List of dicts with prompt, response, reward, breakdown
        """
        data: List[Dict[str, Any]] = []
        for group in groups:
            for s in group.samples:
                data.append({
                    "prompt": s.prompt,
                    "response": s.response,
                    "reward": round(s.reward.total, 4),
                    "breakdown": s.reward.to_dict(),
                })
        return data

    # ─────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────

    @staticmethod
    def export_dpo(pairs: List[PreferencePair], path: str) -> None:
        """Export DPO pairs to JSONL (TRL / OpenRLHF compatible)."""
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"[DPO] Exported {len(pairs)} pairs to {path}")

    @staticmethod
    def export_kto(samples: List[KTOSample], path: str) -> None:
        """Export KTO samples to JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        logger.info(f"[KTO] Exported {len(samples)} samples to {path}")

    @staticmethod
    def export_reward_data(data: List[Dict[str, Any]], path: str) -> None:
        """Export reward model training data to JSONL."""
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        logger.info(f"[RM] Exported {len(data)} samples to {path}")

    # ─────────────────────────────────────────────────────
    # Internal: Pair selection strategies
    # ─────────────────────────────────────────────────────

    def _make_pair(
        self, chosen: GRPOSample, rejected: GRPOSample
    ) -> PreferencePair:
        return PreferencePair(
            prompt=chosen.prompt,
            chosen=chosen.response,
            rejected=rejected.response,
            reward_chosen=chosen.reward.total,
            reward_rejected=rejected.reward.total,
            reward_margin=chosen.reward.total - rejected.reward.total,
            chosen_breakdown=chosen.reward.to_dict(),
            rejected_breakdown=rejected.reward.to_dict(),
        )

    def _pairs_best_worst(self, group: GRPOGroup) -> List[PreferencePair]:
        """Generate single pair: best vs worst in group."""
        if group.best and group.worst and group.best is not group.worst:
            return [self._make_pair(group.best, group.worst)]
        return []

    def _pairs_all(self, group: GRPOGroup) -> List[PreferencePair]:
        """Generate all ordered pairs (capped by max_pairs_per_group)."""
        pairs: List[PreferencePair] = []
        samples = group.samples
        max_pairs = self._config.max_pairs_per_group

        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                if len(pairs) >= max_pairs:
                    return pairs
                pairs.append(self._make_pair(samples[i], samples[j]))

        return pairs

    def _pairs_adjacent(self, group: GRPOGroup) -> List[PreferencePair]:
        """Generate adjacent pairs: rank 0 vs 1, 1 vs 2, etc."""
        pairs: List[PreferencePair] = []
        samples = group.samples

        for i in range(len(samples) - 1):
            pairs.append(self._make_pair(samples[i], samples[i + 1]))

        return pairs
