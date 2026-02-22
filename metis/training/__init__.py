"""
METIS Training Module — Cognitive Reward Signals for RLHF/GRPO/DPO

Transforms METIS inference-time cognitive signals into training-time
reward signals. Unlike traditional RLHF reward models (LLM-as-judge),
these rewards are information-theoretic and objectively measurable.

Core components:
    - CognitiveRewardComputer: CognitiveTrace → scalar rewards
    - CognitiveGRPO: N-sample generation + cognitive ranking
    - PreferencePairGenerator: Generate DPO preference pairs
"""
from .rewards import CognitiveRewardComputer, RewardBreakdown, RewardConfig
from .grpo import CognitiveGRPO
from .dataset import PreferencePairGenerator
from .trl_adapter import (
    prepare_dpo_dataset,
    prepare_kto_dataset,
    MetisRewardFunction,
)

__all__ = [
    "CognitiveRewardComputer",
    "RewardBreakdown",
    "RewardConfig",
    "CognitiveGRPO",
    "PreferencePairGenerator",
    "prepare_dpo_dataset",
    "prepare_kto_dataset",
    "MetisRewardFunction",
]
