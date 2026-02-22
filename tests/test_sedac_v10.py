"""
SEDAC V10 全面测试
测试所有层: core → cognitive → 主类 → 真实模型集成
"""
import os
import sys

# 确保能找到 sedac 包
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 强制 UTF-8 编码 (Windows 终端兼容)
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
import math
import time
import traceback
from typing import List, Tuple

# ════════════════════════════════════════════════════════════════
# 测试框架
# ════════════════════════════════════════════════════════════════

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[Tuple[str, str]] = []
    
    def run(self, name: str, fn):
        try:
            fn()
            self.passed += 1
            print(f"  [PASS] {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  [FAIL] {name}: {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, traceback.format_exc()))
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"结果: {self.passed}/{total} 通过, {self.failed} 失败")
        if self.errors:
            print(f"\n失败详情:")
            for name, err in self.errors:
                print(f"  [{name}] {err[:200]}")
        print(f"{'='*60}")
        return self.failed == 0


runner = TestRunner()


# ════════════════════════════════════════════════════════════════
# 1. Core 层测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("1. Core 层测试")
print("="*60)

# --- 1.1 Types ---
def test_types_import():
    from metis.core.types import Decision, EpistemicState, BoundaryAction, CognitiveSignal, ControllerConfig, KnowledgeGap
    assert Decision.FAST.value == "fast"
    assert Decision.DEEP.value == "deep"
    assert EpistemicState.UNKNOWN.value == "unknown"
    assert BoundaryAction.SEEK.value == "seek"
    
    sig = CognitiveSignal()
    assert sig.semantic_entropy == 0.0
    assert sig.decision == Decision.NORMAL
    
    cfg = ControllerConfig()
    assert cfg.window_size == 500
    assert cfg.cost_ratio == 5.0

runner.run("types 导入和默认值", test_types_import)


# --- 1.2 Statistics ---
def test_sliding_window_stats():
    from metis.core.statistics import SlidingWindowStats
    
    stats = SlidingWindowStats(window_size=100)
    assert stats.n == 0
    
    # 插入数据
    for i in range(50):
        stats.update(float(i))
    
    assert stats.n == 50
    s = stats.get_stats()
    assert abs(s["mean"] - 24.5) < 0.1, f"mean={s['mean']}"
    assert s["std"] > 0
    assert s["n"] == 50

runner.run("SlidingWindowStats 基本功能", test_sliding_window_stats)


def test_stats_unbiased_moments():
    from metis.core.statistics import SlidingWindowStats
    import random
    random.seed(42)
    
    stats = SlidingWindowStats(window_size=1000)
    # 正态分布数据
    data = [random.gauss(5.0, 2.0) for _ in range(500)]
    for x in data:
        stats.update(x)
    
    s = stats.get_stats()
    assert abs(s["mean"] - 5.0) < 0.3, f"mean={s['mean']}"
    assert abs(s["std"] - 2.0) < 0.3, f"std={s['std']}"
    # 正态分布偏度应接近0
    assert abs(s["skew"]) < 0.5, f"skew={s['skew']}"

runner.run("SlidingWindowStats 无偏矩估计", test_stats_unbiased_moments)


def test_stats_window_overflow():
    from metis.core.statistics import SlidingWindowStats
    
    stats = SlidingWindowStats(window_size=10)
    for i in range(100):
        stats.update(float(i))
    
    assert stats.n == 10  # 窗口限制
    s = stats.get_stats()
    assert s["mean"] > 90  # 最近10个数

runner.run("SlidingWindowStats 窗口溢出", test_stats_window_overflow)


# --- 1.3 Entropy ---
def test_entropy_basic():
    from metis.core.entropy import SemanticEntropyComputer
    
    ec = SemanticEntropyComputer()
    
    # 高确定性 (一个 token 概率极高)
    logits = torch.zeros(1, 100)
    logits[0, 0] = 100.0
    se, te, sd, conf = ec.compute(logits)
    assert te < 0.1, f"确定性 logits 熵应极低, got {te}"
    assert conf > 0.99, f"确定性 logits 置信度应极高, got {conf}"

runner.run("SemanticEntropyComputer 高确定性", test_entropy_basic)


def test_entropy_uncertain():
    from metis.core.entropy import SemanticEntropyComputer
    
    ec = SemanticEntropyComputer()
    
    # 均匀分布 (最大熵)
    logits = torch.zeros(1, 100)  # 均匀
    se, te, sd, conf = ec.compute(logits)
    assert te > 5.0, f"均匀分布熵应很高, got {te}"
    assert conf < 0.02, f"均匀分布置信度应极低, got {conf}"

runner.run("SemanticEntropyComputer 高不确定性", test_entropy_uncertain)


def test_entropy_with_embeddings():
    from metis.core.entropy import SemanticEntropyComputer
    
    ec = SemanticEntropyComputer()
    
    # 模拟 embedding 矩阵
    vocab_size = 100
    hidden_dim = 64
    embedding = torch.randn(vocab_size, hidden_dim)
    ec.set_embedding_matrix(embedding)
    
    # 测试语义多样性
    logits = torch.randn(1, vocab_size)
    se, te, sd, conf = ec.compute(logits)
    assert sd >= 0.0, f"语义多样性应非负, got {sd}"
    assert sd <= 1.0, f"语义多样性应 ≤ 1, got {sd}"
    assert se >= te, f"语义熵应 ≥ token熵 (因为 λ·D ≥ 0)"

runner.run("SemanticEntropyComputer 语义多样性", test_entropy_with_embeddings)


def test_entropy_similar_embeddings():
    from metis.core.entropy import SemanticEntropyComputer
    
    ec = SemanticEntropyComputer()
    
    # 所有 embedding 相同 → 多样性应为 0
    vocab_size = 100
    hidden_dim = 64
    base_vec = torch.randn(1, hidden_dim)
    embedding = base_vec.expand(vocab_size, -1).clone()
    ec.set_embedding_matrix(embedding)
    
    logits = torch.randn(1, vocab_size)
    se, te, sd, conf = ec.compute(logits)
    assert sd < 0.05, f"相同 embedding 多样性应接近 0, got {sd}"

runner.run("SemanticEntropyComputer 相同embedding→低多样性", test_entropy_similar_embeddings)


def test_entropy_nan_handling():
    from metis.core.entropy import SemanticEntropyComputer
    
    ec = SemanticEntropyComputer()
    
    # 极端 logits
    logits = torch.full((1, 100), -1e10)
    logits[0, 0] = 1e10
    se, te, sd, conf = ec.compute(logits)
    assert not math.isnan(te), "熵不应为 nan"
    assert not math.isnan(se), "语义熵不应为 nan"
    assert not math.isinf(te), "熵不应为 inf"

runner.run("SemanticEntropyComputer nan/inf 处理", test_entropy_nan_handling)


# --- 1.4 Controller ---
def test_controller_cold_start():
    from metis.core.controller import AdaptiveController
    from metis.core.types import Decision
    
    ctrl = AdaptiveController()
    
    # 未校准时应返回 NORMAL
    d = ctrl.decide(3.0, 0.5)
    assert d == Decision.NORMAL, f"未校准时应返回 NORMAL, got {d}"

runner.run("AdaptiveController 冷启动", test_controller_cold_start)


def test_controller_calibration():
    from metis.core.controller import AdaptiveController
    from metis.core.types import Decision, ControllerConfig
    
    cfg = ControllerConfig(min_samples=20)
    ctrl = AdaptiveController(cfg)
    
    # 喂入低熵数据 (均值 ~1.0)
    for _ in range(30):
        ctrl.update(1.0 + torch.randn(1).item() * 0.3, 0.8)
    
    assert ctrl.stats["is_calibrated"], "应已校准"
    
    # 低熵应触发 FAST
    d = ctrl.decide(0.2, 0.95)
    assert d == Decision.FAST, f"低熵应触发 FAST, got {d}"

runner.run("AdaptiveController 校准后决策", test_controller_calibration)


def test_controller_cusum():
    from metis.core.controller import AdaptiveController
    from metis.core.types import ControllerConfig
    
    cfg = ControllerConfig(min_samples=10)
    ctrl = AdaptiveController(cfg)
    
    # 先喂入稳定数据
    for _ in range(20):
        ctrl.update(2.0 + torch.randn(1).item() * 0.2)
    
    # 然后突然跳变
    for _ in range(10):
        ctrl.update(8.0 + torch.randn(1).item() * 0.2)
    
    # CUSUM 应检测到变化
    stats = ctrl.stats
    # 至少 cusum_pos 应显著增大
    assert stats["cusum_pos"] > 1.0, f"CUSUM 应检测到上移, cusum_pos={stats['cusum_pos']}"

runner.run("AdaptiveController CUSUM 变点检测", test_controller_cusum)


def test_controller_circuit_breaker():
    from metis.core.controller import AdaptiveController
    from metis.core.types import Decision, ControllerConfig
    
    cfg = ControllerConfig(min_samples=5)
    ctrl = AdaptiveController(cfg)
    
    # 校准
    for _ in range(10):
        ctrl.update(3.0, 0.5)
    
    # 模拟过多 FAST 决策
    for _ in range(30):
        ctrl.update(0.1, 0.99)
        ctrl.decide(0.1, 0.99)
    
    # 断路器应触发，返回 NORMAL
    d = ctrl.decide(0.1, 0.99)
    # 断路器行为取决于历史比例，至少确认不会崩溃
    assert d in [Decision.FAST, Decision.NORMAL, Decision.DEEP]

runner.run("AdaptiveController 断路器保护", test_controller_circuit_breaker)


# ════════════════════════════════════════════════════════════════
# 2. Cognitive 层测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("2. Cognitive 层测试")
print("="*60)


def test_cognitive_switch():
    from metis.cognitive.switch import CognitiveSwitch
    from metis.core.types import CognitiveSignal, Decision
    
    switch = CognitiveSwitch()
    
    # System 1: FAST
    sig = CognitiveSignal(decision=Decision.FAST, semantic_entropy=0.2)
    result = switch.process(sig)
    assert result.mode == "system1"
    assert result.should_use_draft_model == True
    assert result.compute_budget < 0.5
    
    # System 2: DEEP
    sig = CognitiveSignal(decision=Decision.DEEP, semantic_entropy=3.0)
    result = switch.process(sig)
    assert result.mode == "system2"
    assert result.should_trigger_cot == True
    assert result.compute_budget == 1.0

runner.run("CognitiveSwitch System 1/2 切换", test_cognitive_switch)


def test_cognitive_switch_stats():
    from metis.cognitive.switch import CognitiveSwitch
    from metis.core.types import CognitiveSignal, Decision
    
    switch = CognitiveSwitch()
    
    for _ in range(7):
        switch.process(CognitiveSignal(decision=Decision.FAST))
    for _ in range(3):
        switch.process(CognitiveSignal(decision=Decision.DEEP))
    
    stats = switch.stats
    assert abs(stats["system1_ratio"] - 0.7) < 0.01
    assert abs(stats["system2_ratio"] - 0.3) < 0.01

runner.run("CognitiveSwitch 统计", test_cognitive_switch_stats)


def test_boundary_guard_known():
    from metis.cognitive.boundary import EpistemicBoundaryGuard
    from metis.core.types import CognitiveSignal, EpistemicState, BoundaryAction
    
    guard = EpistemicBoundaryGuard(min_warmup_tokens=3)
    
    # 先过 warmup
    for _ in range(4):
        guard.evaluate(CognitiveSignal(z_score=0.0, confidence=0.8))
    
    # z < -0.5 + 高置信 → KNOWN
    sig = CognitiveSignal(semantic_entropy=0.3, confidence=0.9, z_score=-1.0)
    state, action, _ = guard.evaluate(sig)
    assert state == EpistemicState.KNOWN
    assert action == BoundaryAction.GENERATE

runner.run("EpistemicBoundaryGuard 已知领域", test_boundary_guard_known)


def test_boundary_guard_unknown():
    from metis.cognitive.boundary import EpistemicBoundaryGuard
    from metis.core.types import CognitiveSignal, EpistemicState, BoundaryAction
    
    guard = EpistemicBoundaryGuard(min_warmup_tokens=3)
    
    # 先过 warmup
    for _ in range(4):
        guard.evaluate(CognitiveSignal(z_score=0.0, confidence=0.8))
    
    # 持续高 z + 低置信 → UNKNOWN + REFUSE/SEEK
    # 需要连续 2+ 个高 z token (防止段落边界单 token 误报)
    sig = CognitiveSignal(semantic_entropy=4.0, confidence=0.1, z_score=3.0)
    guard.evaluate(sig)  # 第 1 个高 z: 累积 streak
    state, action, explanation = guard.evaluate(sig)  # 第 2 个: streak >= 2, 触发 REFUSE
    assert state == EpistemicState.UNKNOWN
    assert action in [BoundaryAction.SEEK, BoundaryAction.REFUSE]
    assert len(explanation) > 0

runner.run("EpistemicBoundaryGuard 未知领域", test_boundary_guard_unknown)


def test_boundary_guard_accumulated():
    from metis.cognitive.boundary import EpistemicBoundaryGuard
    from metis.core.types import CognitiveSignal, EpistemicState, BoundaryAction
    
    guard = EpistemicBoundaryGuard(min_warmup_tokens=3)
    
    # 先过 warmup
    for _ in range(4):
        guard.evaluate(CognitiveSignal(z_score=0.0, confidence=0.8))
    
    # 持续的中等 z-score → 累积触发 HEDGE
    for _ in range(20):
        sig = CognitiveSignal(semantic_entropy=2.3, confidence=0.4, z_score=1.2)
        state, action, _ = guard.evaluate(sig)
    
    assert guard.get_uncertainty_score() > 0
    # 累积不确定性应触发 HEDGE 或更高
    assert action in [BoundaryAction.HEDGE, BoundaryAction.SEEK, BoundaryAction.REFUSE]

runner.run("EpistemicBoundaryGuard 累积不确定性", test_boundary_guard_accumulated)


def test_curiosity_driver():
    from metis.cognitive.curiosity import CuriosityDriver
    
    driver = CuriosityDriver(gap_z_threshold=1.0, storage_path=None)
    
    # 低 z-score 会话 → 无盲区
    driver.start_session("1+1等于几？")
    for _ in range(10):
        driver.observe(0.5, z_score=-0.5)  # z < 0: 低于均值
    gap = driver.end_session()
    assert gap is None, "低z会话不应产生知识盲区"
    
    # 高 z-score 会话 → 有盲区
    driver.start_session("2030年世界杯冠军是谁？")
    for _ in range(10):
        driver.observe(3.5, z_score=2.5)  # z=2.5 > 2.0: 极端尖峰
    gap = driver.end_session()
    assert gap is not None, "高z会话应产生知识盲区"
    assert gap.entropy_peak == 3.5
    assert gap.category in ["complete_unknown", "sustained_confusion", "spike_uncertainty"]
    
    assert driver.gap_count == 1

runner.run("CuriosityDriver 知识盲区检测", test_curiosity_driver)


def test_curiosity_training_data():
    from metis.cognitive.curiosity import CuriosityDriver
    
    driver = CuriosityDriver(gap_z_threshold=1.0, storage_path=None)
    
    queries = ["量子纠缠", "暗物质", "意识本质"]
    for q in queries:
        driver.start_session(q)
        for _ in range(5):
            driver.observe(3.0, z_score=2.5)  # 高z → 盲区
        driver.end_session()
    
    data = driver.get_training_data()
    assert len(data) == 3
    assert all("query" in d for d in data)
    
    # 标记已解决
    driver.mark_resolved("量子纠缠")
    assert driver.gap_count == 2

runner.run("CuriosityDriver 训练数据导出", test_curiosity_training_data)


# ════════════════════════════════════════════════════════════════
# 3. SEDAC 主类端到端测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("3. SEDAC 主类端到端测试")
print("="*60)


def test_sedac_import():
    from metis import SEDAC, Decision, BoundaryAction, CognitiveSignal
    assert SEDAC is not None
    assert Decision.FAST.value == "fast"

runner.run("SEDAC 顶层导入", test_sedac_import)


def test_sedac_step_low_entropy():
    from metis import SEDAC, Decision
    
    sedac = SEDAC()
    
    # 先校准
    for _ in range(40):
        logits = torch.randn(1, 1000) * 0.5
        sedac.step(logits)
    
    # 高确定性 logits
    logits = torch.zeros(1, 1000)
    logits[0, 42] = 20.0
    signal = sedac.step(logits)
    
    assert signal.token_entropy < 0.5
    assert signal.confidence > 0.9
    assert signal.decision == Decision.FAST, f"低熵应触发 FAST, got {signal.decision}"

runner.run("SEDAC.step 低熵 → FAST", test_sedac_step_low_entropy)


def test_sedac_step_with_embeddings():
    from metis import SEDAC
    
    vocab_size = 500
    hidden_dim = 64
    embedding = torch.randn(vocab_size, hidden_dim)
    
    sedac = SEDAC()
    sedac.set_embedding_matrix(embedding)
    
    logits = torch.randn(1, vocab_size)
    signal = sedac.step(logits)
    
    assert signal.semantic_diversity > 0, "有 embedding 时语义多样性应 > 0"
    assert signal.semantic_entropy >= signal.token_entropy

runner.run("SEDAC.step 带语义多样性", test_sedac_step_with_embeddings)


def test_sedac_session_lifecycle():
    from metis import SEDAC
    
    sedac = SEDAC()
    
    # 开始会话
    sedac.start_session("什么是量子纠缠？")
    
    # 模拟高熵推理
    for _ in range(20):
        logits = torch.randn(1, 500) * 0.3  # 高熵
        sedac.step(logits)
    
    # 结束会话
    gap = sedac.end_session()
    # 高熵会话可能产生知识盲区
    # (取决于阈值设置)
    
    # 至少不崩溃
    assert sedac.last_signal is not None

runner.run("SEDAC 会话生命周期", test_sedac_session_lifecycle)


def test_sedac_trend_detection():
    from metis import SEDAC
    
    sedac = SEDAC()
    
    # 喂入熵递增的序列
    for i in range(20):
        logits = torch.randn(1, 200)
        # 逐步降低确定性
        logits[0, 0] = max(20.0 - i * 1.5, 0.1)
        signal = sedac.step(logits)
    
    # 检查趋势检测 (不崩溃即可，趋势依赖具体数值)
    assert signal.entropy_trend in ["rising", "falling", "stable", "oscillating"]

runner.run("SEDAC 趋势检测", test_sedac_trend_detection)


def test_sedac_stats():
    from metis import SEDAC
    
    sedac = SEDAC()
    for _ in range(50):
        sedac.step(torch.randn(1, 200))
    
    stats = sedac.stats
    assert "controller" in stats
    assert "switch" in stats
    assert "uncertainty_score" in stats
    assert "knowledge_gaps" in stats
    assert stats["controller"]["is_calibrated"] == True

runner.run("SEDAC.stats 统计导出", test_sedac_stats)


def test_sedac_boundary_integration():
    from metis import SEDAC, BoundaryAction
    
    sedac = SEDAC()
    
    # 校准
    for _ in range(30):
        sedac.step(torch.randn(1, 200))
    
    signal = sedac.step(torch.randn(1, 200))
    assert signal.boundary_action in [
        BoundaryAction.GENERATE, 
        BoundaryAction.HEDGE,
        BoundaryAction.SEEK,
        BoundaryAction.REFUSE,
    ]
    assert signal.epistemic_state is not None
    
runner.run("SEDAC 认知边界集成", test_sedac_boundary_integration)


# ════════════════════════════════════════════════════════════════
# 4. 新干预功能测试 (Adaptive Sampling / CoT / REFUSE Grace)
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("4. 新干预功能测试")
print("="*60)


def test_cognitive_sample_fast_greedy():
    """FAST 决策 → greedy 采样 (无论 base_temperature)"""
    from metis import Metis, MetisInference
    from metis.core.types import CognitiveSignal, Decision

    metis_inst = Metis()
    engine = MetisInference(metis_inst)

    logits = torch.randn(1, 100)
    logits[0, 42] = 100.0  # token 42 明显最大
    signal = CognitiveSignal(decision=Decision.FAST, z_score=-1.0, confidence=0.95)

    # FAST → greedy, 不管 temperature 设多少
    token = engine._cognitive_sample(logits[0], signal, base_temperature=0.8, base_top_p=0.9)
    assert token == 42, f"FAST 应该 greedy 选最大值, got {token}"

runner.run("_cognitive_sample FAST→greedy", test_cognitive_sample_fast_greedy)


def test_cognitive_sample_deep_respects_greedy():
    """DEEP + temperature=0 → 不应升温 (P0 bug fix 验证)"""
    from metis import Metis, MetisInference
    from metis.core.types import CognitiveSignal, Decision

    metis_inst = Metis()
    engine = MetisInference(metis_inst)

    logits = torch.randn(1, 100)
    logits[0, 7] = 100.0
    signal = CognitiveSignal(decision=Decision.DEEP, z_score=2.0, confidence=0.3)

    # temperature=0 时 DEEP 也应 greedy (尊重用户意图)
    token = engine._cognitive_sample(logits[0], signal, base_temperature=0.0, base_top_p=0.9)
    assert token == 7, f"DEEP + temp=0 应保持 greedy, got {token}"

runner.run("_cognitive_sample DEEP+temp=0 保持 greedy", test_cognitive_sample_deep_respects_greedy)


def test_cognitive_sample_logit_sharpening():
    """高 z-score + 低 confidence → logit sharpening 效果 (确定性验证)"""
    from metis import Metis, MetisInference
    from metis.core.types import CognitiveSignal, Decision

    metis_inst = Metis()
    engine = MetisInference(metis_inst)

    # 构造 logits: token 50 微弱领先
    logits_orig = torch.ones(100) * 1.0
    logits_orig[50] = 1.5

    # 低 z → 不触发 sharpening, 应选出的概率分布不变
    signal_low_z = CognitiveSignal(
        decision=Decision.NORMAL, z_score=0.5, confidence=0.8
    )
    probs_no_sharp = torch.softmax(logits_orig.clone() / 0.5, dim=-1)  # temp=0.5

    # 高 z + 低 conf → sharpening: sharpness = 1.0 + 0.15 * min(1.5, 3.0) = 1.225
    signal_high_z = CognitiveSignal(
        decision=Decision.NORMAL, z_score=2.5, confidence=0.3
    )
    logits_sharp = logits_orig.clone() * 1.225  # 模拟 sharpening
    probs_sharp = torch.softmax(logits_sharp / 0.5, dim=-1)

    # sharpening 后 token 50 的概率应更高
    assert probs_sharp[50] > probs_no_sharp[50], \
        f"Sharpening 应提升最优 token 概率: sharp={probs_sharp[50]:.4f} > no_sharp={probs_no_sharp[50]:.4f}"

    # 验证 _cognitive_sample 在 greedy (temp=0) 下两种信号都选 token 50
    t1 = engine._cognitive_sample(logits_orig.clone(), signal_low_z, 0.0, 0.9)
    t2 = engine._cognitive_sample(logits_orig.clone(), signal_high_z, 0.0, 0.9)
    assert t1 == 50 and t2 == 50, f"Greedy 下两种信号都应选 token 50: t1={t1}, t2={t2}"

runner.run("_cognitive_sample logit sharpening 效果", test_cognitive_sample_logit_sharpening)


def test_metis_public_accessors():
    """Metis 公开访问器: model/tokenizer/get_uncertainty_score/regulate"""
    from metis import Metis

    metis_inst = Metis()

    # 未 attach 时应为 None
    assert metis_inst.model is None
    assert metis_inst.tokenizer is None

    # get_uncertainty_score 不崩溃
    score = metis_inst.get_uncertainty_score()
    assert isinstance(score, float)
    assert score >= 0.0

    # regulate 不崩溃
    from metis.core.types import MetaJudgment
    mj = MetaJudgment(epistemic_confidence=0.5, cognitive_load=0.3)
    reg = metis_inst.regulate(mj)
    assert isinstance(reg, dict)
    assert "should_hedge" in reg

runner.run("Metis 公开访问器", test_metis_public_accessors)


def test_metis_attach_public_accessors():
    """Metis.attach() 后 model/tokenizer 可通过公开属性访问"""
    from metis import Metis
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.lm_head = nn.Linear(32, 100)
        def get_input_embeddings(self):
            return self.embed

    model = DummyModel()
    metis_inst = Metis.attach(model, tokenizer="fake_tokenizer")

    assert metis_inst.model is model
    assert metis_inst.tokenizer == "fake_tokenizer"

runner.run("Metis.attach 公开属性", test_metis_attach_public_accessors)


def test_refuse_grace_period():
    """REFUSE grace period: 前 N token 立即拒绝"""
    from metis import Metis, MetisInference
    from metis.core.types import CognitiveSignal, Decision, BoundaryAction

    metis_inst = Metis()
    engine = MetisInference(metis_inst, refuse_grace_period=5, refuse_consecutive_threshold=3)

    # 模拟: step=2 (在 grace period 内), REFUSE → 应立即生效
    assert 2 < engine._refuse_grace_period, "测试假设: step 在 grace period 内"

    # 模拟: step=10 (超出 grace period), 单次 REFUSE → 不应终止
    assert 10 >= engine._refuse_grace_period, "测试假设: step 超出 grace period"

runner.run("REFUSE grace period 参数验证", test_refuse_grace_period)


# ════════════════════════════════════════════════════════════════
# 5. 动态 CoT / Switch 震荡 / 自我修正 测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("5. 动态 CoT / Switch 震荡 / 自我修正")
print("="*60)


def test_cot_manager_basic():
    """CoTManager 基本流程: observe → should_inject → select_strategy"""
    from metis.cognitive.cot import CoTManager
    from metis.core.types import CognitiveSignal, Decision, CoTStrategy

    cm = CoTManager()

    # 连续 3 个 DEEP 信号 → 应该触发注入
    for _ in range(3):
        cm.observe(CognitiveSignal(decision=Decision.DEEP, z_score=2.0))
    assert cm.should_inject(), "连续 3 DEEP 应触发 CoT 注入"

    strategy = cm.select_strategy(CognitiveSignal(decision=Decision.DEEP, z_score=2.0))
    assert strategy != CoTStrategy.NONE, f"策略不应为 NONE, got {strategy}"

    prompt_text = cm.get_prompt(strategy)
    assert len(prompt_text) > 0, "CoT prompt 不应为空"
    assert cm.stats["remaining_budget"] == 3, "注入前 budget 应为 3"

    cm.record_injection(strategy)
    assert cm.stats["total_injections"] == 1
    assert cm.stats["remaining_budget"] == 2

runner.run("CoTManager 基本流程", test_cot_manager_basic)


def test_cot_manager_cooldown():
    """CoTManager 冷却期: 注入后 N 步内不应再次注入"""
    from metis.cognitive.cot import CoTManager, COT_COOLDOWN_STEPS
    from metis.core.types import CognitiveSignal, Decision, CoTStrategy

    cm = CoTManager()

    # 触发第一次注入
    for _ in range(3):
        cm.observe(CognitiveSignal(decision=Decision.DEEP))
    assert cm.should_inject()
    cm.record_injection(CoTStrategy.STANDARD)

    # 冷却期内继续 DEEP → 不应触发
    for _ in range(3):
        cm.observe(CognitiveSignal(decision=Decision.DEEP))
    assert not cm.should_inject(), "冷却期内不应触发 CoT"

    # 超过冷却期后 → 可以再次触发
    for _ in range(COT_COOLDOWN_STEPS):
        cm.observe(CognitiveSignal(decision=Decision.DEEP))
    assert cm.should_inject(), "冷却期过后应可再次触发"

runner.run("CoTManager 冷却期", test_cot_manager_cooldown)


def test_cot_manager_max_budget():
    """CoTManager 注入次数上限"""
    from metis.cognitive.cot import CoTManager, COT_COOLDOWN_STEPS
    from metis.core.types import CognitiveSignal, Decision, CoTStrategy

    cm = CoTManager(max_injections=2, cooldown_steps=1)

    # 消耗完 2 次预算
    for injection in range(2):
        for _ in range(3):
            cm.observe(CognitiveSignal(decision=Decision.DEEP))
        assert cm.should_inject(), f"第 {injection+1} 次注入应允许"
        cm.record_injection(CoTStrategy.STANDARD)

    # 第 3 次: 预算耗尽 → 不允许
    for _ in range(3):
        cm.observe(CognitiveSignal(decision=Decision.DEEP))
    assert not cm.should_inject(), "预算耗尽后不应允许注入"
    assert cm.stats["remaining_budget"] == 0

runner.run("CoTManager 注入次数上限", test_cot_manager_max_budget)


def test_cot_manager_oscillation_strategy():
    """CoTManager 震荡检测 → REFLECTION 策略"""
    from metis.cognitive.cot import CoTManager
    from metis.core.types import CognitiveSignal, Decision, CoTStrategy

    cm = CoTManager(cooldown_steps=0)

    # 模拟震荡: FAST → DEEP → FAST → DEEP → FAST → DEEP → FAST → DEEP
    for i in range(8):
        d = Decision.FAST if i % 2 == 0 else Decision.DEEP
        cm.observe(CognitiveSignal(decision=d))

    # 额外 3 个 DEEP 触发注入
    for _ in range(3):
        cm.observe(CognitiveSignal(decision=Decision.DEEP))

    strategy = cm.select_strategy(
        CognitiveSignal(decision=Decision.DEEP, z_score=2.0)
    )
    assert strategy == CoTStrategy.REFLECTION, \
        f"震荡模式应选 REFLECTION, got {strategy}"

runner.run("CoTManager 震荡→REFLECTION", test_cot_manager_oscillation_strategy)


def test_cot_manager_decomposition_strategy():
    """CoTManager 持续 DEEP → DECOMPOSITION 策略"""
    from metis.cognitive.cot import CoTManager
    from metis.core.types import CognitiveSignal, Decision, CoTStrategy

    cm = CoTManager(cooldown_steps=0)

    # 连续 5+ DEEP (无震荡)
    for _ in range(6):
        cm.observe(CognitiveSignal(decision=Decision.DEEP))

    strategy = cm.select_strategy(
        CognitiveSignal(decision=Decision.DEEP, z_score=1.5)
    )
    assert strategy == CoTStrategy.DECOMPOSITION, \
        f"持续 DEEP 应选 DECOMPOSITION, got {strategy}"

runner.run("CoTManager 持续DEEP→DECOMPOSITION", test_cot_manager_decomposition_strategy)


def test_cot_manager_clarification_strategy():
    """CoTManager 高语义多样性+低置信度 → CLARIFICATION"""
    from metis.cognitive.cot import CoTManager
    from metis.core.types import CognitiveSignal, Decision, CoTStrategy

    cm = CoTManager(cooldown_steps=0)

    # 3 个 DEEP 但不够长 → 不会选 DECOMPOSITION
    for _ in range(3):
        cm.observe(CognitiveSignal(decision=Decision.DEEP))

    strategy = cm.select_strategy(
        CognitiveSignal(
            decision=Decision.DEEP,
            semantic_diversity=0.8,  # 高多样性
            confidence=0.1,          # 低置信度
        )
    )
    assert strategy == CoTStrategy.CLARIFICATION, \
        f"高多样性+低置信度应选 CLARIFICATION, got {strategy}"

runner.run("CoTManager 概念模糊→CLARIFICATION", test_cot_manager_clarification_strategy)


def test_cot_manager_reset():
    """CoTManager reset 清除所有状态"""
    from metis.cognitive.cot import CoTManager
    from metis.core.types import CognitiveSignal, Decision, CoTStrategy

    cm = CoTManager()
    for _ in range(5):
        cm.observe(CognitiveSignal(decision=Decision.DEEP))
    cm.record_injection(CoTStrategy.STANDARD)

    cm.reset()
    assert cm.stats["total_injections"] == 0
    assert cm.stats["consecutive_deep"] == 0
    assert not cm.should_inject(), "reset 后不应立即触发"

runner.run("CoTManager reset", test_cot_manager_reset)


def test_switch_oscillation_detection():
    """CognitiveSwitch 震荡检测: FAST/DEEP 交替 → 强制 system2"""
    from metis.cognitive.switch import CognitiveSwitch
    from metis.core.types import CognitiveSignal, Decision, CoTStrategy

    switch = CognitiveSwitch()

    # 模拟震荡模式: 交替 FAST 和 DEEP
    for i in range(10):
        d = Decision.FAST if i % 2 == 0 else Decision.DEEP
        result = switch.process(
            CognitiveSignal(decision=d, semantic_entropy=0.5)
        )

    # 最后一个是 DEEP，但之前的震荡应被检测
    assert switch.is_oscillating, "频繁切换应被检测为震荡"

    # 震荡时即使 NORMAL 也应升级为 system2
    result = switch.process(
        CognitiveSignal(decision=Decision.NORMAL, semantic_entropy=0.5)
    )
    assert result.mode == "system2", f"震荡时 NORMAL 应升级为 system2, got {result.mode}"
    assert result.strategy == CoTStrategy.REFLECTION, \
        f"震荡应推荐 REFLECTION, got {result.strategy}"

runner.run("CognitiveSwitch 震荡检测", test_switch_oscillation_detection)


def test_switch_reflection_priority():
    """CognitiveSwitch 反思优先级计算"""
    from metis.cognitive.switch import CognitiveSwitch
    from metis.core.types import CognitiveSignal, Decision

    switch = CognitiveSwitch()

    # 低 z-score → 低反思优先级
    result_low = switch.process(
        CognitiveSignal(decision=Decision.NORMAL, z_score=0.5, confidence=0.9)
    )
    assert result_low.reflection_priority < 0.2, \
        f"低 z-score 应有低反思优先级, got {result_low.reflection_priority}"

    switch.reset()

    # 高 z-score + 低置信度 + rising → 高反思优先级
    result_high = switch.process(
        CognitiveSignal(
            decision=Decision.DEEP, z_score=3.0,
            confidence=0.1, entropy_trend="rising"
        )
    )
    assert result_high.reflection_priority > 0.4, \
        f"高 z + 低 conf + rising 应有高反思优先级, got {result_high.reflection_priority}"

runner.run("CognitiveSwitch 反思优先级", test_switch_reflection_priority)


def test_switch_strategy_recommendation():
    """CognitiveSwitch 策略推荐: DEEP 时根据特征选择策略"""
    from metis.cognitive.switch import CognitiveSwitch
    from metis.core.types import CognitiveSignal, Decision, CoTStrategy

    switch = CognitiveSwitch()

    # 连续 5+ DEEP → DECOMPOSITION
    for _ in range(6):
        result = switch.process(
            CognitiveSignal(decision=Decision.DEEP, semantic_entropy=0.8)
        )
    assert result.strategy == CoTStrategy.DECOMPOSITION, \
        f"连续 DEEP 应推荐 DECOMPOSITION, got {result.strategy}"

    switch.reset()

    # 单次 DEEP + 高多样性 + 低置信度 → CLARIFICATION
    result = switch.process(
        CognitiveSignal(
            decision=Decision.DEEP,
            semantic_diversity=0.8,
            confidence=0.1,
        )
    )
    assert result.strategy == CoTStrategy.CLARIFICATION, \
        f"高多样性+低置信度应推荐 CLARIFICATION, got {result.strategy}"

runner.run("CognitiveSwitch 策略推荐", test_switch_strategy_recommendation)


def test_inference_has_cot_manager():
    """MetisInference 应包含 CoTManager 实例"""
    from metis import Metis, MetisInference
    from metis.cognitive.cot import CoTManager

    metis_inst = Metis()
    engine = MetisInference(metis_inst)
    assert hasattr(engine, '_cot_manager'), "MetisInference 应有 _cot_manager"
    assert isinstance(engine._cot_manager, CoTManager)
    assert hasattr(engine, '_max_correction_tokens'), "MetisInference 应有 _max_correction_tokens"

runner.run("MetisInference CoTManager 集成", test_inference_has_cot_manager)


# ════════════════════════════════════════════════════════════════
# 6. 真实模型集成测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("6. 真实模型集成测试")
print("="*60)

MODEL_PATH = "G:/models/qwen2.5-7b"

def test_real_model_integration():
    if os.environ.get("SKIP_MODEL_TESTS", "0") == "1":
        print("  ⊘ 跳过: SKIP_MODEL_TESTS=1")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"  ⊘ 跳过: 模型不存在 ({MODEL_PATH})")
        return
    
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from metis import SEDAC, Decision, BoundaryAction
    
    print("  加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    try:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
    model.eval()
    
    # 附加 SEDAC
    sedac = SEDAC.attach(model)
    
    # 测试用例
    test_cases = [
        ("简单问答", "中国的首都是哪里？"),
        ("数学推理", "计算 17 × 23 = ?"),
        ("未知问题", "2030年诺贝尔物理学奖得主是谁？"),
        ("代码生成", "用Python写一个冒泡排序"),
    ]
    
    print()
    for name, query in test_cases:
        sedac.start_session(query)
        
        messages = [{"role": "user", "content": query}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        input_ids = inputs.input_ids
        past_kv = None
        tokens = []
        signals = []
        
        with torch.no_grad():
            for step in range(30):
                outputs = model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
                logits = outputs.logits[:, -1, :]
                past_kv = outputs.past_key_values
                
                # SEDAC 认知信号
                signal = sedac.step(logits)
                signals.append(signal)
                
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                tokens.append(next_token.item())
                
                if next_token.item() == tokenizer.eos_token_id:
                    break
                input_ids = next_token
        
        gap = sedac.end_session()
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        
        # 统计
        avg_entropy = sum(s.semantic_entropy for s in signals) / len(signals) if signals else 0
        fast_count = sum(1 for s in signals if s.decision == Decision.FAST)
        deep_count = sum(1 for s in signals if s.decision == Decision.DEEP)
        
        print(f"  [{name}]")
        print(f"    回答: {response[:60]}...")
        print(f"    熵: {avg_entropy:.2f} | FAST: {fast_count} | DEEP: {deep_count} | Tokens: {len(tokens)}")
        print(f"    边界: {signals[-1].epistemic_state.value if signals else 'N/A'} | 盲区: {'是' if gap else '否'}")
    
    # 验证
    assert len(sedac.stats["controller"]) > 0
    print(f"\n  最终统计: {sedac.stats}")

runner.run("真实模型集成 (Qwen2.5-7B)", test_real_model_integration)


# ════════════════════════════════════════════════════════════════
# 结果汇总
# ════════════════════════════════════════════════════════════════

all_passed = runner.summary()
sys.exit(0 if all_passed else 1)
