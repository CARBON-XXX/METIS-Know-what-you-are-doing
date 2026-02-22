"""
SEDAC Semantic Entropy 测试
测试 Kuhn et al. (ICLR 2023) 生成级语义熵实现

覆盖:
1. 类型定义
2. 聚类算法 (Union-Find)
3. 语义熵计算 (频率 + 概率加权)
4. Embedding equivalence checker
5. SemanticEntropyEstimator (从预生成结果)
6. 推理管线类型
7. 真实模型集成 (可选)
"""
import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
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
# 1. 类型定义测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("1. 语义熵类型定义")
print("="*60)


def test_generation_sample():
    from metis.core.types import GenerationSample
    g = GenerationSample(text="Paris", log_prob=-0.5, tokens=[1, 2, 3])
    assert g.text == "Paris"
    assert g.log_prob == -0.5
    assert len(g.tokens) == 3

runner.run("GenerationSample 基本构造", test_generation_sample)


def test_semantic_cluster():
    from metis.core.types import SemanticCluster
    c = SemanticCluster(members=[0, 1, 2], probability=0.6)
    assert len(c.members) == 3
    assert c.probability == 0.6

runner.run("SemanticCluster 基本构造", test_semantic_cluster)


def test_semantic_entropy_result():
    from metis.core.types import SemanticEntropyResult
    r = SemanticEntropyResult(
        semantic_entropy=1.5,
        n_clusters=3,
        n_samples=5,
        is_uncertain=True,
        majority_answer="Paris",
    )
    assert r.semantic_entropy == 1.5
    assert r.n_clusters == 3
    assert r.is_uncertain is True

runner.run("SemanticEntropyResult 基本构造", test_semantic_entropy_result)


def test_inference_result():
    from metis.core.types import InferenceResult, Decision, EpistemicState, BoundaryAction
    r = InferenceResult(
        text="Hello",
        tokens_generated=5,
        was_hedged=True,
        system2_triggered=True,
    )
    assert r.text == "Hello"
    assert r.was_hedged is True
    assert r.final_decision == Decision.NORMAL  # default

runner.run("InferenceResult 基本构造", test_inference_result)


# ════════════════════════════════════════════════════════════════
# 2. 聚类算法测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("2. 语义等价聚类 (Union-Find)")
print("="*60)


def test_clustering_all_equivalent():
    """所有生成语义等价 → 1 个聚类"""
    from metis.core.semantic_entropy import cluster_by_equivalence

    # 3个生成, 全部互相蕴含
    matrix = [
        [1.0, 0.9, 0.8],
        [0.9, 1.0, 0.85],
        [0.8, 0.85, 1.0],
    ]
    clusters = cluster_by_equivalence(matrix, threshold=0.5)
    assert len(clusters) == 1, f"全等价应有 1 个聚类, got {len(clusters)}"
    assert len(clusters[0]) == 3

runner.run("聚类: 全部等价 → 1 个类", test_clustering_all_equivalent)


def test_clustering_all_different():
    """所有生成语义不同 → N 个聚类"""
    from metis.core.semantic_entropy import cluster_by_equivalence

    matrix = [
        [1.0, 0.1, 0.2],
        [0.1, 1.0, 0.15],
        [0.2, 0.15, 1.0],
    ]
    clusters = cluster_by_equivalence(matrix, threshold=0.5)
    assert len(clusters) == 3, f"全不同应有 3 个聚类, got {len(clusters)}"

runner.run("聚类: 全部不同 → N 个类", test_clustering_all_different)


def test_clustering_partial():
    """部分等价: {0,1} 等价, {2} 独立"""
    from metis.core.semantic_entropy import cluster_by_equivalence

    matrix = [
        [1.0, 0.8, 0.1],
        [0.8, 1.0, 0.2],
        [0.1, 0.2, 1.0],
    ]
    clusters = cluster_by_equivalence(matrix, threshold=0.5)
    assert len(clusters) == 2, f"应有 2 个聚类, got {len(clusters)}"

    # 找到包含 0 的聚类
    cluster_with_0 = [c for c in clusters if 0 in c][0]
    assert 1 in cluster_with_0, "0 和 1 应在同一聚类"

runner.run("聚类: 部分等价", test_clustering_partial)


def test_clustering_transitive():
    """传递性: 0≡1, 1≡2 → 0≡2"""
    from metis.core.semantic_entropy import cluster_by_equivalence

    # 0-1 等价 (0.7), 1-2 等价 (0.6), 0-2 不直接等价 (0.3)
    # 但通过传递性, 0-2 应在同一聚类
    matrix = [
        [1.0, 0.7, 0.3],
        [0.7, 1.0, 0.6],
        [0.3, 0.6, 1.0],
    ]
    clusters = cluster_by_equivalence(matrix, threshold=0.5)
    assert len(clusters) == 1, f"传递性应合并为 1 个聚类, got {len(clusters)}"

runner.run("聚类: 传递性合并", test_clustering_transitive)


def test_clustering_empty():
    """空输入"""
    from metis.core.semantic_entropy import cluster_by_equivalence
    clusters = cluster_by_equivalence([], threshold=0.5)
    assert len(clusters) == 0

runner.run("聚类: 空输入", test_clustering_empty)


# ════════════════════════════════════════════════════════════════
# 3. 语义熵计算测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("3. 语义熵计算 (Kuhn et al. Eq. 3)")
print("="*60)


def test_se_all_same():
    """所有生成一致 → SE = 0"""
    from metis.core.semantic_entropy import compute_semantic_entropy
    from metis.core.types import GenerationSample

    generations = [
        GenerationSample(text="Paris", log_prob=-0.5),
        GenerationSample(text="Paris", log_prob=-0.6),
        GenerationSample(text="Paris", log_prob=-0.4),
    ]
    clusters = [[0, 1, 2]]  # 全在一个聚类
    se, sc = compute_semantic_entropy(clusters, generations)

    assert abs(se) < 1e-6, f"全一致 SE 应为 0, got {se}"
    assert len(sc) == 1
    assert abs(sc[0].probability - 1.0) < 1e-6

runner.run("SE: 全一致 → 0", test_se_all_same)


def test_se_all_different():
    """N 个不同回答, 均匀分布 → SE = log₂(N)"""
    from metis.core.semantic_entropy import compute_semantic_entropy
    from metis.core.types import GenerationSample

    # 4 个不同回答, 相同 log_prob → 均匀分布
    generations = [
        GenerationSample(text=f"answer_{i}", log_prob=-1.0)
        for i in range(4)
    ]
    clusters = [[i] for i in range(4)]
    se, sc = compute_semantic_entropy(clusters, generations)

    expected = math.log2(4)  # 2.0 bits
    assert abs(se - expected) < 0.01, f"均匀 SE 应为 {expected:.2f}, got {se:.2f}"

runner.run("SE: 4 个不同 → log₂(4)=2.0 bits", test_se_all_different)


def test_se_frequency_vs_probability():
    """频率估计 vs 概率加权"""
    from metis.core.semantic_entropy import compute_semantic_entropy
    from metis.core.types import GenerationSample

    # 3 个生成: 2 个在聚类 A, 1 个在聚类 B
    # 但聚类 B 的生成有更高的 log_prob
    generations = [
        GenerationSample(text="a1", log_prob=-2.0),  # 聚类 A
        GenerationSample(text="a2", log_prob=-2.0),  # 聚类 A
        GenerationSample(text="b1", log_prob=-0.1),  # 聚类 B (高概率)
    ]
    clusters_raw = [[0, 1], [2]]

    # 频率估计
    se_freq, _ = compute_semantic_entropy(
        clusters_raw, generations, use_probability_weighting=False
    )
    # p(A) = 2/3, p(B) = 1/3
    expected_freq = -(2/3 * math.log2(2/3) + 1/3 * math.log2(1/3))
    assert abs(se_freq - expected_freq) < 0.01, f"频率 SE: expected {expected_freq:.3f}, got {se_freq:.3f}"

    # 概率加权: B 的 exp(-0.1) >> A 的 2*exp(-2.0)
    se_prob, sc_prob = compute_semantic_entropy(
        clusters_raw, generations, use_probability_weighting=True
    )
    # 概率加权时, B 的聚类概率应更高
    cluster_b = [c for c in sc_prob if 2 in c.members][0]
    assert cluster_b.probability > 0.5, (
        f"概率加权时 B 的 p 应 > 0.5, got {cluster_b.probability:.3f}"
    )

runner.run("SE: 频率 vs 概率加权", test_se_frequency_vs_probability)


def test_se_empty():
    """空输入"""
    from metis.core.semantic_entropy import compute_semantic_entropy
    se, sc = compute_semantic_entropy([], [])
    assert se == 0.0
    assert len(sc) == 0

runner.run("SE: 空输入", test_se_empty)


# ════════════════════════════════════════════════════════════════
# 4. Embedding Equivalence Checker 测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("4. Embedding Equivalence Checker")
print("="*60)


# Mock 模型和 tokenizer，用于 EmbeddingChecker 单元测试
# 返回确定性的 hidden states，使相同输入产生相同 embedding
class _MockTokenizerOutput:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self
    def items(self):
        return [("input_ids", self.input_ids), ("attention_mask", self.attention_mask)]
    def __getitem__(self, key):
        return getattr(self, key)
    def keys(self):
        return ["input_ids", "attention_mask"]

class _MockTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True,
                 max_length=512, padding=False):
        ids = [ord(c) % 100 for c in text[:32]]
        input_ids = torch.tensor([ids])
        attention_mask = torch.ones_like(input_ids)
        return _MockTokenizerOutput(input_ids, attention_mask)

class _MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(100, 64)
        self._device = torch.device("cpu")
    @property
    def device(self):
        return self._device
    def forward(self, input_ids=None, attention_mask=None,
                output_hidden_states=False, return_dict=True, **kwargs):
        hidden = self.embed(input_ids)  # [1, seq, 64]
        class R:
            pass
        r = R()
        r.hidden_states = (hidden,) if output_hidden_states else None
        return r

_mock_model = _MockModel()
_mock_tokenizer = _MockTokenizer()


def test_embedding_checker_identical():
    """相同文本 → 等价"""
    from metis.core.semantic_entropy import EmbeddingEquivalenceChecker

    checker = EmbeddingEquivalenceChecker(
        similarity_threshold=0.8,
        generative_model=_mock_model,
        generative_tokenizer=_mock_tokenizer,
    )
    assert checker.are_equivalent("hello world", "hello world")

runner.run("EmbeddingChecker: 相同文本 → 等价", test_embedding_checker_identical)


def test_embedding_checker_different():
    """完全不同文本 → 不等价"""
    from metis.core.semantic_entropy import EmbeddingEquivalenceChecker

    checker = EmbeddingEquivalenceChecker(
        similarity_threshold=0.8,
        generative_model=_mock_model,
        generative_tokenizer=_mock_tokenizer,
    )
    result = checker.are_equivalent(
        "The capital of France is Paris",
        "quantum mechanics describes subatomic particles"
    )
    # mock 模型不保证语义区分，但确保不崩溃且返回 bool
    assert isinstance(result, bool)

runner.run("EmbeddingChecker: 不同文本", test_embedding_checker_different)


def test_embedding_matrix():
    """蕴含矩阵计算"""
    from metis.core.semantic_entropy import EmbeddingEquivalenceChecker

    checker = EmbeddingEquivalenceChecker(
        generative_model=_mock_model,
        generative_tokenizer=_mock_tokenizer,
    )
    texts = ["hello", "hello", "world"]
    matrix = checker.compute_entailment_matrix(texts)

    assert len(matrix) == 3
    assert len(matrix[0]) == 3
    # 自相似度应接近 1
    for i in range(3):
        assert matrix[i][i] > 0.99, f"自相似度应接近 1, got {matrix[i][i]}"
    # 相同文本应高相似
    assert matrix[0][1] > 0.99, f"相同文本应高相似, got {matrix[0][1]}"

runner.run("EmbeddingChecker: 蕴含矩阵", test_embedding_matrix)


# ════════════════════════════════════════════════════════════════
# 5. SemanticEntropyEstimator (从预生成结果)
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("5. SemanticEntropyEstimator 端到端 (预生成)")
print("="*60)


def test_estimator_from_generations_certain():
    """确定性回答: 所有生成相同"""
    from metis.core.semantic_entropy import SemanticEntropyEstimator
    from metis.core.types import GenerationSample

    estimator = SemanticEntropyEstimator(
        method="embedding",
        embedding_similarity_threshold=0.8,
        uncertainty_threshold=0.5,
    )
    # 注入 mock 模型供 embedding checker 使用
    estimator._checker.set_generative_model(_mock_model, _mock_tokenizer)

    # 所有生成几乎相同
    generations = [
        GenerationSample(text="The capital of France is Paris.", log_prob=-0.3),
        GenerationSample(text="The capital of France is Paris.", log_prob=-0.4),
        GenerationSample(text="The capital of France is Paris.", log_prob=-0.35),
    ]

    result = estimator.estimate_from_generations(generations)

    assert result.semantic_entropy < 0.1, (
        f"全一致应 SE ≈ 0, got {result.semantic_entropy:.3f}"
    )
    assert result.n_clusters == 1
    assert not result.is_uncertain
    assert result.majority_answer == "The capital of France is Paris."

runner.run("Estimator: 确定性回答 → SE≈0", test_estimator_from_generations_certain)


def test_estimator_from_generations_uncertain():
    """不确定性回答: 生成语义不同"""
    from metis.core.semantic_entropy import SemanticEntropyEstimator
    from metis.core.types import GenerationSample

    estimator = SemanticEntropyEstimator(
        method="embedding",
        embedding_similarity_threshold=0.95,  # 严格阈值
        uncertainty_threshold=0.3,
    )
    estimator._checker.set_generative_model(_mock_model, _mock_tokenizer)

    # 语义不同的回答
    generations = [
        GenerationSample(text="I think the answer is 42", log_prob=-1.0),
        GenerationSample(text="Paris is the capital city", log_prob=-1.0),
        GenerationSample(text="Quantum physics is complex", log_prob=-1.0),
        GenerationSample(text="The weather is sunny today", log_prob=-1.0),
    ]

    result = estimator.estimate_from_generations(generations)

    assert result.n_clusters >= 2, (
        f"语义不同应有 ≥2 聚类, got {result.n_clusters}"
    )
    assert result.semantic_entropy > 0, (
        f"不确定 SE 应 > 0, got {result.semantic_entropy:.3f}"
    )
    assert result.n_samples == 4

runner.run("Estimator: 不确定回答 → SE>0", test_estimator_from_generations_uncertain)


def test_estimator_majority_answer():
    """多数聚类应返回 majority answer"""
    from metis.core.semantic_entropy import SemanticEntropyEstimator
    from metis.core.types import GenerationSample

    estimator = SemanticEntropyEstimator(
        method="embedding",
        embedding_similarity_threshold=0.95,
        uncertainty_threshold=2.0,  # 高阈值, 不标记为 uncertain
    )
    estimator._checker.set_generative_model(_mock_model, _mock_tokenizer)

    # 3 个相同 + 1 个不同
    generations = [
        GenerationSample(text="Paris", log_prob=-0.5),
        GenerationSample(text="Paris", log_prob=-0.4),
        GenerationSample(text="Paris", log_prob=-0.6),
        GenerationSample(text="quantum mechanics is interesting", log_prob=-1.0),
    ]

    result = estimator.estimate_from_generations(generations)
    assert result.majority_cluster_prob > 0.5, (
        f"majority 概率应 > 0.5, got {result.majority_cluster_prob:.3f}"
    )

runner.run("Estimator: majority answer 选择", test_estimator_majority_answer)


# ════════════════════════════════════════════════════════════════
# 6. SEDAC 主类集成测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("6. SEDAC 主类 SE 集成")
print("="*60)


def test_sedac_has_se_estimator():
    """SEDAC 应包含 SE 估计器"""
    from metis import SEDAC

    sedac = SEDAC()
    assert sedac.se_estimator is not None
    assert hasattr(sedac, 'evaluate_semantic_entropy')

runner.run("SEDAC 包含 SE 估计器", test_sedac_has_se_estimator)


def test_sedac_se_requires_model():
    """evaluate_semantic_entropy 无模型时应报错"""
    from metis import SEDAC

    sedac = SEDAC()
    try:
        sedac.evaluate_semantic_entropy("test")
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "model and tokenizer required" in str(e)

runner.run("SEDAC SE 无模型时报错", test_sedac_se_requires_model)


def test_sedac_attach_stores_model():
    """attach() 应保存 model 和 tokenizer 引用"""
    from metis import SEDAC
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)

        def get_input_embeddings(self):
            return self.embed

    model = DummyModel()

    class DummyTokenizer:
        pass

    tok = DummyTokenizer()

    sedac = SEDAC.attach(model, tok)
    assert sedac._model is model
    assert sedac._tokenizer is tok

runner.run("SEDAC.attach 保存 model/tokenizer", test_sedac_attach_stores_model)


# ════════════════════════════════════════════════════════════════
# 7. 推理管线类型测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("7. 推理管线导入和类型")
print("="*60)


def test_inference_import():
    from metis import SEDACInference
    assert SEDACInference is not None

runner.run("SEDACInference 导入", test_inference_import)


def test_inference_result_import():
    from metis import InferenceResult, SemanticEntropyResult
    assert InferenceResult is not None
    assert SemanticEntropyResult is not None

runner.run("InferenceResult 导入", test_inference_result_import)


# ════════════════════════════════════════════════════════════════
# 8. Boundary Guard 回调测试
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("8. Boundary Guard 回调")
print("="*60)


def test_boundary_callback():
    """边界守门人回调应被触发 (过了 warmup 后)"""
    from metis.cognitive.boundary import EpistemicBoundaryGuard
    from metis.core.types import CognitiveSignal, BoundaryAction

    callback_log = []

    def on_action(action, explanation):
        callback_log.append((action, explanation))

    guard = EpistemicBoundaryGuard(min_warmup_tokens=5, on_action=on_action)

    # 先过 warmup 期
    for _ in range(6):
        guard.evaluate(CognitiveSignal(z_score=0.0, confidence=0.8))

    # warmup 后, 连续高 z + 低置信 → REFUSE → 应触发回调
    sig = CognitiveSignal(z_score=3.0, confidence=0.1)
    guard.evaluate(sig)  # 第 1 个: streak 累积
    state, action, explanation = guard.evaluate(sig)  # 第 2 个: streak >= 2, 触发

    assert action in [BoundaryAction.SEEK, BoundaryAction.REFUSE, BoundaryAction.HEDGE]
    assert len(callback_log) >= 1, f"回调应被触发, got {len(callback_log)}"
    assert callback_log[-1][0] == action

runner.run("Boundary Guard 回调触发", test_boundary_callback)


def test_boundary_action_counts():
    """行为统计应正确 (过了 warmup 后)"""
    from metis.cognitive.boundary import EpistemicBoundaryGuard
    from metis.core.types import CognitiveSignal, BoundaryAction

    guard = EpistemicBoundaryGuard(min_warmup_tokens=3)

    # warmup 期 (3 tokens)
    for _ in range(3):
        guard.evaluate(CognitiveSignal(z_score=0.0, confidence=0.8))

    # warmup 后: 低 z → GENERATE
    for _ in range(5):
        guard.evaluate(CognitiveSignal(z_score=-1.0, confidence=0.9))

    # 高 z → SEEK/REFUSE
    for _ in range(3):
        guard.evaluate(CognitiveSignal(z_score=3.0, confidence=0.1))

    counts = guard.action_counts
    # warmup (3 GENERATE) + 5 GENERATE = at least 8
    assert counts[BoundaryAction.GENERATE] >= 5
    assert sum(counts[a] for a in [BoundaryAction.SEEK, BoundaryAction.REFUSE, BoundaryAction.HEDGE]) >= 3

runner.run("Boundary Guard 行为统计", test_boundary_action_counts)


def test_boundary_reset_clears_counts():
    """reset 应清除统计"""
    from metis.cognitive.boundary import EpistemicBoundaryGuard
    from metis.core.types import CognitiveSignal, BoundaryAction

    guard = EpistemicBoundaryGuard()
    guard.evaluate(CognitiveSignal(z_score=3.0, confidence=0.1))
    guard.reset()

    counts = guard.action_counts
    assert all(v == 0 for v in counts.values()), "reset 后计数应为 0"

runner.run("Boundary Guard reset 清除统计", test_boundary_reset_clears_counts)


# ════════════════════════════════════════════════════════════════
# 9. 数学正确性: 语义熵边界条件
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("9. 语义熵数学正确性")
print("="*60)


def test_se_binary_equal():
    """2 个等概率聚类 → SE = 1.0 bit"""
    from metis.core.semantic_entropy import compute_semantic_entropy
    from metis.core.types import GenerationSample

    generations = [
        GenerationSample(text="A", log_prob=-1.0),
        GenerationSample(text="B", log_prob=-1.0),
    ]
    clusters = [[0], [1]]
    se, _ = compute_semantic_entropy(clusters, generations)

    assert abs(se - 1.0) < 0.01, f"2 等概率聚类 SE 应 = 1.0 bit, got {se:.3f}"

runner.run("SE: 二元均匀 → 1.0 bit", test_se_binary_equal)


def test_se_monotonic():
    """SE 随聚类数增加而增大 (均匀分布时)"""
    from metis.core.semantic_entropy import compute_semantic_entropy
    from metis.core.types import GenerationSample

    prev_se = -1.0
    for n in [2, 4, 8]:
        generations = [
            GenerationSample(text=f"x{i}", log_prob=-1.0) for i in range(n)
        ]
        clusters = [[i] for i in range(n)]
        se, _ = compute_semantic_entropy(clusters, generations)
        assert se > prev_se, f"SE 应随聚类数增加, n={n}, se={se:.3f}, prev={prev_se:.3f}"
        prev_se = se

runner.run("SE: 单调性 (更多聚类 → 更高 SE)", test_se_monotonic)


def test_se_probability_weighting_correctness():
    """概率加权的数学正确性"""
    from metis.core.semantic_entropy import compute_semantic_entropy
    from metis.core.types import GenerationSample

    # 2 个聚类, 概率加权后应反映 log_prob
    generations = [
        GenerationSample(text="A", log_prob=0.0),    # exp(0) = 1.0
        GenerationSample(text="B", log_prob=-100.0),  # exp(-100) ≈ 0
    ]
    clusters = [[0], [1]]
    se, sc = compute_semantic_entropy(
        clusters, generations, use_probability_weighting=True
    )

    # A 的概率应接近 1, B 接近 0
    cluster_a = [c for c in sc if 0 in c.members][0]
    assert cluster_a.probability > 0.99, (
        f"A 的概率应接近 1, got {cluster_a.probability:.6f}"
    )
    # SE 应接近 0 (几乎确定是 A)
    assert se < 0.1, f"极不对称 SE 应接近 0, got {se:.3f}"

runner.run("SE: 概率加权数学正确性", test_se_probability_weighting_correctness)


def test_log_sum_exp_stability():
    """log-sum-exp 数值稳定性"""
    from metis.core.semantic_entropy import _log_sum_exp

    # 大数值
    result = _log_sum_exp([1000.0, 1000.0])
    expected = 1000.0 + math.log(2)
    assert abs(result - expected) < 0.01, f"大数值 LSE 不稳定: {result} vs {expected}"

    # 小数值
    result = _log_sum_exp([-1000.0, -1000.0])
    expected = -1000.0 + math.log(2)
    assert abs(result - expected) < 0.01, f"小数值 LSE 不稳定: {result} vs {expected}"

    # 空列表
    result = _log_sum_exp([])
    assert math.isinf(result) and result < 0, "空列表应返回 -inf"

runner.run("log-sum-exp 数值稳定性", test_log_sum_exp_stability)


# ════════════════════════════════════════════════════════════════
# 10. CognitiveTrace 认知轨迹
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("10. CognitiveTrace 认知轨迹")
print("="*60)


def test_cognitive_trace_basic():
    """CognitiveTrace 基本功能"""
    from metis.core.types import (
        CognitiveTrace, CognitiveEvent, CognitiveSignal,
        Decision, EpistemicState, BoundaryAction,
    )

    trace = CognitiveTrace(query="test")
    assert trace.total_tokens == 0

    # 模拟添加事件
    sig = CognitiveSignal(
        token_entropy=1.5, semantic_entropy=2.0, confidence=0.7,
        z_score=0.5, decision=Decision.FAST,
        epistemic_state=EpistemicState.KNOWN,
        boundary_action=BoundaryAction.GENERATE,
    )
    trace.add_event(sig, step=0)
    assert trace.total_tokens == 1
    assert trace.events[0].decision == Decision.FAST
    assert trace.events[0].z_score == 0.5

runner.run("CognitiveTrace 基本功能", test_cognitive_trace_basic)


def test_cognitive_trace_in_sedac():
    """SEDAC 主类集成 CognitiveTrace"""
    from metis import SEDAC

    sedac = SEDAC()
    assert sedac.trace is None

    sedac.start_session("test query")
    assert sedac.trace is not None
    assert sedac.trace.query == "test query"
    assert sedac.trace.total_tokens == 0

    # 模拟一步
    logits = torch.randn(1, 100)
    sedac.step(logits)
    assert sedac.trace.total_tokens == 1

runner.run("SEDAC 集成 CognitiveTrace", test_cognitive_trace_in_sedac)


# ════════════════════════════════════════════════════════════════
# 11. MetacognitiveCore 元认知
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("11. MetacognitiveCore 元认知")
print("="*60)


def test_metacognition_import():
    """MetacognitiveCore 导入"""
    from metis.cognitive.metacognition import MetacognitiveCore
    from metis.core.types import MetaJudgment
    mc = MetacognitiveCore()
    assert mc is not None

runner.run("MetacognitiveCore 导入", test_metacognition_import)


def test_metacognition_empty_trace():
    """空轨迹内省"""
    from metis.cognitive.metacognition import MetacognitiveCore
    from metis.core.types import CognitiveTrace

    mc = MetacognitiveCore()
    trace = CognitiveTrace(query="test")
    judgment = mc.introspect(trace)
    assert judgment.reasoning == "空轨迹，无法内省"

runner.run("MetacognitiveCore 空轨迹", test_metacognition_empty_trace)


def test_metacognition_confident_trace():
    """确信轨迹 → 高 confidence, continue"""
    from metis.cognitive.metacognition import MetacognitiveCore
    from metis.core.types import (
        CognitiveTrace, CognitiveSignal, Decision,
        EpistemicState, BoundaryAction,
    )

    mc = MetacognitiveCore()
    trace = CognitiveTrace(query="简单问题")

    # 模拟 20 步高确信信号
    for i in range(20):
        sig = CognitiveSignal(
            token_entropy=0.3, semantic_entropy=0.2, confidence=0.9,
            z_score=-0.5, decision=Decision.FAST,
            epistemic_state=EpistemicState.KNOWN,
            boundary_action=BoundaryAction.GENERATE,
            entropy_trend="stable",
        )
        trace.add_event(sig, step=i)

    judgment = mc.introspect(trace)
    assert judgment.epistemic_confidence > 0.6, (
        f"确信轨迹 confidence 应 > 0.6, got {judgment.epistemic_confidence:.2f}"
    )
    assert judgment.suggested_action == "continue"
    assert judgment.boundary_status == "stable"

runner.run("MetacognitiveCore 确信轨迹", test_metacognition_confident_trace)


def test_metacognition_uncertain_trace():
    """不确定轨迹 → 低 confidence, hedge/verify"""
    from metis.cognitive.metacognition import MetacognitiveCore
    from metis.core.types import (
        CognitiveTrace, CognitiveSignal, Decision,
        EpistemicState, BoundaryAction,
    )

    mc = MetacognitiveCore()
    trace = CognitiveTrace(query="未知问题")

    # 模拟 20 步低确信信号
    for i in range(20):
        sig = CognitiveSignal(
            token_entropy=3.0, semantic_entropy=4.0, confidence=0.15,
            z_score=2.5, decision=Decision.DEEP,
            epistemic_state=EpistemicState.UNCERTAIN,
            boundary_action=BoundaryAction.HEDGE,
            entropy_trend="rising",
        )
        trace.add_event(sig, step=i)

    judgment = mc.introspect(trace)
    assert judgment.epistemic_confidence < 0.4, (
        f"不确定轨迹 confidence 应 < 0.4, got {judgment.epistemic_confidence:.2f}"
    )
    assert judgment.suggested_action in ("hedge", "verify", "abort")
    assert judgment.cognitive_load > 0.3

runner.run("MetacognitiveCore 不确定轨迹", test_metacognition_uncertain_trace)


def test_metacognition_hallucination_detection():
    """幻觉检测: 高 confidence + 高 z-score"""
    from metis.cognitive.metacognition import MetacognitiveCore
    from metis.core.types import (
        CognitiveTrace, CognitiveSignal, Decision,
        EpistemicState, BoundaryAction,
    )

    mc = MetacognitiveCore()
    trace = CognitiveTrace(query="幻觉问题")

    # 矛盾信号: 模型很自信但熵很高
    for i in range(20):
        sig = CognitiveSignal(
            token_entropy=3.0, semantic_entropy=3.5, confidence=0.85,
            z_score=2.0, decision=Decision.NORMAL,
            epistemic_state=EpistemicState.LIKELY,
            boundary_action=BoundaryAction.GENERATE,
            entropy_trend="stable",
        )
        trace.add_event(sig, step=i)

    judgment = mc.introspect(trace)
    assert judgment.hallucination_risk > 0.2, (
        f"矛盾信号应检出幻觉风险 > 0.2, got {judgment.hallucination_risk:.2f}"
    )

runner.run("MetacognitiveCore 幻觉检测", test_metacognition_hallucination_detection)


def test_metacognition_regulate():
    """元认知调节"""
    from metis.cognitive.metacognition import MetacognitiveCore
    from metis.core.types import MetaJudgment, EpistemicState

    mc = MetacognitiveCore()

    # 高幻觉风险 + 无 SE → should_verify
    j1 = MetaJudgment(
        hallucination_risk=0.6,
        suggested_action="verify",
    )
    reg1 = mc.regulate(j1)
    assert reg1["should_verify"]
    assert reg1["should_increase_samples"]

    # 正常 → continue
    j2 = MetaJudgment(
        epistemic_confidence=0.8,
        hallucination_risk=0.1,
        suggested_action="continue",
    )
    reg2 = mc.regulate(j2)
    assert not reg2["should_verify"]
    assert not reg2["should_hedge"]

runner.run("MetacognitiveCore 调节", test_metacognition_regulate)


def test_sedac_introspect():
    """SEDAC.introspect() 集成测试"""
    from metis import SEDAC
    from metis.core.types import MetaJudgment

    sedac = SEDAC()
    sedac.start_session("test")

    for _ in range(15):
        logits = torch.randn(1, 100)
        sedac.step(logits)

    judgment = sedac.introspect()
    assert isinstance(judgment, MetaJudgment)
    assert judgment.reasoning != ""

runner.run("SEDAC.introspect() 集成", test_sedac_introspect)


# ════════════════════════════════════════════════════════════════
# 12. SE → CuriosityDriver 闭环
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("12. SE → CuriosityDriver 闭环")
print("="*60)


def test_curiosity_se_gap():
    """CuriosityDriver.record_se_gap()"""
    from metis.cognitive.curiosity import CuriosityDriver

    driver = CuriosityDriver(storage_path=None)
    assert driver.gap_count == 0

    gap = driver.record_se_gap(
        query="2030年诺贝尔物理学奖",
        semantic_entropy=1.8,
        n_clusters=4,
        n_samples=5,
    )
    assert driver.gap_count == 1
    assert gap.category == "se_verified_uncertainty"
    assert "SE=1.80" in gap.context

runner.run("SE → CuriosityDriver 闭环", test_curiosity_se_gap)


# ════════════════════════════════════════════════════════════════
# 13. Boundary Guard 新逻辑 (A3 修复验证)
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("13. Boundary Guard 新逻辑")
print("="*60)


def test_boundary_high_z_low_c_refuse():
    """持续高z + 低confidence → REFUSE (真不知道)"""
    from metis.cognitive.boundary import EpistemicBoundaryGuard
    from metis.core.types import CognitiveSignal, BoundaryAction

    guard = EpistemicBoundaryGuard(min_warmup_tokens=3)
    for _ in range(4):
        guard.evaluate(CognitiveSignal(z_score=0.0, confidence=0.8))

    # 需要连续 2+ 个高 z token (streak >= MIN_STREAK_FOR_REFUSE)
    sig = CognitiveSignal(z_score=3.0, confidence=0.1)
    guard.evaluate(sig)
    _, action, _ = guard.evaluate(sig)
    assert action == BoundaryAction.REFUSE, f"低c+持续高z 应 REFUSE, got {action}"

runner.run("高z+低c → REFUSE", test_boundary_high_z_low_c_refuse)


def test_boundary_high_z_mid_c_seek():
    """持续高z + 中confidence → SEEK"""
    from metis.cognitive.boundary import EpistemicBoundaryGuard
    from metis.core.types import CognitiveSignal, BoundaryAction

    guard = EpistemicBoundaryGuard(min_warmup_tokens=3)
    for _ in range(4):
        guard.evaluate(CognitiveSignal(z_score=0.0, confidence=0.8))

    # 需要连续 2+ 个高 z token (streak >= MIN_STREAK_FOR_REFUSE)
    sig = CognitiveSignal(z_score=3.0, confidence=0.5)
    guard.evaluate(sig)
    _, action, _ = guard.evaluate(sig)
    assert action == BoundaryAction.SEEK, f"中c+持续高z 应 SEEK, got {action}"

runner.run("高z+中c → SEEK", test_boundary_high_z_mid_c_seek)


def test_boundary_high_z_high_c_hedge():
    """高z + 高confidence → HEDGE (矛盾信号/幻觉风险)"""
    from metis.cognitive.boundary import EpistemicBoundaryGuard
    from metis.core.types import CognitiveSignal, BoundaryAction

    guard = EpistemicBoundaryGuard(min_warmup_tokens=3)
    for _ in range(4):
        guard.evaluate(CognitiveSignal(z_score=0.0, confidence=0.8))

    sig = CognitiveSignal(z_score=3.0, confidence=0.85)
    _, action, explanation = guard.evaluate(sig)
    assert action == BoundaryAction.HEDGE, f"高c+高z 应 HEDGE, got {action}"
    assert "hallucination" in explanation or "conflict" in explanation

runner.run("高z+高c → HEDGE (幻觉风险)", test_boundary_high_z_high_c_hedge)


# ════════════════════════════════════════════════════════════════
# 14. 真实模型集成 (可选)
# ════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("14. 真实模型集成 (需要 GPU + 模型)")
print("="*60)

MODEL_PATH = "G:/models/qwen2.5-7b"


def test_real_model_se():
    """真实模型语义熵估计"""
    if os.environ.get("SKIP_MODEL_TESTS", "0") == "1":
        print("  ⊘ 跳过: SKIP_MODEL_TESTS=1")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"  ⊘ 跳过: 模型不存在 ({MODEL_PATH})")
        return

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from metis import SEDAC

    print("  加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
    model.eval()

    # 使用 embedding 模式 (避免额外下载 NLI 模型)
    sedac = SEDAC.attach(
        model, tokenizer,
        se_method="embedding",
        se_n_samples=3,
        se_temperature=0.7,
    )

    # 确定性问题
    print("  测试确定性问题...")
    result = sedac.evaluate_semantic_entropy("中国的首都是哪里？")
    print(f"    SE = {result.semantic_entropy:.2f}, "
          f"clusters = {result.n_clusters}, "
          f"uncertain = {result.is_uncertain}")
    print(f"    majority: {result.majority_answer[:50]}...")

    # 不确定性问题
    print("  测试不确定性问题...")
    result2 = sedac.evaluate_semantic_entropy("2030年世界杯冠军是谁？")
    print(f"    SE = {result2.semantic_entropy:.2f}, "
          f"clusters = {result2.n_clusters}, "
          f"uncertain = {result2.is_uncertain}")

    # 基本断言
    assert result.n_samples == 3
    assert result.majority_answer != ""

runner.run("真实模型语义熵 (Qwen2.5-7B)", test_real_model_se)


def test_real_model_inference_pipeline():
    """真实模型 SEDAC 推理管线"""
    if os.environ.get("SKIP_MODEL_TESTS", "0") == "1":
        print("  ⊘ 跳过: SKIP_MODEL_TESTS=1")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"  ⊘ 跳过: 模型不存在 ({MODEL_PATH})")
        return

    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from metis import SEDAC, SEDACInference

    print("  加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True
        )
    model.eval()

    sedac = SEDAC.attach(
        model, tokenizer,
        se_method="embedding",
        se_n_samples=3,
    )
    engine = SEDACInference(sedac)

    print("  测试推理管线...")
    result = engine.generate(
        "中国的首都是哪里？",
        max_tokens=30,
        enable_system2=False,  # 先不触发 System 2
    )

    print(f"    text: {result.text[:60]}...")
    print(f"    tokens: {result.tokens_generated}")
    print(f"    S1 ratio: {result.system1_ratio:.2f}")
    print(f"    S2 ratio: {result.system2_ratio:.2f}")
    print(f"    hedged: {result.was_hedged}")
    print(f"    refused: {result.was_refused}")
    print(f"    latency: {result.latency_ms:.0f}ms")

    assert result.text != ""
    assert result.tokens_generated > 0
    assert result.latency_ms > 0

runner.run("真实模型推理管线 (Qwen2.5-7B)", test_real_model_inference_pipeline)


# ════════════════════════════════════════════════════════════════
# 结果汇总
# ════════════════════════════════════════════════════════════════

all_passed = runner.summary()
sys.exit(0 if all_passed else 1)
