# SEDAC V9.0 Production

**Semantic Entropy-guided Dynamic Attention Core** - 生产级 LLM 推理加速框架

基于语义熵的自适应早退机制，实现智能计算资源分配：简单 Token 快速退出，复杂 Token 深度推理。

---

## 核心特性

| 功能 | 描述 |
|------|------|
| **自适应阈值校准** | 从推理数据自动学习最优熵阈值，无需手动调参 |
| **Ghost KV 生成器** | 轻量级 MLP 预测跳过层的 KV Cache，保持输出质量 |
| **O1 深度推理** | 高熵 Token 触发迭代思考，提升复杂问题准确率 |
| **CUDA 加速内核** | 融合熵计算与 Token 路由，降低 GPU 开销 |
| **交互式可视化** | 实时显示 SEDAC 决策过程（熵值、置信度、退出层）|

---

## 快速开始

### 安装

```bash
git clone https://github.com/CARBON-XXX/SEDAC-V9.0-Pre-release-Test-Version.git
cd SEDAC-V9.0-Pre-release-Test-Version
pip install -r requirements.txt
```

### 交互式对话测试

```bash
# 使用本地模型
python -m sedac.v9.production.interactive_chat --model /path/to/model --local

# 在线下载模型
python -m sedac.v9.production.interactive_chat --model Qwen/Qwen2.5-0.5B-Instruct
```

### 运行单元测试

```bash
python -m sedac.v9.production.tests
```

---

## 项目结构

```
sedac/v9/production/
├── config.py              # 生产配置（自适应参数）
├── engine.py              # SEDAC 核心引擎
├── inference.py           # 推理管线
├── auto_calibration.py    # 自动参数校准
├── interactive_chat.py    # 交互式对话测试
├── trainer.py             # Ghost KV 训练器
├── benchmark.py           # 性能基准测试
├── server.py              # FastAPI 服务
└── tests.py               # 单元测试套件
```

---

## 核心原理

### 语义熵计算

$$H(x) = -\sum_{i} p_i \log_2 p_i$$

- **低熵** ($H < \tau_{low}$): Token 确定性高 → 早退 + Ghost KV
- **中熵** ($\tau_{low} < H < \tau_{high}$): 正常推理
- **高熵** ($H > \tau_{high}$): 触发 O1 深度推理

### 自适应阈值校准

阈值不再是固定值，而是从数据中学习：

```yaml
# config.yaml - 阈值会被 AutoCalibrator 自动覆盖
sedac:
  auto_calibrate: true
  entropy_threshold_base: 0.5  # → P50 自动学习
  entropy_threshold_min: 0.2   # → P20 自动学习
  o1_high_entropy_threshold: 4.5  # → P90 自动学习
```

---

## API 使用

```python
from sedac.v9.production import SEDACInferencePipeline, create_pipeline

# 创建推理管线
pipeline = create_pipeline("Qwen/Qwen2.5-7B-Instruct")

# 推理
result = pipeline("解释量子纠缠")

print(f"回答: {result.generated_text}")
print(f"加速比: {result.skip_ratio:.1%}")
print(f"平均退出层: {result.avg_exit_layer:.1f}")
```

---

## 性能指标

在 Qwen2.5-7B 上的测试结果：

| 指标 | 数值 |
|------|------|
| 平均跳过层数 | 40-60% |
| 延迟降低 | 30-50% |
| 输出质量保持 | >98% |

---

## 配置说明

```yaml
# sedac/v9/production/config.yaml

model:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  num_hidden_layers: 28

sedac:
  auto_calibrate: true           # 启用自动校准
  enable_ghost_kv: true          # 启用 Ghost KV
  enable_o1_reasoning: true      # 启用 O1 深度推理
  adaptive_threshold: true       # 在线自适应阈值

performance:
  kernel_backend: "cuda_cpp"     # CUDA 加速
  enable_flash_attention: true   # Flash Attention
```

---

## 测试命令

```bash
# 单元测试
python -m sedac.v9.production.tests

# 集成测试
python -m sedac.v9.production.integration_test --model Qwen/Qwen2.5-0.5B-Instruct

# 性能基准
python -m sedac.v9.production.benchmark --model Qwen/Qwen2.5-7B-Instruct

# 自动校准
python -m sedac.v9.production.auto_calibration --model Qwen/Qwen2.5-0.5B-Instruct
```

---

## License
Apache-2.0 许可证
