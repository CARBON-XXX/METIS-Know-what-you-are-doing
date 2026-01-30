"""
SEDAC V9.0 - Semantic Entropy-guided Dynamic Attention Core
============================================================

核心理念:
    不是"算得更快"，而是"算得更少"
    不是"加速器"，而是"认知协处理器"

架构:
    低熵 (< 2.5)  → Early Exit (跳过71%层)
    中熵 (2.5-5)  → Normal (完整推理)
    高熵 (> 5.0)  → O1 Thinking (深度思考)
"""

__version__ = "9.0.0"

# V9.0 Production Module (生产级 - 推荐使用)
from sedac.v9.production import (
    ProductionConfig,
    ProductionSEDACEngine,
    SEDACInferencePipeline,
)

__all__ = [
    "ProductionConfig",
    "ProductionSEDACEngine",
    "SEDACInferencePipeline",
    "__version__",
]
