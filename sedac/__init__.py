"""
SEDAC V9.0 - Semantic Entropy-guided Dynamic Attention Core
============================================================

核心理念:
    不是"算得更快"，而是"算得更少"
    不是"加速器"，而是"认知协处理器"

Quick Start:
    from sedac.v9.production import ProductionSEDACEngine, ProductionConfig
    
    config = ProductionConfig()
    engine = ProductionSEDACEngine(config)
    
    # 在推理循环中使用
    should_exit, entropy, confidence = engine.should_exit(hidden, logits, layer_idx, total_layers)
"""

__version__ = "9.0.0"
__author__ = "CARBON-XXX"

# V9 Production API
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
