"""wmflow 类型系统: 端口类型 + 数据容器.

IMAGE 约定: np.ndarray float32 (H,W,3) RGB [0,1] —— 与 scw_research/scw 库一致。
MESSAGE 约定: bits np.ndarray uint8 (L,) 或 (B,L), 值域 {0,1}。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np

# --- 端口类型名 (大写字符串, 用于连线类型检查) ---
IMAGE = "IMAGE"
MASK = "MASK"
WM_MODEL = "WM_MODEL"
MESSAGE = "MESSAGE"
ATTACK = "ATTACK"
PARAMS = "PARAMS"
METRICS = "METRICS"
REPORT = "REPORT"
TENSOR = "TENSOR"  # 网络内部张量 (B,C,H,W) torch.Tensor —— 模块积木的连线类型
# 标量/控制面
INT = "INT"
FLOAT = "FLOAT"
STRING = "STRING"
BOOL = "BOOL"

Image = np.ndarray


@dataclass
class Message:
    """一条水印消息。bits: (L,) 或 (B,L) uint8/bool, 值 {0,1}。"""
    bits: np.ndarray
    text: Optional[str] = None

    def __post_init__(self):
        self.bits = np.asarray(self.bits)
        if self.bits.dtype != np.uint8:
            self.bits = self.bits.astype(np.uint8)
        if self.bits.ndim not in (1, 2):
            raise ValueError(f"bits 需为 1D(L,) 或 2D(B,L), got {self.bits.shape}")

    @property
    def length(self) -> int:
        return int(self.bits.shape[-1])

    def to_dict(self) -> Dict[str, Any]:
        return {"bits": self.bits.tolist(), "text": self.text}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        return cls(np.asarray(d["bits"], dtype=np.uint8), d.get("text"))


@dataclass
class WMModel:
    """水印模型句柄: 框架名 + 编解码调用 + 元信息。

    encode(cover, message, strength) -> Image
    decode(image) -> bits (L,)
    """
    framework: str
    msg_len: int
    encode_fn: Callable[[Image, np.ndarray, float], Image]
    decode_fn: Callable[[Image], np.ndarray]
    card_hash: str = ""
    weights_path: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Attack:
    """退化句柄: fn(Image, rng) -> (Image, params)。本身不在图上执行, 供链节点消费。"""
    name: str
    fn: Callable[[Image, "np.random.Generator"], tuple]
    meta: Dict[str, Any] = field(default_factory=dict)
