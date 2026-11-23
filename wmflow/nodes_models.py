"""模型侧节点: WMModelLoader / Embed / Decode —— 框架适配层。

适配器 = 把各经典框架的编解码接口统一成:
    encode(cover: (H,W,3) f32, bits: (L,), strength: float) -> (H,W,3) f32
    decode(img: (H,W,3) f32) -> (L,) uint8

内置:
  - lsb_toy : 玩具 LSB 模型, 零权重零依赖, 专供冒烟/图机制验证
  - s2r     : scw_research 的 S2R EncoderMP/Decoder (需 scw 在 PYTHONPATH + 权重)
待接 (接口留好, P1): stegastamp / hidden / mbrs / videoseal
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .node import Node
from .registry import register
from .types import (IMAGE, INT, FLOAT, STRING, MESSAGE, WM_MODEL, Message,
                    WMModel)

# ---------------------------------------------------------------------------
# LSB 玩具模型 (自包含, 供冒烟: 干净往返 BER=0)
# ---------------------------------------------------------------------------

def _lsb_encode(cover: np.ndarray, bits: np.ndarray, strength: float) -> np.ndarray:
    """比特铺满低位平面 (重复填充), strength 控混合。"""
    h, w, _ = cover.shape
    n = h * w
    flat = np.repeat(bits, int(np.ceil(n / len(bits))))[:n]
    plane = flat.reshape(h, w).astype(np.float32)
    img = cover.copy()
    for c in range(3):
        q = np.floor(img[:, :, c] * 255.0)
        q = (q // 2) * 2 + plane  # 写入 LSB
        img[:, :, c] = (1 - strength) * img[:, :, c] + strength * (q / 255.0)
    return np.clip(img, 0, 1)


def _lsb_decode(img: np.ndarray, msg_len: int) -> np.ndarray:
    h, w, _ = img.shape
    q = np.floor(img[:, :, 0] * 255.0)
    bits = (q.astype(np.int64) % 2).reshape(-1)
    reps = int(np.ceil(h * w / msg_len))
    view = bits[: msg_len * reps].reshape(msg_len, reps)
    return (view.mean(axis=1) > 0.5).astype(np.uint8)


def make_lsb_toy(msg_len: int = 64) -> WMModel:
    return WMModel(
        framework="lsb_toy", msg_len=msg_len,
        encode_fn=lambda cover, bits, s: _lsb_encode(cover, bits, s),
        decode_fn=lambda img: _lsb_decode(img, msg_len),
        meta={"note": "冒烟用玩具模型, 非基线"},
    )


# ---------------------------------------------------------------------------
# S2R 适配 (scw_research 复刻; 需权重)
# ---------------------------------------------------------------------------

def make_s2r(weights_path: str, device: str = "cpu") -> WMModel:
    import torch
    from scw.s2r_models import EncoderMP, Decoder
    enc = EncoderMP(H=128, W=128, message_length=64)
    dec = Decoder(H=128, W=128, message_length=64)
    sd = torch.load(weights_path, map_location=device, weights_only=True)
    enc.load_state_dict(sd["encoder"], strict=True)
    dec.load_state_dict(sd["decoder"], strict=True)
    enc.eval(); dec.eval()

    def encode(cover, bits, strength):
        import cv2
        small = cv2.resize(cover, (128, 128), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.transpose(2, 0, 1))[None].float() * 2 - 1
        m = torch.from_numpy(bits.astype(np.float32))[None]
        with torch.no_grad():
            y = (enc(x, m) + 1) / 2
        out = y[0].permute(1, 2, 0).cpu().numpy()
        out = cv2.resize(out, (cover.shape[1], cover.shape[0]))
        return (1 - strength) * cover + strength * np.clip(out, 0, 1)

    def decode(img):
        import cv2
        small = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(small.transpose(2, 0, 1))[None].float() * 2 - 1
        with torch.no_grad():
            logit = dec(x)
        return (torch.sigmoid(logit)[0] > 0.5).cpu().numpy().astype(np.uint8)

    return WMModel(framework="s2r", msg_len=64,
                   encode_fn=encode, decode_fn=decode,
                   weights_path=str(weights_path))


ADAPTERS = {
    "lsb_toy": lambda **kw: make_lsb_toy(kw.get("msg_len", 64)),
    "s2r": lambda **kw: make_s2r(kw["weights_path"], kw.get("device", "cpu")),
}


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
@register
class WMModelLoader(Node):
    """加载水印模型句柄 → WM_MODEL (adapter 统一接口)。"""
    DISPLAY_NAME = "WM Model Loader"
    CATEGORY = "wm/model"
    RETURN_TYPES = (WM_MODEL,)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "framework": (STRING, {"default": "lsb_toy", "choices": [
                "lsb_toy", "s2r", "stegastamp", "hidden", "mbrs", "videoseal"]}),
            "weights_path": (STRING, {"default": ""}),
            "msg_len": (INT, {"default": 64}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, framework, **_):
        if framework not in ADAPTERS:
            return [f"框架 '{framework}' 适配器未接入 (P1), 现有: {list(ADAPTERS)}"]
        return []

    @classmethod
    def fingerprint_inputs(cls, framework, weights_path, msg_len, **_):
        p = Path(weights_path) if weights_path else None
        mt = p.stat().st_mtime_ns if p and p.exists() else "none"
        return f"{framework}:{weights_path}:{mt}:{msg_len}"

    def execute(self, framework, weights_path, msg_len):
        kw = {"msg_len": msg_len}
        if weights_path:
            kw["weights_path"] = weights_path
        return (ADAPTERS[framework](**kw),)


@register
class Embed(Node):
    """嵌入: model + cover + message + strength → 水印图。"""
    DISPLAY_NAME = "Embed"
    CATEGORY = "wm/model"
    RETURN_TYPES = (IMAGE,)
    RETURN_NAMES = ("watermarked",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (WM_MODEL, {}),
            "cover": (IMAGE, {}),
            "message": (MESSAGE, {}),
            "strength": (FLOAT, {"default": 1.0, "min": 0.0, "max": 1.0}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, model, message, strength, **_):
        errs = []
        if message.length != model.msg_len:
            errs.append(f"消息长 {message.length} != 模型 msg_len {model.msg_len}")
        if not (0.0 <= strength <= 1.0):
            errs.append(f"strength 超界: {strength}")
        return errs

    def execute(self, model, cover, message, strength):
        return (model.encode_fn(cover, message.bits, float(strength)),)


@register
class Decode(Node):
    """提取: model + image → message。"""
    DISPLAY_NAME = "Decode"
    CATEGORY = "wm/model"
    RETURN_TYPES = (MESSAGE,)
    RETURN_NAMES = ("message",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": (WM_MODEL, {}),
            "image": (IMAGE, {}),
        }}

    def execute(self, model, image):
        bits = model.decode_fn(image)
        return (Message(np.asarray(bits).reshape(-1).astype(np.uint8)),)
