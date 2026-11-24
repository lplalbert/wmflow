"""核心节点包: 全框架共用的 IO / 攻击链 / 度量节点 (12 核心节点的 9 个)。

自带一套最小内置实现 (LSB 玩具模型 + 基础攻击层), 保证仓库开箱可跑冒烟;
scw_research 在 PYTHONPATH 时, NoiseLayer/ImageProfile 自动扩展到完整
LAYERS 注册表与 glare/moiré 度量族 (适配, 非复制)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .node import Node
from .registry import register
from .types import (IMAGE, INT, FLOAT, STRING, MESSAGE, METRICS, PARAMS,
                    Message)

# ---------------------------------------------------------------------------
# 可选外部库探测: scw_research/scw (噪声层注册表 + 度量族)
# ---------------------------------------------------------------------------
try:
    from scw import noise_layers as _scw_layers
    from scw import metrics as _scw_metrics
    _HAS_SCW = True
except Exception:
    _scw_layers = _scw_metrics = None
    _HAS_SCW = False


# ---------------------------------------------------------------------------
# 内置攻击层 (自包含, 供冒烟与无 scw 环境)
# ---------------------------------------------------------------------------
def _jpg(img: np.ndarray, rng: np.random.Generator, q=(40, 95)):
    import cv2
    qq = int(rng.integers(*q))
    ok, buf = cv2.imencode(".jpg", (img * 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, qq])
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    # 通道翻转是负 stride 视图: 直接喂 torch.from_numpy 会报
    # "At least one stride ... is negative", 所以这里就落成连续数组。
    return np.ascontiguousarray(out[:, :, ::-1]), {"jpeg_q": qq}


def _blur(img, rng, k=(3, 9)):
    import cv2
    kk = int(rng.integers(*k)) | 1
    return cv2.GaussianBlur(img, (kk, kk), 0).astype(np.float32), {"blur_k": kk}


def _noise(img, rng, s=(0.005, 0.03)):
    sigma = float(rng.uniform(*s))
    return np.clip(img + rng.normal(0, sigma, img.shape).astype(np.float32), 0, 1), \
        {"noise_sigma": sigma}


def _combo_builtin(img, rng):
    """StegaStamp 式随机组合: 模糊/JPEG/噪声 串联。"""
    params = {}
    if rng.random() < 0.6:
        img, p = _blur(img, rng); params.update(p)
    if rng.random() < 0.6:
        img, p = _jpg(img, rng); params.update(p)
    img, p = _noise(img, rng); params.update(p)
    return np.clip(img, 0, 1), params


BUILTIN_LAYERS = {
    "none": lambda img, rng: (img, {}),
    "jpeg": _jpg,
    "blur": _blur,
    "noise": _noise,
    "combo_builtin": _combo_builtin,
}


def all_layer_names():
    names = list(BUILTIN_LAYERS)
    if _HAS_SCW:
        names += [k for k in _scw_layers.LAYERS if k not in names]
    return names


def run_layer(name: str, img: np.ndarray, rng: np.random.Generator):
    if name in BUILTIN_LAYERS:
        return BUILTIN_LAYERS[name](img, rng)
    if _HAS_SCW and name in _scw_layers.LAYERS:
        return _scw_layers.LAYERS[name](img, rng)
    raise ValueError(f"未知噪声层: {name}; 可用: {all_layer_names()}")


# ---------------------------------------------------------------------------
# IO 节点
# ---------------------------------------------------------------------------
@register
class CoverLoader(Node):
    """加载载体图 → IMAGE (H,W,3) float32 RGB [0,1]。fingerprint 按文件内容哈希。"""
    DISPLAY_NAME = "Cover Loader"
    CATEGORY = "wm/io"
    RETURN_TYPES = (IMAGE,)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": (STRING, {"default": ""})}}

    @classmethod
    def fingerprint_inputs(cls, path, **_):
        p = Path(path)
        return f"{path}:{p.stat().st_mtime_ns if p.exists() else 'missing'}"

    def execute(self, path):
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"读不到图像: {path}")
        return (img[:, :, ::-1].astype(np.float32) / 255.0,)


@register
class MessageGen(Node):
    """生成随机消息比特 (显式 seed, 可复现)。"""
    DISPLAY_NAME = "Message Gen"
    CATEGORY = "wm/io"
    RETURN_TYPES = (MESSAGE,)
    RETURN_NAMES = ("message",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "length": (INT, {"default": 64}),
            "seed": (INT, {"default": 0}),
        }}

    def execute(self, length, seed):
        rng = np.random.default_rng(seed)
        return (Message(rng.integers(0, 2, size=int(length)).astype(np.uint8)),)


@register
class SaveReport(Node):
    """汇总 METRICS 写 report.json (多来源合并)。"""
    DISPLAY_NAME = "Save Report"
    CATEGORY = "wm/io"
    RETURN_TYPES = (METRICS,)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "out_dir": (STRING, {"default": "out"}),
            "metrics_json": (STRING, {"default": "{}", "multiline": True}),
        }}

    def execute(self, out_dir, metrics_json):
        merged = json.loads(metrics_json or "{}")
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=1)
        return (merged,)


# ---------------------------------------------------------------------------
# 攻击链节点
# ---------------------------------------------------------------------------
@register
class NoiseLayer(Node):
    """可组合退化层: img → 退化 img, params 落 PARAMS。combo 选层。"""
    DISPLAY_NAME = "Noise Layer"
    CATEGORY = "wm/attack"
    RETURN_TYPES = (IMAGE, PARAMS)
    RETURN_NAMES = ("image", "params")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": (IMAGE, {}),
            "layer": (STRING, {"default": "combo_builtin",
                               "choices": "__dynamic__:layers"}),
            "seed": (INT, {"default": 0}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, layer, **_):
        if layer not in all_layer_names():
            return [f"未知层 '{layer}', 可用: {all_layer_names()}"]
        return []

    def execute(self, image, layer, seed):
        rng = np.random.default_rng(seed)
        out, params = run_layer(layer, image, rng)
        return (np.clip(out, 0, 1).astype(np.float32), params)


@register
class TranscodeChain(Node):
    """转码链预设: none / jpeg95 / 420 / resize / wechat_lite (轻量复刻配对析因臂)。"""
    DISPLAY_NAME = "Transcode Chain"
    CATEGORY = "wm/attack"
    RETURN_TYPES = (IMAGE, PARAMS)
    RETURN_NAMES = ("image", "params")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": (IMAGE, {}),
            "preset": (STRING, {"default": "none", "choices": [
                "none", "jpeg95", "jpeg75", "chroma420", "resize_half", "wechat_lite"]}),
        }}

    def execute(self, image, preset):
        import cv2
        img = image
        params = {"preset": preset}

        def _jpeg(q):
            nonlocal img
            ok, buf = cv2.imencode(".jpg", (np.clip(img, 0, 1) * 255).astype(np.uint8),
                                   [cv2.IMWRITE_JPEG_QUALITY, q])
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
            img = img[:, :, ::-1]
            params["jpeg_q"] = q

        if preset == "none":
            pass
        elif preset == "jpeg95":
            _jpeg(95)
        elif preset == "jpeg75":
            _jpeg(75)
        elif preset == "chroma420":
            ycc = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
            h, w = ycc.shape[:2]
            for c in (1, 2):
                small = cv2.resize(ycc[:, :, c], (w // 2, h // 2),
                                   interpolation=cv2.INTER_AREA)
                ycc[:, :, c] = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
        elif preset == "resize_half":
            h, w = img.shape[:2]
            small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        elif preset == "wechat_lite":
            h, w = img.shape[:2]
            ycc = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
            for c in (1, 2):
                s = cv2.resize(ycc[:, :, c], (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                ycc[:, :, c] = cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB).astype(np.float32) / 255.0
            _jpeg(80)
            small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        return (np.clip(img, 0, 1).astype(np.float32), params)


# ---------------------------------------------------------------------------
# 度量节点
# ---------------------------------------------------------------------------
@register
class BERCompare(Node):
    """消息对比 → {ber, bit_errors, n_bits}。"""
    DISPLAY_NAME = "BER Compare"
    CATEGORY = "wm/metrics"
    RETURN_TYPES = (METRICS,)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ref": (MESSAGE, {}),
            "dec": (MESSAGE, {}),
        }}

    def execute(self, ref, dec):
        a, b = np.asarray(ref.bits), np.asarray(dec.bits)
        if a.shape != b.shape:
            raise ValueError(f"消息长度不匹配: {a.shape} vs {b.shape}")
        errs = int(np.sum(a != b))
        return ({"ber": errs / a.size, "bit_errors": errs,
                 "n_bits": int(a.size)},)


@register
class ImageProfile(Node):
    """图像质量画像: 基础 PSNR/亮度/对比度; scw 在位时追加 moiré/glare 族。"""
    DISPLAY_NAME = "Image Profile"
    CATEGORY = "wm/metrics"
    RETURN_TYPES = (METRICS,)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": (IMAGE, {}),
            "ref": (IMAGE, {"default": None}),
        }}

    def execute(self, image, ref=None):
        prof = {}
        if ref is not None:
            mse = float(np.mean((image - ref) ** 2))
            prof["psnr"] = 10 * np.log10(1.0 / max(mse, 1e-10))
        gray = image.mean(axis=2)
        prof["luma_mean"] = float(gray.mean())
        prof["contrast"] = float(gray.std())
        if _HAS_SCW:
            prof.update(_scw_metrics.image_profile(image, ref))
        return (prof,)


# ---------------------------------------------------------------------------
# P0 占位 (P1 接 scw 实现): 校正 / 可微物理信道
# ---------------------------------------------------------------------------
@register
class Rectify(Node):
    """几何校正占位: P1 接 scw.rectify 双轨 (auto 单应 / golden 四角逐位)。"""
    DISPLAY_NAME = "Rectify"
    CATEGORY = "wm/post"
    RETURN_TYPES = (IMAGE, PARAMS)
    RETURN_NAMES = ("image", "info")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": (IMAGE, {}),
            "mode": (STRING, {"default": "identity",
                               "choices": ["identity", "auto", "golden"]}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, mode, **_):
        if mode != "identity":
            return [f"Rectify mode='{mode}' 需 P1 接入 scw.rectify 后启用 (当前占位仅支持 identity)"]
        return []

    def execute(self, image, mode):
        return (image, {"mode": mode, "placeholder": True})


@register
class PhysicsChannel(Node):
    """可微屏摄物理信道占位: P1 接 scw.physics.screen_camera_channel。"""
    DISPLAY_NAME = "Physics Channel"
    CATEGORY = "wm/attack"
    RETURN_TYPES = (IMAGE, PARAMS)
    RETURN_NAMES = ("image", "params")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": (IMAGE, {}),
            "seed": (INT, {"default": 0}),
            "params_json": (STRING, {"default": "{}", "multiline": True}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, **_):
        return []  # 占位直通可用; P1 接入 scw.physics 后增加参数预检

    def execute(self, image, seed, params_json):
        return (image, {"seed": seed, "placeholder": True})
