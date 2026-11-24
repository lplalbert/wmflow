"""经典水印框架的积木配方 (模块级忠实, 层级简化)。

每个配方 = encoder/decoder 两张 NetSpec + 元信息 (论文/攻击套件/保真口径)。
保真口径 (对齐既往裁决纪律): 称"模块级结构对照复现", 不称官方等价复现;
层宽/层深按 128² 输入合理化, 差异在此逐条注明。

HiDDeN (Zhu et al. ECCV 2018; jirenz/HiDDeN, ando-khachatryan/HiDDeN):
  encoder = msg_map → concat → conv×2(下采样) → res×4 → up×2 → conv3 → +cover
  decoder = conv×3(下采样) → res×4 → GAP → bit_head
  训练期 Denoiser (crop/dropout/blur/noise/JPEG) 在编解码之间 —— 属训练侧, 不进推理积木。

StegaStamp (Tancik et al. CVPR 2020; tancik/StegaStamp):
  encoder = msg_map(100bit) → concat → conv → res×7 → conv3 → +cover
  decoder = conv×2 → res×4 → GAP → bit_head
  可微扰动 (blur/noise/DCT-JPEG/spatial) 是训练侧; 推理图只留编解码。

MBRS (Jia et al. ACM MM 2021; jzyustc/MBRS):
  骨干同 HiDDeN 形; 贡献在训练侧 (mini-batch 真实+模拟 JPEG)——配方记 train_note,
  推理积木与 hidden 同构 (msg_len/深度略调)。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .netspec import NetSpec


def _enc(name: str, msg_len: int, size: int, n_res: int,
         mid_ch: int = 64) -> Dict[str, Any]:
    """通用 encoder 骨架: msg_map→concat→down×2→res×n→up×2→conv3→+cover。"""
    mods = [
        {"id": "mm", "type": "msg_map", "params": {"out_ch": 1}, "from": ["message"]},
        {"id": "cat", "type": "concat", "from": ["image", "mm"]},
        {"id": "d1", "type": "conv_block", "params": {"out_ch": 32, "stride": 2}},
        {"id": "d2", "type": "conv_block", "params": {"out_ch": mid_ch, "stride": 2}},
    ]
    prev = "d2"
    for i in range(n_res):
        mods.append({"id": f"r{i+1}", "type": "res_block"})
        prev = f"r{i+1}"
    mods += [
        {"id": "u1", "type": "up_block", "params": {"out_ch": 32}},
        {"id": "u2", "type": "up_block", "params": {"out_ch": 16}},
        {"id": "c3", "type": "conv_block", "params": {"out_ch": 3}},
        {"id": "out", "type": "residual_add", "from": ["c3", "image"]},
    ]
    return {"name": name,
            "input": {"image": [3, size, size], "message": [msg_len]},
            "modules": mods, "output": "out"}


def _dec(name: str, msg_len: int, size: int, n_res: int,
         mid_ch: int = 64) -> Dict[str, Any]:
    mods = [
        {"id": "d1", "type": "conv_block", "params": {"out_ch": 32, "stride": 2}},
        {"id": "d2", "type": "conv_block", "params": {"out_ch": mid_ch, "stride": 2}},
        {"id": "d3", "type": "conv_block", "params": {"out_ch": mid_ch, "stride": 2}},
    ]
    for i in range(n_res):
        mods.append({"id": f"r{i+1}", "type": "res_block"})
    mods.append({"id": "out", "type": "bit_head", "params": {"msg_len": msg_len}})
    return {"name": name,
            "input": {"image": [3, size, size]},
            "modules": mods, "output": "out"}


# ---------------------------------------------------------------------------
# HiDDeN
# ---------------------------------------------------------------------------
HIDDEN = {
    "framework": "hidden",
    "paper": "HiDDeN: Hiding Data with Deep Networks (Zhu et al., ECCV 2018)",
    "msg_len": 30, "size": 128,
    "fidelity": "模块级结构对照 (msg预处理→concat→下采样→残差→上采样→+cover); "
                "层宽统一到 32/64, 原实现 hidden_size=64 一致; "
                "训练期 Denoiser 5 层属训练侧不进推理积木",
    "train_note": "encoder→Denoiser(crop/dropout/gauss-blur/noise/JPEG近似)→decoder 联训",
    "attacks_train": ["crop", "dropout", "gaussian_blur", "gaussian_noise", "jpeg_approx"],
}


def hidden(msg_len: int = 30, size: int = 128) -> Tuple[NetSpec, NetSpec, dict]:
    enc = NetSpec(_enc("hidden_encoder", msg_len, size, n_res=4))
    dec = NetSpec(_dec("hidden_decoder", msg_len, size, n_res=4))
    return enc, dec, HIDDEN


# ---------------------------------------------------------------------------
# StegaStamp
# ---------------------------------------------------------------------------
STEGASTAMP = {
    "framework": "stegastamp",
    "paper": "StegaStamp: Invisible Hyperlinks in Physical Photographs "
             "(Tancik et al., CVPR 2020)",
    "msg_len": 100, "size": 128,
    "fidelity": "模块级 (残差更深 7 块, 对应原 encoder 7×residual); "
                "decoder 两段下采样; 原实现输入 400², 此处 128² 归一化",
    "train_note": "可微扰动层 (blur/noise/DCT-JPEG/spatial transformer) 插编解码间联训",
    "attacks_train": ["blur", "noise", "dct_jpeg", "spatial_affine"],
}


def stegastamp(msg_len: int = 100, size: int = 128) -> Tuple[NetSpec, NetSpec, dict]:
    enc = NetSpec(_enc("stegastamp_encoder", msg_len, size, n_res=7, mid_ch=64))
    dec = NetSpec(_dec("stegastamp_decoder", msg_len, size, n_res=4))
    return enc, dec, STEGASTAMP


# ---------------------------------------------------------------------------
# MBRS
# ---------------------------------------------------------------------------
MBRS = {
    "framework": "mbrs",
    "paper": "MBRS: Enhancing Robustness of DNN-based Watermarking by "
             "Mini-Batch of Real and Simulated JPEG Compression (Jia et al., ACM MM 2021)",
    "msg_len": 30, "size": 128,
    "fidelity": "骨干与 HiDDeN 同构 (论文即以多骨干做容器); "
                "贡献在训练侧 mini-batch 真实 JPEG 插桩, 用 train_note 表达",
    "train_note": "mini-batch 内真实 JPEG 压缩与模拟 JPEG 混合 (真 JPEG 插桩), "
                  "两阶段: 预训 100ep + 微调 50ep (lr 1e-3→1e-4)",
    "attacks_train": ["jpeg_real", "jpeg_sim", "crop", "dropout", "gaussian_blur", "gaussian_noise"],
}


def mbrs(msg_len: int = 30, size: int = 128) -> Tuple[NetSpec, NetSpec, dict]:
    enc = NetSpec(_enc("mbrs_encoder", msg_len, size, n_res=4))
    dec = NetSpec(_dec("mbrs_decoder", msg_len, size, n_res=4))
    return enc, dec, MBRS


# ---------------------------------------------------------------------------
# VideoSeal (图像模型侧)
# ---------------------------------------------------------------------------
VIDEOSEAL = {
    "framework": "videoseal",
    "paper": "VideoSeal: Open and efficient video and image watermarking "
             "(Facebook Research, 2024; 含 PixelSeal/ChunkySeal 后续)",
    "msg_len": 256, "size": 128,
    "fidelity": "模块级结构对照 (图像模型侧: 低频嵌入思想 → msg_map 平滑铺图 + "
                "较深解码主干对应鲁棒检测+提取双任务); 时域一致性是视频侧逻辑不在此",
    "train_note": "以可微模糊为主的攻击训练 (blurring), 对物理拍摄/再编码鲁棒; "
                  "检测头=bit logits 阈值 (detect), 提取=同 logits>0",
    "attacks_train": ["blur", "noise", "jpeg_sim", "resize"],
}


def videoseal(msg_len: int = 256, size: int = 128) -> Tuple[NetSpec, NetSpec, dict]:
    enc = NetSpec(_enc("videoseal_encoder", msg_len, size, n_res=3, mid_ch=64))
    dec = NetSpec(_dec("videoseal_decoder", msg_len, size, n_res=5, mid_ch=64))
    return enc, dec, VIDEOSEAL


RECIPES = {
    "hidden": hidden,
    "stegastamp": stegastamp,
    "mbrs": mbrs,
    "videoseal": videoseal,
}


def describe_framework(framework: str, msg_len=None, size=None) -> str:
    """整个框架的系统化 prompt (encoder+decoder+元信息), 给大模型 API 用。"""
    fn = RECIPES[framework]
    kw = {}
    if msg_len:
        kw["msg_len"] = msg_len
    if size:
        kw["size"] = size
    enc, dec, meta = fn(**kw)
    parts = [
        f"# 水印框架系统化描述: {meta['framework']}",
        f"- 论文: {meta['paper']}",
        f"- 消息长: {enc.inputs['message'][0]}, 输入 {enc.inputs['image']}",
        f"- 保真口径: {meta['fidelity']}",
        f"- 训练侧: {meta.get('train_note', '')}",
        f"- 训练期攻击: {meta.get('attacks_train', [])}",
        "",
        enc.render_prompt("## 以下为 encoder"),
        "",
        dec.render_prompt("## 以下为 decoder"),
    ]
    return "\n".join(parts)
