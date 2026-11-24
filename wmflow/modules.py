"""网络模块积木注册表 (粒度约定见 README §模块粒度)。

三层粒度:
  层 (Conv/BN/ReLU)     —— 不作为积木, 埋进模块内部, 只在 LAYERS 描述串里显示
  模块 (本文件)          —— 2~5 层的积木, 可显示/可换/可自动适配 ← 主粒度
  网络 (NetSpec)         —— 模块串成 encoder/decoder

每个模块 = 一个 factory:
    build(in_shapes: [(C,H,W)...], **params) -> (callable, out_shapes)
返回的 callable 收 torch.Tensor (B,C,H,W) 或列表; 形状推理与前向共用参数。

自动适配 (用户需求):
  - in_ch 不填 = 自动继承上游 out_ch
  - 上游 out_ch 与本模块显式 in_ch 不符 → 自动插 1x1 adapter (记录在 spec.auto_adapters)
  - msg_map 的目标空间尺寸自动取 image 输入的 H,W
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Shape = Tuple[int, int, int]  # (C,H,W)

MODULES: Dict[str, "ModuleDef"] = {}


class ModuleDef:
    def __init__(self, name: str, layers: str, category: str,
                 params: Dict[str, Dict[str, Any]], n_inputs: int = 1,
                 build_fn: Optional[Callable] = None):
        self.name = name
        self.layers = layers          # 层组成描述串 (显示用)
        self.category = category
        self.params = params          # {名: {type, default, auto, help, range}}
        self.n_inputs = n_inputs
        self.build_fn = build_fn

    def build(self, in_shapes: Sequence[Shape], **params):
        return self.build_fn(in_shapes, **params)

    def describe(self) -> str:
        ps = ", ".join(
            f"{k}: {v['type'].__name__ if isinstance(v['type'], type) else v['type']}"
            f"={v.get('default')}" + (" (auto)" if v.get("auto") else "")
            for k, v in self.params.items())
        return (f"- **{self.name}** [{self.category}] 层组成: `{self.layers}`\n"
                f"  参数: {ps or '(无)'}; 输入数: {self.n_inputs}")


def register_module(name: str, layers: str, category: str,
                    params: Dict[str, Dict[str, Any]], n_inputs: int = 1):
    def deco(fn):
        MODULES[name] = ModuleDef(name, layers, category, params, n_inputs, fn)
        return fn
    return deco


# ---------------------------------------------------------------------------
# 形状工具
# ---------------------------------------------------------------------------
def _conv_hw(h: int, w: int, k: int, s: int) -> Tuple[int, int]:
    return ((h + 2 * (k // 2) - k) // s + 1, (w + 2 * (k // 2) - k) // s + 1)


def _act():
    import torch.nn as nn
    return nn.ReLU(inplace=True)


# ---------------------------------------------------------------------------
# 基础积木
# ---------------------------------------------------------------------------
@register_module(
    "conv_block", "Conv3x3(s)→BN→ReLU", "basic",
    {"in_ch": {"type": int, "default": None, "auto": True},
     "out_ch": {"type": int, "default": None, "auto": True,
                "help": "缺省=in_ch"},
     "stride": {"type": int, "default": 1}})
def _conv_block(in_shapes, in_ch=None, out_ch=None, stride=1):
    import torch.nn as nn
    c, h, w = in_shapes[0]
    in_ch = in_ch or c
    out_ch = out_ch or in_ch
    m = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride, 1),
                      nn.BatchNorm2d(out_ch), _act())
    oh, ow = _conv_hw(h, w, 3, stride)
    return m, [(out_ch, oh, ow)]


@register_module(
    "down_block", "Conv3x3(s2)→BN→ReLU→Conv3x3→BN→ReLU", "basic",
    {"in_ch": {"type": int, "default": None, "auto": True},
     "out_ch": {"type": int, "default": None, "auto": True}})
def _down_block(in_shapes, in_ch=None, out_ch=None):
    import torch.nn as nn
    c, h, w = in_shapes[0]
    in_ch = in_ch or c
    out_ch = out_ch or in_ch * 2
    m = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 2, 1), nn.BatchNorm2d(out_ch),
                      _act(), nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                      nn.BatchNorm2d(out_ch), _act())
    oh, ow = _conv_hw(h, w, 3, 2)
    return m, [(out_ch, oh, ow)]


@register_module(
    "up_block", "ConvT4x4(s2)→BN→ReLU→Conv3x3→BN→ReLU", "basic",
    {"in_ch": {"type": int, "default": None, "auto": True},
     "out_ch": {"type": int, "default": None, "auto": True}})
def _up_block(in_shapes, in_ch=None, out_ch=None):
    import torch.nn as nn
    c, h, w = in_shapes[0]
    in_ch = in_ch or c
    out_ch = out_ch or max(c // 2, 8)
    m = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                      nn.BatchNorm2d(out_ch), _act(),
                      nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                      nn.BatchNorm2d(out_ch), _act())
    return m, [(out_ch, h * 2, w * 2)]


@register_module(
    "res_block", "Conv3x3→BN→ReLU→Conv3x3→BN (+skip)→ReLU", "basic",
    {"ch": {"type": int, "default": None, "auto": True}})
def _res_block(in_shapes, ch=None):
    import torch.nn as nn

    class Res(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.f = nn.Sequential(nn.Conv2d(ch, ch, 3, 1, 1),
                                   nn.BatchNorm2d(ch), _act(),
                                   nn.Conv2d(ch, ch, 3, 1, 1),
                                   nn.BatchNorm2d(ch))

        def forward(self, x):
            return _act()(self.f(x) + x)

    c = ch or in_shapes[0][0]
    return Res(c), [(c,) + in_shapes[0][1:]]


@register_module(
    "se_block", "GAP→FC→ReLU→FC→Sigmoid× (+skip)", "attention",
    {"ch": {"type": int, "default": None, "auto": True},
     "reduction": {"type": int, "default": 16}})
def _se_block(in_shapes, ch=None, reduction=16):
    import torch.nn as nn

    class SE(nn.Module):
        def __init__(self, ch, r):
            super().__init__()
            self.f = nn.Sequential(nn.Linear(ch, max(ch // r, 4)), _act(),
                                   nn.Linear(max(ch // r, 4), ch), nn.Sigmoid())

        def forward(self, x):
            w = self.f(x.mean((2, 3)))
            return x * w.unsqueeze(-1).unsqueeze(-1)

    c = ch or in_shapes[0][0]
    return SE(c, reduction), [(c,) + in_shapes[0][1:]]


@register_module(
    "dense_block", " [Conv3x3→BN→ReLU→cat]×n (growth)", "basic",
    {"in_ch": {"type": int, "default": None, "auto": True},
     "growth": {"type": int, "default": 16},
     "n_layers": {"type": int, "default": 3}})
def _dense_block(in_shapes, in_ch=None, growth=16, n_layers=3):
    import torch
    import torch.nn as nn

    class Dense(nn.Module):
        def __init__(self, c0, g, n):
            super().__init__()
            self.convs = nn.ModuleList([
                nn.Sequential(nn.Conv2d(c0 + g * i, g, 3, 1, 1),
                              nn.BatchNorm2d(g), _act())
                for i in range(n)])
            self.out_ch = c0 + g * n

        def forward(self, x):
            feats = [x]
            for conv in self.convs:
                out = conv(torch.cat(feats, 1))
                feats.append(out)
            return torch.cat(feats, 1)

    c = in_ch or in_shapes[0][0]
    m = Dense(c, growth, n_layers)
    return m, [(m.out_ch,) + in_shapes[0][1:]]


@register_module(
    "msg_map", "Linear(msg→C·H·W)→reshape→Conv3x3→BN→ReLU", "message",
    {"msg_len": {"type": int, "default": None, "auto": True},
     "out_ch": {"type": int, "default": 1},
     "target_hw": {"type": tuple, "default": None, "auto": True,
                   "help": "自动取 image 输入空间尺寸"}})
def _msg_map(in_shapes, msg_len=None, out_ch=1, target_hw=None):
    import torch.nn as nn
    ml = msg_len or in_shapes[0][0]
    if target_hw is None:
        raise ValueError("msg_map 需要 target_hw (NetSpec 自动从 image 形状注入)")
    th, tw = target_hw

    class MsgMap(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(ml, out_ch * th * tw)
            self.conv = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, 1, 1),
                                      nn.BatchNorm2d(out_ch), _act())
            self.shape = (out_ch, th, tw)

        def forward(self, m):
            x = self.fc(m).view(m.shape[0], out_ch, th, tw)
            return self.conv(x)

    return MsgMap(), [(out_ch, th, tw)]


@register_module(
    "concat", "torch.cat(axis=1)→Conv3x3→BN→ReLU", "fusion",
    {"out_ch": {"type": int, "default": None, "auto": True}}, n_inputs=2)
def _concat(in_shapes, out_ch=None):
    import torch
    import torch.nn as nn
    cin = sum(s[0] for s in in_shapes)
    h, w = in_shapes[0][1:]
    for s in in_shapes[1:]:
        if s[1:] != (h, w):
            raise ValueError(f"concat 空间尺寸不齐: {in_shapes}")
    oc = out_ch or cin

    class Cat(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(cin, oc, 3, 1, 1),
                                      nn.BatchNorm2d(oc), _act())

        def forward(self, xs):
            return self.conv(torch.cat(xs, 1))

    return Cat(), [(oc, h, w)]


@register_module(
    "residual_add", "out = clamp(x + cover)", "fusion",
    {}, n_inputs=2)
def _residual_add(in_shapes, **_):
    import torch
    import torch.nn as nn

    class ResAdd(nn.Module):
        def forward(self, xs):
            x, cover = xs
            return (x + cover).clamp(0, 1)

    # 形状取 cover (x+cover 广播: x 通道少时向 cover 对齐)
    return ResAdd(), [in_shapes[1]]


@register_module(
    "bit_head", "GAP→Linear(C→msg_len)→logits", "head",
    {"in_ch": {"type": int, "default": None, "auto": True},
     "msg_len": {"type": int, "default": None, "auto": True}})
def _bit_head(in_shapes, in_ch=None, msg_len=None):
    import torch.nn as nn
    c = in_ch or in_shapes[0][0]
    if msg_len is None:
        raise ValueError("bit_head 需要 msg_len (NetSpec 从消息长度注入)")

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(c, msg_len)

        def forward(self, x):
            return self.fc(x.mean((2, 3)))

    return Head(), [(msg_len,)]


@register_module(
    "clamp_img", "clamp(0,1)  [不改形状]", "util", {})
def _clamp_img(in_shapes, **_):
    import torch.nn as nn

    class Clamp(nn.Module):
        def forward(self, x):
            return x.clamp(0, 1)

    return Clamp(), [in_shapes[0]]


@register_module(
    "identity", "直通", "util", {})
def _identity(in_shapes, **_):
    import torch.nn as nn
    return nn.Identity(), [in_shapes[0]]


@register_module(
    "adapter1x1", "Conv1x1→BN→ReLU  [自动插适配]", "auto",
    {"in_ch": {"type": int, "default": None, "auto": True},
     "out_ch": {"type": int, "default": None, "auto": True}})
def _adapter1x1(in_shapes, in_ch=None, out_ch=None):
    import torch.nn as nn
    c, h, w = in_shapes[0]
    in_ch = in_ch or c
    out_ch = out_ch or c
    m = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), _act())
    return m, [(out_ch, h, w)]
