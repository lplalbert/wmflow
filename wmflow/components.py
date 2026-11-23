"""组件库: 按水印框架收集积木段 (片段), 支持自由拣选组装。

一个"组件"= 一段功能完整的积木链 (如 hidden 的「残差组×4」、videoseal 的
「消息融合段」), 不是单层也不必是整网。切分规则 (自动, 适用任意 NetSpec):
  - 同族连续积木合并成段: res_block 连跑 → 残差组×N; stride2 conv 连跑 →
    下采样组×N; up_block 连跑 → 上采样组×N; 其余单块成段
  - 关键件单列: msg_map/concat (消息融合)、bit_head (比特头)、residual_add (残差输出)

组装 (assemble): 若干段按序拼接 → 合法 NetSpec
  - id 加段前缀防冲突; 段内显式 from 重写, image/message 输入引用透传
  - 隐式串联跨段生效 (NetSpec 的 prev 逻辑); 通道不齐由自动适配插 adapter
  - mode=encoder 声明 image+message 双输入, decoder 声明 image 单输入
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .modules import MODULES
from .netspec import NetSpec

_FAMILY_OF = {
    "res_block": "res", "up_block": "up", "bit_head": "head",
    "residual_add": "out", "se_block": "attention", "dense_block": "dense",
    "msg_map": "fusion", "concat": "fusion",
}
_GROUPABLE = {"res", "down", "up", "conv"}
_FAMILY_LABEL = {
    "res": "残差组", "down": "下采样组", "up": "上采样组", "conv": "卷积组",
}


def _family(mod: Dict[str, Any]) -> str:
    t = mod["type"]
    if t == "conv_block" and (mod.get("params") or {}).get("stride", 1) == 2:
        return "down"
    return _FAMILY_OF.get(t, t)


def segment_spec(spec_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 NetSpec 切成组件段 (含形状与层组成摘要)。"""
    spec = NetSpec(spec_dict)
    built = spec.build()          # 顺带校验; 坏 spec 在此报错
    shape_of = {b.id: (b.in_shapes, b.out_shapes) for b in built.modules}

    segs: List[Dict[str, Any]] = []
    cur: List[Dict[str, Any]] = []

    def flush():
        nonlocal cur
        if not cur:
            return
        fam = _family(cur[0])
        n = len(cur)
        if fam in _GROUPABLE and n > 1:
            label = f"{_FAMILY_LABEL[fam]} ×{n}"
        else:
            label = {
                "msg_map": "消息映射", "concat": "特征汇合",
                "residual_add": "残差输出(+cover)", "bit_head": "比特头",
                "se_block": "SE 注意力", "dense_block": "密集块",
            }.get(cur[0]["type"], cur[0]["type"])
        ids = [m["id"] for m in cur]
        types = (f"{cur[0]['type']}×{n}" if all(m["type"] == cur[0]["type"] for m in cur)
                 else "+".join(m["type"] for m in cur))
        in_s = shape_of[ids[0]][0]
        out_s = shape_of[ids[-1]][1]
        layers = " → ".join(MODULES[m["type"]].layers for m in cur[:3]) + ("…" if n > 3 else "")
        segs.append({
            "key": f"{fam}{len(segs)}", "label": label, "family": fam,
            "types": types, "modules": cur,
            "in": [list(s) for s in in_s], "out": [list(s) for s in out_s],
            "layers": layers,
        })
        cur = []

    for m in spec.modules:
        if cur and _family(m) == _family(cur[0]) and _family(m) in _GROUPABLE:
            cur.append(m)
        else:
            flush()
            cur = [m]
    flush()
    return segs


def single_module_segments() -> List[Dict[str, Any]]:
    """13 个单积木也作为组件列出 (自由混入组装台)。"""
    out = []
    for name in sorted(MODULES):
        md = MODULES[name]
        out.append({
            "key": f"mod_{name}", "label": name, "family": md.category,
            "types": name, "modules": [{"id": "m1", "type": name, "params": {}}],
            "in": [], "out": [], "layers": md.layers,
        })
    return out


def assemble(segments: List[Dict[str, Any]], mode: str = "encoder",
             msg_len: int = 64, size: int = 128) -> Dict[str, Any]:
    """组件段按序拼成 NetSpec dict (自动 id 重写 + 隐式串联)。"""
    inputs: Dict[str, List[int]] = {"image": [3, size, size]}
    if mode == "encoder":
        inputs["message"] = [msg_len]

    modules: List[Dict[str, Any]] = []
    for si, seg in enumerate(segments):
        seg_mods = seg["modules"]
        remap = {m["id"]: f"s{si}_{m['id']}" for m in seg_mods}
        for mod in seg_mods:
            m: Dict[str, Any] = {"id": remap[mod["id"]], "type": mod["type"]}
            if mod.get("params"):
                m["params"] = dict(mod["params"])
            frm = mod.get("from")
            if frm:
                new_frm = []
                for f in frm:
                    if f in ("image", "message") or f in remap:
                        new_frm.append(f if f in ("image", "message") else remap[f])
                    elif modules:
                        # 悬空引用 (如 residual_add 的 "c3" 在别的段) → 回落上一块
                        new_frm.append(modules[-1]["id"])
                    else:
                        new_frm.append(f)   # 首块悬空, 让 build 报清晰错
                m["from"] = new_frm
            modules.append(m)

    if not modules:
        raise ValueError("组装台是空的, 先从组件库拣段")
    return {"name": f"assembled_{mode}", "input": inputs,
            "modules": modules, "output": modules[-1]["id"]}
