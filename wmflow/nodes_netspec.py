"""NetSpec 桥接节点: 配方/积木 JSON → WM_MODEL, 与 prompt 导出节点。

- NetSpecModel: 用 RECIPES 配方或自定义 NetSpec JSON 搭 encoder/decoder,
  产出与 Embed/Decode 节点直连的 WM_MODEL (numpy 接口壳套 torch 链)。
- NetSpecPrompt: 把 NetSpec 渲染成 LLM 专用 prompt (STRING 输出, 可存档/转发)。
"""
from __future__ import annotations

import numpy as np

from .node import Node
from .registry import register
from .types import (IMAGE, INT, FLOAT, STRING, MESSAGE, WM_MODEL, METRICS,
                    Message, WMModel)


@register
class NetSpecModel(Node):
    """按积木配方搭模型: framework(hidden/stegastamp/mbrs) 或 spec_json 自定义。"""
    DISPLAY_NAME = "NetSpec Model"
    CATEGORY = "wm/model"
    RETURN_TYPES = (WM_MODEL,)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "framework": (STRING, {"default": "hidden", "choices": [
                "hidden", "stegastamp", "mbrs", "custom"],
                "help": "内置配方名, 或 'custom:<配方名>' 引用自定义配方库"}),
            "spec_json": (STRING, {"default": "{}", "multiline": True,
                                    "help": "framework=custom 时用: "
                                            '{"encoder": {...}, "decoder": {...}}'}),
            "msg_len": (INT, {"default": 0, "help": "0=用配方默认"}),
            "size": (INT, {"default": 128}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, framework, spec_json, **_):
        if framework.startswith("custom:"):
            from .custom_recipes import exists
            if not exists(framework.split(":", 1)[1]):
                return [f"自定义配方不存在: {framework.split(':', 1)[1]}"]
            return []
        if framework not in ("hidden", "stegastamp", "mbrs", "custom"):
            return [f"未知框架 {framework}"]
        if framework == "custom":
            import json
            try:
                d = json.loads(spec_json or "{}")
            except json.JSONDecodeError as e:
                return [f"spec_json 解析失败: {e}"]
            if "encoder" not in d or "decoder" not in d:
                return ["custom 需要 {encoder:{...}, decoder:{...}}"]
        return []

    def execute(self, framework, spec_json, msg_len, size):
        import json
        from .netspec import NetSpec
        from .recipes import RECIPES
        from . import custom_recipes as crec

        if framework.startswith("custom:") or framework == "custom":
            if framework.startswith("custom:"):
                enc, dec, meta = crec.pair(framework.split(":", 1)[1])
            else:
                d = json.loads(spec_json)
                enc, dec = NetSpec(d["encoder"]), NetSpec(d["decoder"])
                meta = {"framework": "custom"}
        else:
            kw = {"size": size}
            if msg_len:
                kw["msg_len"] = msg_len
            enc, dec, meta = RECIPES[framework](**kw)

        enc_b, dec_b = enc.build(), dec.build()
        enc_mod, dec_mod = enc_b.as_module(), dec_b.as_module()
        enc_mod.eval()
        dec_mod.eval()
        msg_len_eff = enc.inputs["message"][0]

        def encode(cover, bits, strength):
            import torch
            x = torch.from_numpy(np.ascontiguousarray(
                cover.transpose(2, 0, 1)))[None].float()
            m = torch.from_numpy(np.asarray(bits, dtype=np.float32))[None]
            with torch.no_grad():
                y = enc_mod({"image": x, "message": m})[0]
            out = y.permute(1, 2, 0).cpu().numpy()
            return (1 - strength) * cover + strength * np.clip(out, 0, 1)

        def decode(img):
            import torch
            x = torch.from_numpy(np.ascontiguousarray(
                img.transpose(2, 0, 1)))[None].float()
            with torch.no_grad():
                logit = dec_mod({"image": x})[0]
            return (logit > 0.5).cpu().numpy().astype(np.uint8)

        model = WMModel(framework=meta.get("framework", framework),
                        msg_len=msg_len_eff, encode_fn=encode, decode_fn=decode,
                        meta={"netspec": {"encoder": enc.name, "decoder": dec.name}})
        return (model,)


@register
class NetSpecPrompt(Node):
    """渲染 NetSpec/配方为 LLM 专用 prompt 文本 (找 bug/调参用)。"""
    DISPLAY_NAME = "NetSpec Prompt"
    CATEGORY = "wm/llm"
    RETURN_TYPES = (STRING,)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "framework": (STRING, {"default": "hidden", "choices": [
                "hidden", "stegastamp", "mbrs", "custom"],
                "help": "内置配方名, 或 'custom:<配方名>' 引用自定义配方库"}),
            "spec_json": (STRING, {"default": "{}", "multiline": True}),
            "mode": (STRING, {"default": "describe", "choices": [
                "describe", "debug", "tune"]}),
            "problem": (STRING, {"default": "", "multiline": True,
                                  "help": "debug 模式的现象描述"}),
            "metrics_json": (STRING, {"default": "{}", "multiline": True,
                                       "help": "tune 模式的当前指标"}),
            "goal": (STRING, {"default": "降低 BER 同时保持 PSNR"}),
        }}

    def execute(self, framework, spec_json, mode, problem, metrics_json, goal):
        import json
        from .netspec import NetSpec
        from .recipes import RECIPES, describe_framework
        from . import custom_recipes as crec
        from . import llm

        if framework.startswith("custom:"):
            enc, dec, meta = crec.pair(framework.split(":", 1)[1])
            head = [f"# 自定义配方: {meta.get('title', '')}",
                    f"- 消息长: {enc.inputs['message'][0]}, 输入 {enc.inputs['image']}",
                    enc.render_prompt("## 以下为 encoder"), "",
                    dec.render_prompt("## 以下为 decoder")]
            if mode == "describe":
                return ("\n".join(head),)
            spec = enc if mode == "debug" else enc
        elif framework != "custom":
            kw = {}
            enc, dec, meta = RECIPES[framework](**kw)
            if mode == "describe":
                return (describe_framework(framework),)
            spec = enc
        else:
            d = json.loads(spec_json or "{}")
            spec = NetSpec(d.get("encoder") or d)
            if mode == "describe":
                return (spec.render_prompt(),)

        if mode == "debug":
            return (llm.debug_prompt(spec, problem),)
        metrics = json.loads(metrics_json or "{}")
        return (llm.tune_prompt(spec, metrics, goal),)
