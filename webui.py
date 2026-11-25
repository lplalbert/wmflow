#!/usr/bin/env python
"""wmflow 网页积木工作台 —— 跑在服务器上, 浏览器打开使用 (零第三方依赖)。

功能:
  - 积木链可视化: 模块级积木卡片 (显示内部层组成, 不显示单层), 可换积木/改参数
  - 自动适配: 通道不符自动插 adapter, 形状表实时回显
  - 服务器资源: 前向测试在服务器 GPU/CPU 真跑, 返回耗时/设备/形状/参数量
  - LLM 专用 prompt: describe/debug/tune 三模式生成; 贴模型回复 JSON 补丁
    → 自动重适配 → 链更新

启动: python webui.py --host 0.0.0.0 --port 7860
(远程访问: 浏览器开 http://<服务器IP>:7860; 端口不通时用 SSH 隧道
 ssh -L 7860:localhost:7860 lpl2025@<服务器IP>)
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from wmflow.modules import MODULES
from wmflow.netspec import NetSpec, NetSpecError
from wmflow.recipes import RECIPES, describe_framework
from wmflow import custom_recipes as crec
from wmflow import llm

ROOT = Path(__file__).parent

# 访问令牌: 空 = 不鉴权 (默认, 适合本地/SSH 隧道); 非空时 /api/* 必须带
# X-Wmflow-Token 头或 ?token= 查询参数。可用 --token 或 WMFLOW_WEBUI_TOKEN 设置。
TOKEN = ""


# ---------------------------------------------------------------- helpers
def build_report(spec: NetSpec):
    try:
        built = spec.build()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {
        "ok": True,
        "out_shape": list(built.out_shape),
        "auto_adapters": built.auto_adapters,
        "table": [{
            "id": b.id, "type": b.type,
            "in": [list(s) for s in b.in_shapes],
            "out": [list(s) for s in b.out_shapes],
            "params": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in b.params.items()},
        } for b in built.modules],
    }


def op_modules(_d):
    out = {}
    for name, md in MODULES.items():
        out[name] = {
            "layers": md.layers, "category": md.category,
            "n_inputs": md.n_inputs,
            "params": {k: {"type": getattr(v.get("type"), "__name__",
                                          str(v.get("type"))),
                           "default": (list(v["default"])
                                       if isinstance(v.get("default"), tuple)
                                       else v.get("default")),
                           "auto": bool(v.get("auto"))}
                       for k, v in md.params.items()},
            "describe": md.describe(),
        }
    return {"modules": out}


def op_recipes(_d):
    out = {}
    for name, fn in RECIPES.items():
        enc, dec, meta = fn()
        out[name] = {
            "meta": meta,
            "encoder": enc.spec, "decoder": dec.spec,
            "encoder_prompt": enc.render_prompt(),
            "decoder_prompt": dec.render_prompt(),
        }
    return {"recipes": out}


def op_build(d):
    try:
        spec = NetSpec(d["spec"])
    except Exception as e:
        return {"report": {"ok": False, "error": f"spec 解析失败: {e}"}}
    return {"report": build_report(spec), "prompt": spec.render_prompt()}


def op_swap(d):
    try:
        spec = NetSpec(d["spec"])
        new = spec.apply_patch(d.get("patch", []))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "spec": new.spec,
            "report": build_report(new), "prompt": new.render_prompt()}


def op_prompt(d):
    mode = d.get("mode", "describe")
    fw = d.get("framework")
    if fw and fw in RECIPES and d.get("use_framework"):
        if mode == "describe":
            return {"prompt": describe_framework(fw)}
        spec = RECIPES[fw]()[0]
    else:
        spec = NetSpec(d["spec"])
    if mode == "describe":
        text = spec.render_prompt()
    elif mode == "debug":
        text = llm.debug_prompt(spec, d.get("problem", ""), d.get("traceback", ""))
    else:
        text = llm.tune_prompt(spec, d.get("metrics", {}),
                               d.get("goal", "降低 BER 同时保持 PSNR"),
                               d.get("constraints", ""))
    return {"prompt": text}


def op_apply_reply(d):
    try:
        spec = NetSpec(d["spec"])
        new = llm.apply_response(spec, d["reply"])
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "spec": new.spec,
            "report": build_report(new), "prompt": new.render_prompt()}


def op_forward(d):
    try:
        spec_e = NetSpec(d["encoder"])
        spec_d = NetSpec(d["decoder"])
        eb, db = spec_e.build(), spec_d.build()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    try:
        import torch
    except ImportError:
        return {"ok": False, "error": "服务器环境缺 torch"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    em = eb.as_module().to(device).eval()
    dm = db.as_module().to(device).eval()
    msg_len = spec_e.inputs["message"][0]
    size = spec_e.inputs["image"][1]
    n = int(d.get("batch", 2))
    x = torch.rand(n, 3, size, size, device=device)
    m = (torch.rand(n, msg_len, device=device) > 0.5).float()
    try:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            y = em({"image": x, "message": m})
            logit = dm({"image": y})
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
    except Exception as e:
        return {"ok": False, "error": f"前向失败: {e}"}
    params = (sum(p.numel() for p in em.parameters())
              + sum(p.numel() for p in dm.parameters()))
    return {
        "ok": True, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "ms": round(dt, 1),
        "enc_out": list(y.shape), "dec_out": list(logit.shape),
        "params": params,
        "enc_range": [round(float(y.min()), 4), round(float(y.max()), 4)],
    }


# ---------------------------------------------------------------- demo 图像工具
def _png_b64(arr):
    """float [0,1] HWC(1|3 通道) 或灰度 HW → PNG base64。"""
    import base64
    import cv2
    import numpy as np
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    u8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", u8[:, :, ::-1])
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _thumb(t):
    """torch 张量 (B,C,H,W) 或 (B,L) → 缩略图 base64 / 比特列表。"""
    import numpy as np
    a = t.detach().cpu().numpy()[0] if hasattr(t, "detach") else np.asarray(t)[0]
    if a.ndim == 3:  # (C,H,W)
        if a.shape[0] == 3:
            return {"kind": "image", "png": _png_b64(a.transpose(1, 2, 0))}
        m = a.mean(axis=0)                       # 特征图 → 通道均值热图
        lo, hi = float(m.min()), float(m.max())
        m = (m - lo) / (hi - lo + 1e-9)
        return {"kind": "feat", "png": _png_b64(m),
                "ch": a.shape[0], "range": [round(lo, 3), round(hi, 3)]}
    return {"kind": "vec", "values": [round(float(v), 3) for v in a[:256]]}


def _demo_cover(size, upload_b64=None):
    """演示载体: 用户上传图 (base64 dataURL) 或合成纹理图。"""
    import base64
    import cv2
    import numpy as np
    if upload_b64:
        raw = base64.b64decode(upload_b64.split(",", 1)[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (size, size))
            return img[:, :, ::-1].astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack([
        0.5 + 0.4 * np.sin(6 * xx + 3 * yy),
        0.5 + 0.4 * np.sin(9 * yy + np.cos(4 * xx)),
        0.5 + 0.4 * np.cos(7 * (xx + yy)),
    ], axis=-1)
    rng = np.random.default_rng(11)
    img += rng.normal(0, 0.03, img.shape)
    # 画点几何纹理, 利于看水印/攻击差异
    img = np.ascontiguousarray(img)
    cv2.circle(img, (size // 3, size // 2), size // 5, (0.9, 0.2, 0.2), -1)
    cv2.rectangle(img, (size // 2, size // 4), (size * 5 // 6, size * 3 // 4),
                  (0.2, 0.3, 0.9), -1)
    return np.clip(img, 0, 1).astype(np.float32)


def _resolve_demo_model(d):
    """解析演示用模型: ckpt 结构快照 > 显式混搭 spec > 自定义配方 > 内置配方。

    返回 dict: encode/decode/msg_len/size/status/note。
    - ckpt 里存着训练时的 spec_enc/spec_dec (结构快照), 只要没给显式混搭 spec 就用它,
      因此配方被删除/改写后旧 ckpt 仍按原结构回放。
    - status: 'trained' (已载入 ckpt 权重) / 'untrained' (随机初始化, 仅演示结构)。
    """
    import numpy as np
    explicit_mix = bool(d.get("enc_spec"))
    fw = str(d.get("framework") or "hidden")
    recipe_name = d.get("custom_recipe")
    if not recipe_name and fw.startswith("custom:"):
        recipe_name = fw.split(":", 1)[1]

    ck = None
    note = None
    ckpt_path = d.get("ckpt_path")
    if ckpt_path:
        import os
        import torch
        real = os.path.realpath(ckpt_path)
        from wmflow.train import RUNS_DIR
        runs_root = os.path.realpath(RUNS_DIR)
        # 前缀校验必须带分隔符, 否则 /path/runs_evil 会被误判成在 /path/runs 下
        if not (real == runs_root or real.startswith(runs_root + os.sep)):
            raise ValueError("ckpt 须在 runs/ 目录下")
        try:
            ck = torch.load(real, map_location="cpu", weights_only=False)
        except Exception as e:
            raise ValueError(f"ckpt 读取失败: {e}")

    if ck is not None and not explicit_mix:
        # 结构快照优先: 配方文件在不在都不影响回放
        enc, dec = NetSpec(ck["spec_enc"]), NetSpec(ck["spec_dec"])
        size = int(ck.get("size") or enc.inputs["image"][1])
        src = ck.get("recipe") or ck.get("framework") or "ckpt"
        if recipe_name:
            try:
                cur_e, cur_d, _m = crec.pair(recipe_name)
                same = (cur_e.spec.get("modules") == ck["spec_enc"].get("modules")
                        and cur_d.spec.get("modules") == ck["spec_dec"].get("modules"))
                if not same:
                    note = (f"配方 {recipe_name} 的结构已被改动, "
                            f"实际按 ckpt 训练时的结构快照渲染")
            except Exception:
                note = f"配方 {recipe_name} 已不存在, 按 ckpt 的结构快照渲染"
        elif fw not in ("custom", "lsb_toy", src):
            note = (f"已按 ckpt 结构快照 ({src} {size}²/"
                    f"{ck.get('msg_len')}bit) 渲染, 忽略所选框架 {fw}")
    else:
        size = int(d.get("size", 128))
        if explicit_mix:
            enc, dec = NetSpec(d["enc_spec"]), NetSpec(d["dec_spec"])
            size = int(d["enc_spec"]["input"]["image"][1])
            # dec 的 bit_head 消息长对齐 enc (跨框架混搭 30bit <-> 256bit)
            want = enc.inputs["message"][0]
            heads = [{"id": m["id"], "params": {"msg_len": want}}
                     for m in dec.modules if m["type"] == "bit_head"]
            if heads and dec.modules[-1].get("params", {}).get("msg_len") != want:
                dec = dec.apply_patch(heads)
        elif recipe_name:
            enc, dec, _meta = crec.pair(recipe_name)
        else:
            if fw not in RECIPES:
                raise ValueError(f"未知框架 '{fw}' (内置: {sorted(RECIPES)}, "
                                 f"或自定义配方用 custom:<名>)")
            enc, dec, _meta = RECIPES[fw]()

    eb, db = enc.build(), dec.build()
    msg_len = int(enc.inputs["message"][0])
    enc_mod, dec_mod = eb.as_module(), db.as_module()   # 只建一次, 权重就落这个实例
    enc_mod.eval()
    dec_mod.eval()
    status = "untrained"
    if ck is not None:
        try:
            enc_mod.load_state_dict(ck["enc"])
            dec_mod.load_state_dict(ck["dec"])
        except Exception as e:
            raise ValueError(f"ckpt 与当前结构不匹配 (权重无法载入): {e}")
        status = "trained"

    def encode(cover, bits, s):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(
            cover.transpose(2, 0, 1)))[None].float()
        m = torch.from_numpy(np.asarray(bits, dtype=np.float32))[None]
        with torch.no_grad():
            y = enc_mod({"image": x, "message": m})[0]
        out = y.permute(1, 2, 0).cpu().numpy()
        return (1 - s) * cover + s * np.clip(out, 0, 1)

    def decode(img):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(
            img.transpose(2, 0, 1)))[None].float()
        with torch.no_grad():
            logit = dec_mod({"image": x})[0]
        # bit_head 出的是 logits (Linear 无激活), BCEWithLogits 的判决边界是 0
        return (logit > 0).cpu().numpy().astype(np.uint8)

    return {"encode": encode, "decode": decode, "msg_len": msg_len,
            "size": size, "status": status, "note": note}


def op_run_demo(d):
    """全流程真跑: 载体→嵌入→攻击→提取, 逐节点输出可视化 + 指标。"""
    import numpy as np
    d = dict(d)
    fw = str(d.get("framework") or "hidden")
    explicit_mix = bool(d.get("enc_spec"))
    if fw == "lsb_toy" and not explicit_mix and not d.get("ckpt_path"):
        from wmflow.nodes_models import make_lsb_toy
        size = int(d.get("size", 128))
        msg_len = int(d.get("msg_len", 64))
        toy = make_lsb_toy(msg_len)
        encode, decode = toy.encode_fn, toy.decode_fn
        status = "algorithmic"
        note = "LSB 玩具嵌入 (确定性算法, 非训练权重)"
    else:
        try:
            m = _resolve_demo_model(d)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        encode, decode = m["encode"], m["decode"]
        msg_len, size, status, note = (m["msg_len"], m["size"],
                                       m["status"], m["note"])

    strength = float(d.get("strength", 1.0))
    seed = int(d.get("seed", 0))
    layer = d.get("layer", "none")
    preset = d.get("preset", "none")

    cover = _demo_cover(size, d.get("cover_upload"))
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=msg_len).astype(np.uint8)
    embedded = encode(cover, bits, strength)
    attacked = embedded.copy()
    attack_params = {}
    if layer not in ("", "none"):
        from wmflow.nodes_core import run_layer
        attacked, attack_params = run_layer(
            layer, attacked, np.random.default_rng(seed + 1))
    if preset not in ("", "none"):
        from wmflow.nodes_core import TranscodeChain
        attacked, p2 = TranscodeChain().execute(attacked, preset)
        attack_params.update(p2)
    dec_bits = decode(attacked)
    L = min(len(dec_bits), len(bits))   # 防御: 组装头长不齐时按短边比较
    errs = int(np.sum(dec_bits[:L] != bits[:L]))
    mse = float(np.mean((embedded - cover) ** 2))
    mse_a = float(np.mean((attacked - cover) ** 2))

    return {
        "ok": True, "trained": status == "trained", "model_status": status,
        "msg_len": msg_len, "structure_note": note,
        "attack_params": attack_params,
        "nodes": {
            "cover": {"kind": "image", "png": _png_b64(cover)},
            "message": {"kind": "bits", "bits": bits.tolist()},
            "embedded": {"kind": "image", "png": _png_b64(embedded),
                         "psnr": round(10 * np.log10(1.0 / max(mse, 1e-10)), 2)},
            "attacked": {"kind": "image", "png": _png_b64(attacked),
                         "psnr": round(10 * np.log10(1.0 / max(mse_a, 1e-10)), 2)},
            "decoded": {"kind": "bits", "bits": dec_bits.tolist(),
                        "ref": bits.tolist()},
            "metrics": {"ber": round(errs / max(L, 1), 4),
                        "bit_errors": errs, "n_bits": int(L),
                        "psnr_embed": round(10 * np.log10(1.0 / max(mse, 1e-10)), 2),
                        "psnr_attacked": round(10 * np.log10(1.0 / max(mse_a, 1e-10)), 2)},
        },
    }


def op_trace(d):
    """逐积木前向追踪: 每块输出缩略图 (特征图取通道均值) + 形状。"""
    spec = NetSpec(d["spec"])
    built = spec.build()
    import numpy as np
    try:
        import torch
    except ImportError:
        return {"ok": False, "error": "缺 torch"}
    size = next(v[1] for v in spec.inputs.values() if len(v) == 3)
    msg_len = next((v[0] for v in spec.inputs.values() if len(v) == 1), None)
    inp = {}
    if "image" in spec.inputs:
        inp["image"] = torch.from_numpy(np.ascontiguousarray(
            _demo_cover(size).transpose(2, 0, 1)))[None].float()
    if msg_len and "message" in spec.inputs:
        rng = np.random.default_rng(0)
        inp["message"] = torch.from_numpy(
            rng.integers(0, 2, size=msg_len).astype(np.float32))[None]
    out, env = built.forward_trace(inp)
    thumbs = {}
    for k, v in env.items():
        thumbs[k] = {"shape": list(v.shape) if hasattr(v, "shape") else None,
                     **_thumb(v)}
    return {"ok": True, "output": built.output_id, "thumbs": thumbs}


# ---------------------------------------------------------------- 数据集/训练
_env_roots = os.environ.get("WMFLOW_FS_ROOTS")
FS_ROOTS = ([r for r in _env_roots.split(os.pathsep) if r] if _env_roots
            else ["/home/lpl2025", "/data", "/mnt", "/root"])


def op_fs(d):
    """服务器目录浏览: 子目录 + 图片数 (数据集选择用)。"""
    import os
    from wmflow.train import scan_dataset
    path = d.get("path", "")
    if not path:
        return {"roots": [r for r in FS_ROOTS if os.path.isdir(r)]}
    real = os.path.realpath(path)
    if not any(real == r or real.startswith(r + os.sep) for r in FS_ROOTS):
        return {"error": f"路径须在 {FS_ROOTS} 之下"}
    if not os.path.isdir(real):
        return {"error": f"不是目录: {real}"}
    info = scan_dataset(real)
    info["parent"] = os.path.dirname(real.rstrip("/")) or "/"
    return info


def op_train_start(d):
    """启动训练。数据集路径必须落在 FS_ROOTS 白名单内 (与目录浏览同一约束,
    否则可绕过白名单让服务器读任意目录); 顺带校验 val_ratio 合法性。"""
    from wmflow.train import start_training, ATTACKS
    import os
    cfg = dict(d)
    dp = str(cfg.get("dataset_path", "") or "")
    if not dp:
        return {"ok": False, "error": "缺数据集路径 dataset_path"}
    real = os.path.realpath(dp)
    roots = [os.path.realpath(r) for r in FS_ROOTS]
    if not any(real == r or real.startswith(r + os.sep) for r in roots):
        return {"ok": False, "error": f"数据集路径须在 {FS_ROOTS} 之下"}
    if not os.path.isdir(real):
        return {"ok": False, "error": f"数据集目录不存在: {real}"}
    try:
        vr = float(cfg.get("val_ratio", 0.1))
    except (TypeError, ValueError):
        return {"ok": False, "error": "val_ratio 必须是数字"}
    if not (0.0 <= vr < 1.0):
        return {"ok": False, "error": f"val_ratio 须在 [0,1) 区间, 当前 {vr}"}
    cfg["val_ratio"] = vr
    cfg["dataset_path"] = real
    att = [a for a in cfg.get("attacks", []) if a in ATTACKS]
    cfg["attacks"] = att or ["combo"]
    return {"ok": True, "result": start_training(cfg)}


def op_train_status(_d):
    from wmflow.train import STATE
    return {"ok": True, "state": STATE.snapshot()}


def op_train_stop(_d):
    from wmflow.train import STATE
    STATE.stop_flag = True
    return {"ok": True, "msg": "停止信号已发"}


def op_runs(_d):
    """列出训练产出 ckpt (供 run_demo 加载)。"""
    from wmflow.train import RUNS_DIR
    import os
    out = []
    if os.path.isdir(RUNS_DIR):
        for name in sorted(os.listdir(RUNS_DIR), reverse=True)[:10]:
            rd = os.path.join(RUNS_DIR, name)
            cks = sorted(f for f in os.listdir(rd) if f.endswith(".pt"))
            if cks:
                out.append({"run": name,
                            "ckpt": os.path.join(rd, cks[-1]),
                            "n_ckpts": len(cks)})
    return {"runs": out}


def op_components(_d):
    """组件库: 各框架的积木段货架 + 自由单积木 + 我的配方。"""
    from wmflow.components import segment_spec, single_module_segments
    groups = []
    for fw, fn in RECIPES.items():
        enc, dec, meta = fn()
        groups.append({
            "framework": fw, "paper": meta.get("paper", ""),
            "encoder": segment_spec(enc.spec),
            "decoder": segment_spec(dec.spec),
        })
    return {"groups": groups, "modules": single_module_segments(),
            "custom": crec.list_all()}


def op_assemble(d):
    """组件段拼装 → NetSpec + 形状报告 + prompt。"""
    from wmflow.components import assemble
    try:
        spec_dict = assemble(d.get("segments", []),
                             d.get("mode", "encoder"),
                             int(d.get("msg_len", 64)),
                             int(d.get("size", 128)))
        ns = NetSpec(spec_dict)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "spec": ns.spec, "report": build_report(ns),
            "prompt": ns.render_prompt()}


# ---------------------------------------------------------------- 自定义配方
def _recipe_side(side, part):
    """解析配方一侧网络的来源: 组装台 bench(spec) / 内置框架 framework / 已存 custom。

    只允许一种来源字段; 同时给多个直接拒绝, 免得多余字段被静默忽略 (曾默认按 kind 取)。
    """
    if not isinstance(side, dict) or not side:
        raise ValueError(f"缺 {part} 侧网络 (可来自组装台/内置框架/已存配方)")
    given = [k for k in ("spec", "framework", "name") if side.get(k)]
    if len(given) > 1:
        raise ValueError(f"{part} 侧同时给了 {given}, 只能选一种来源")
    kind = side.get("kind") or (
        "bench" if side.get("spec") else
        "framework" if side.get("framework") else
        "custom" if side.get("name") else "")
    if kind == "bench":
        if not side.get("spec"):
            raise ValueError(f"{part} 侧声明来自组装台但没有 spec (先点构建校验)")
        return side["spec"]
    if kind == "framework":
        fw = side.get("framework")
        if fw not in RECIPES:
            raise ValueError(f"未知内置框架 '{fw}', 可选: {sorted(RECIPES)}")
        kw = {}
        if side.get("msg_len"):
            kw["msg_len"] = int(side["msg_len"])
        if side.get("size"):
            kw["size"] = int(side["size"])
        enc, dec, _meta = RECIPES[fw](**kw)
        return (enc if part == "encoder" else dec).spec
    if kind == "custom":
        enc, dec, _meta = crec.pair(side.get("name"))
        return (enc if part == "encoder" else dec).spec
    raise ValueError(f"不认识的来源 kind='{kind}' (bench/framework/custom)")


def op_recipe_save(d):
    """把组装台/混搭结果存成自定义配方 (自动对齐 dec 消息长 + 硬校验)。"""
    name = (d.get("name") or "").strip()
    if not name:
        return {"ok": False, "error": "配方名不能为空"}
    try:
        enc_spec = _recipe_side(d.get("encoder"), "encoder")
        dec_spec = _recipe_side(d.get("decoder"), "decoder")
        summary = crec.save(name, enc_spec, dec_spec,
                            title=d.get("title"), note=d.get("note", ""),
                            source=d.get("source"),
                            overwrite=bool(d.get("overwrite")))
    except FileExistsError as e:
        return {"ok": False, "exists": True, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "recipe": summary}


def op_recipe_list(_d):
    """列出全部自定义配方。"""
    return {"ok": True, "recipes": crec.list_all(),
            "dir": str(crec.CUSTOM_DIR)}


def op_recipe_get(d):
    """取一条配方: 两张 NetSpec + 形状报告 + prompt + 拆好的积木段 (可回组装台续改)。"""
    from wmflow.components import segment_spec
    try:
        rec = crec.get(d["name"])
    except Exception as e:
        return {"ok": False, "error": str(e)}
    enc, dec = NetSpec(rec["encoder"]), NetSpec(rec["decoder"])
    return {"ok": True, "recipe": rec,
            "report": {"encoder": build_report(enc), "decoder": build_report(dec)},
            "segments": {"encoder": segment_spec(enc.spec),
                         "decoder": segment_spec(dec.spec)},
            "prompt": enc.render_prompt() + "\n\n" + dec.render_prompt()}


def op_recipe_delete(d):
    try:
        crec.delete(d["name"])
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "name": d["name"]}


def op_recipe_export(d):
    """导出为 JSON 文本 (前端可复制/下载)。"""
    try:
        fname, text = crec.export(d["name"])
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "filename": fname, "json": text}


def op_recipe_import(d):
    """从粘贴的 JSON 导入配方 (name 可改名, overwrite 覆盖)。"""
    payload = d.get("json") or d.get("payload")
    if not payload:
        return {"ok": False, "error": "缺 JSON 内容"}
    try:
        summary = crec.import_json(payload, name=d.get("name") or None,
                                   overwrite=bool(d.get("overwrite")))
    except FileExistsError as e:
        return {"ok": False, "exists": True, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "recipe": summary}


OPS = {
    "modules": op_modules, "recipes": op_recipes, "build": op_build,
    "swap": op_swap, "prompt": op_prompt, "apply_reply": op_apply_reply,
    "forward": op_forward, "run_demo": op_run_demo, "trace": op_trace,
    "fs": op_fs, "train_start": op_train_start, "train_status": op_train_status,
    "train_stop": op_train_stop, "runs": op_runs,
    "components": op_components, "assemble": op_assemble,
    "recipe_save": op_recipe_save, "recipe_list": op_recipe_list,
    "recipe_get": op_recipe_get, "recipe_delete": op_recipe_delete,
    "recipe_export": op_recipe_export, "recipe_import": op_recipe_import,
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(
            body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self):
        """token 为空则不鉴权; 否则校验请求头或 query 里的 token。"""
        if not TOKEN:
            return True
        given = self.headers.get("X-Wmflow-Token", "")
        if not given:
            given = parse_qs(urlparse(self.path).query).get("token", [""])[0]
        return hmac.compare_digest(str(given), TOKEN)

    def _deny(self):
        self._send(401, {"ok": False,
                         "error": "未授权: 请在 URL 加 ?token=... 或在请求头带 X-Wmflow-Token"})

    def do_GET(self):
        # 先剥掉 ?query 再匹配路径: 否则 `/?token=...` 这种带参首页会落到 404
        # (token 鉴权模式下文档/e2e 都用 `/?token=xxx` 打开页面)。
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, (ROOT / "webui_index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif path == "/favicon.ico":
            self._send(204, b"")
        elif path.startswith("/api/"):
            if not self._authorized():
                self._deny()
                return
            op = path[len("/api/"):]
            if op in OPS:
                try:
                    self._send(200, OPS[op]({}))
                except Exception as e:
                    self._send(500, {"error": str(e)})
            else:
                self._send(404, {"error": f"未知接口 {op}"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            self._deny()
            return
        ln = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(ln) or b"{}"
        try:
            # 浏览器是 UTF-8; Windows 下 curl/PowerShell 手搓请求可能是 GBK,
            # 先按 UTF-8 解, 失败再按 gb18030 兜一次 (中文配方名/备注别直接 400)。
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("gb18030")
            d = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            self._send(400, {"error": f"JSON 解析失败: {e}"})
            return
        op = self.path[len("/api/"):].split("?")[0] if self.path.startswith("/api/") else ""
        if op not in OPS:
            self._send(404, {"error": f"未知接口 {op}"})
            return
        try:
            self._send(200, OPS[op](d))
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, fmt, *args):  # 安静点
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--token", default=None,
                    help="访问令牌; 缺省读 WMFLOW_WEBUI_TOKEN, 为空则不鉴权")
    args = ap.parse_args()
    global TOKEN
    TOKEN = args.token or os.environ.get("WMFLOW_WEBUI_TOKEN") or ""
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    tip = " (已启用 token 鉴权)" if TOKEN else " (未鉴权, 勿映射公网)"
    print(f"wmflow webui: http://{args.host}:{args.port}{tip}  "
          f"(远程: http://<服务器IP>:{args.port}; 端口不通走 SSH 隧道 "
          f"ssh -L {args.port}:localhost:{args.port} lpl2025@<服务器IP>)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
