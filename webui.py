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
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from wmflow.modules import MODULES
from wmflow.netspec import NetSpec, NetSpecError
from wmflow.recipes import RECIPES, describe_framework
from wmflow import llm

ROOT = Path(__file__).parent


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


OPS = {
    "modules": op_modules, "recipes": op_recipes, "build": op_build,
    "swap": op_swap, "prompt": op_prompt, "apply_reply": op_apply_reply,
    "forward": op_forward,
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

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "webui_index.html").read_bytes(),
                       "text/html; charset=utf-8")
        elif self.path.startswith("/api/"):
            op = self.path[len("/api/"):].split("?")[0]
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
        ln = int(self.headers.get("Content-Length", 0))
        try:
            d = json.loads(self.rfile.read(ln) or b"{}")
        except json.JSONDecodeError as e:
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
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"wmflow webui: http://{args.host}:{args.port}  "
          f"(远程: http://<服务器IP>:{args.port}; 端口不通走 SSH 隧道 "
          f"ssh -L {args.port}:localhost:{args.port} lpl2025@<服务器IP>)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
