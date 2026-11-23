#!/usr/bin/env python
"""LLM 专用 prompt 工具: 生成/找 bug/调参闭环。

用法:
  python prompt_tool.py describe hidden > prompt.txt          # 框架系统化描述
  python prompt_tool.py debug hidden --problem "BER=0.5 不降" --tb err.txt
  python prompt_tool.py tune hidden --metrics '{"ber":0.21,"psnr":28.5}' \
        --goal "BER<0.05 且 PSNR>30" [--out patch_reply.txt]
  python prompt_tool.py apply hidden --from-reply reply.txt   # 解析回复套补丁
  (可选) python prompt_tool.py ask ...   # 直接调 API (需 WMFLOW_LLM_* 环境变量)

prompt 输出到 stdout, 可直接贴任意大模型; ask/apply 为闭环便利件。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wmflow.netspec import NetSpec
from wmflow.recipes import RECIPES, describe_framework
from wmflow import llm


def get_spec(framework):
    if framework == "custom":
        raise SystemExit("custom 请用 --spec 传 NetSpec JSON 文件")
    enc, dec, meta = RECIPES[framework]()
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["describe", "debug", "tune", "apply", "ask"])
    ap.add_argument("framework")
    ap.add_argument("--spec", help="custom NetSpec JSON 文件路径")
    ap.add_argument("--problem", default="")
    ap.add_argument("--tb", help="报错文本文件路径")
    ap.add_argument("--metrics", default="{}")
    ap.add_argument("--goal", default="降低 BER 同时保持 PSNR")
    ap.add_argument("--constraints", default="")
    ap.add_argument("--from-reply", help="apply 模式: 模型回复文本文件")
    ap.add_argument("--out", help="输出文件 (默认 stdout)")
    args = ap.parse_args()

    if args.spec:
        spec = NetSpec(json.loads(Path(args.spec).read_text(encoding="utf-8")))
    else:
        spec = get_spec(args.framework)

    if args.mode == "describe":
        text = (describe_framework(args.framework) if not args.spec
                else spec.render_prompt())
    elif args.mode == "debug":
        tb = Path(args.tb).read_text(encoding="utf-8") if args.tb else ""
        text = llm.debug_prompt(spec, args.problem, tb)
    elif args.mode == "tune":
        text = llm.tune_prompt(spec, json.loads(args.metrics), args.goal,
                               args.constraints)
    elif args.mode == "apply":
        reply = Path(args.from_reply).read_text(encoding="utf-8")
        new = llm.apply_response(spec, reply)
        text = new.to_json()
    else:  # ask
        prompt = llm.tune_prompt(spec, json.loads(args.metrics), args.goal,
                                 args.constraints)
        text = llm.call_llm(prompt, system="你是水印网络结构与调参专家。")

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
