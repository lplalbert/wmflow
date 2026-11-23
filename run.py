#!/usr/bin/env python
"""CLI: python run.py workflows/smoke_ablation.json [--out out/smoke] [--set 5.layer=noise]

--set 支持扫参覆盖: <节点id>.<输入名>=<JSON值>, 可重复。
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from wmflow import Executor, save_report


def parse_set(pairs):
    out = {}
    for p in pairs or []:
        key, _, val = p.partition("=")
        nid, _, name = key.partition(".")
        out[(nid, name)] = json.loads(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--out", default="out/run")
    ap.add_argument("--set", action="append", help="节点.输入=JSON值 覆盖")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    wf = Executor.load(args.workflow)
    for (nid, name), val in parse_set(args.set).items():
        wf["nodes"][nid].setdefault("inputs", {})[name] = val

    errs = Executor.validate(wf)
    if errs:
        print("校验失败:")
        for e in errs:
            print(" -", e)
        sys.exit(1)
    if args.validate_only:
        print("校验通过")
        return

    res = Executor().run(wf, only_outputs=wf.get("outputs"))
    path = save_report(res, wf, args.out)
    for nid, m in res.metrics.items():
        print(f"  {nid}: {m}")
    print(f"缓存 {res.cache_hits}/{len(res.runs)} 命中; 报告 → {path}")


if __name__ == "__main__":
    main()
