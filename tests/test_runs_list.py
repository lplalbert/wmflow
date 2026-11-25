#!/usr/bin/env python
"""op_runs_list 跨 run 汇总回归: 新 run 读 history.json (含曲线), 旧 run 降级为摘要。"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import wmflow.train as T
import webui


def _mk_runs(root):
    # 新 run: 有 history.json
    nd = os.path.join(root, "hidden_1125_100000")
    os.makedirs(nd)
    with open(os.path.join(nd, "history.json"), "w", encoding="utf-8") as f:
        json.dump({
            "framework": "hidden", "recipe": "hidden", "size": 48,
            "msg_len": 30, "dataset_path": "/tmp/ds", "n_images": 12,
            "attacks": ["none"], "epochs": 2, "batch_size": 4, "lr": 1e-3,
            "strength": 1.0, "val_ratio": 0.1, "seed": 0,
            "started_at": 1732500000.0, "updated_at": 1732500060.0,
            "finished": True,
            "history": [
                {"epoch": 1, "loss": 0.9, "loss_img": 0.5, "loss_msg": 0.4,
                 "ber": 0.4, "val_n": 1, "train_n": 11, "val_ber": 0.4},
                {"epoch": 2, "loss": 0.8, "loss_img": 0.4, "loss_msg": 0.4,
                 "ber": 0.3, "val_n": 1, "train_n": 11, "val_ber": 0.3},
            ]}, f)
    # 旧 run: 只有 ckpt (元信息要从 ckpt 抠)
    ld = os.path.join(root, "stegastamp_1124_000000")
    os.makedirs(ld)
    torch.save({"framework": "stegastamp", "size": 128, "msg_len": 256,
                "epoch": 3, "config": {"attacks": ["combo"]}},
               os.path.join(ld, "ckpt_ep3.pt"))
    # 空 run
    os.makedirs(os.path.join(root, "empty_dir"))


def test_runs_list():
    assert "runs_list" in webui.OPS, "OPS 未注册 runs_list"
    with tempfile.TemporaryDirectory() as td:
        _mk_runs(td)
        old = T.RUNS_DIR
        T.RUNS_DIR = td
        try:
            r1 = webui.op_runs_list({})["runs"]
            r2 = webui.op_runs_list({})["runs"]  # 二次调用走缓存路径
            assert r1 == r2, "二次调用结果应一致"
            by = {r["name"]: r for r in r1}
            assert set(by) == {"hidden_1125_100000",
                               "stegastamp_1124_000000", "empty_dir"}
            nw = by["hidden_1125_100000"]
            assert len(nw["history"]) == 2 and nw["finished"] is True
            assert nw["framework"] == "hidden" and nw["size"] == 48
            assert nw["history"][-1]["val_ber"] == 0.3
            assert "legacy" not in nw
            lg = by["stegastamp_1124_000000"]
            assert lg["legacy"] is True and lg["n_ckpts"] == 1
            assert lg["framework"] == "stegastamp" and lg["size"] == 128
            assert lg["epochs"] == 3 and lg["attacks"] == ["combo"]
            em = by["empty_dir"]
            assert em["legacy"] is True and em["n_ckpts"] == 0
            assert em["history"] is None
        finally:
            T.RUNS_DIR = old
    print("[ok] runs_list: new run history curves + legacy ckpt meta + empty run")


if __name__ == "__main__":
    test_runs_list()
    print("ALL RUNS-LIST TESTS PASSED")
