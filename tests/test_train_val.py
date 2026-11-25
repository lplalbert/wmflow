#!/usr/bin/env python
"""验证集划分 + val_ber/train_ber 口径回归 (P0)。

用 hidden 框架在 48² 小图上各跑 1 epoch:
- n=12 (>=10) 应划出验证集: 记录 val_ber, val_n=1, train_n=11;
- n=5 (<10) 不划分: 记录 train_ber (不能把训练集指标叫 val_ber)。
纯 CPU 可跑, 文案 ASCII。
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import wmflow.train as T


def _mk(d, n, size):
    rng = np.random.default_rng(0)
    for i in range(n):
        img = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(d, f"i{i}.png"), img)


def _run(n):
    import json
    with tempfile.TemporaryDirectory() as td:
        size = 48
        _mk(td, n, size)
        T.STATE.__init__()
        T.RUNS_DIR = os.path.join(td, "runs")
        run_dir = T.train(dict(dataset_path=td, framework="hidden", size=size,
                               n_images=n, epochs=1, batch_size=4, lr=1e-3,
                               strength=1.0, attacks=["none"], seed=0,
                               val_ratio=0.1, msg_len=30))
        rec = dict(T.STATE.history[-1])
        # history.json 落盘 (临时目录还在, 顺手断言)
        hj = os.path.join(run_dir, "history.json")
        assert os.path.isfile(hj), "应落盘 history.json"
        with open(hj, encoding="utf-8") as f:
            h = json.load(f)
        assert h["finished"] is True and len(h["history"]) == 1, h
        h0 = h["history"][0]
        assert h0["epoch"] == 1 and h0["loss"] == rec["loss"], (h0, rec)
        assert h0["val_n"] == rec["val_n"] and h0["train_n"] == rec["train_n"], h0
        if "val_ber" in rec:
            assert h0["train_n"] + h0["val_n"] == n, h0
        else:  # 未划分: 全量(训练集)评估, train_n == n
            assert h0["train_n"] == n, h0
        assert h["framework"] == "hidden" and h["size"] == 48
        assert h["n_images"] == n and h["attacks"] == ["none"]
        return rec


def test_val_split_labeling():
    r12 = _run(12)
    assert "val_ber" in r12 and "train_ber" not in r12, r12
    assert r12["val_n"] == 1 and r12["train_n"] == 11, r12
    assert 0.0 <= r12["val_ber"] <= 1.0, r12

    r5 = _run(5)
    assert "train_ber" in r5 and "val_ber" not in r5, r5
    assert r5["train_n"] == 5, r5
    print("[ok] n=12 -> val_ber (val_n=1, train_n=11); n=5 -> train_ber; "
          "labeling + history.json correct")


if __name__ == "__main__":
    test_val_split_labeling()
    print("ALL TRAIN-VAL TESTS PASSED")
