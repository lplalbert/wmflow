"""P1 测试: 自定义配方进训练面板 + ckpt 结构快照绑定 + 权重回放真实性。

python tests/test_train_custom.py 即可跑 (CPU 可跑, 用 3 张 64² 小图 1 epoch)。
文案全 ASCII, 防 Windows GBK 控制台炸。
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmflow import custom_recipes as crec
from wmflow.netspec import NetSpec
from wmflow.recipes import RECIPES
import wmflow.train as T


class temp_store:
    """配方目录指到 tempdir, 不污染仓库 custom_recipes/。"""

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        self._old = crec.CUSTOM_DIR
        crec.CUSTOM_DIR = Path(self._td.name)
        return crec.CUSTOM_DIR

    def __exit__(self, *exc):
        crec.CUSTOM_DIR = self._old
        self._td.cleanup()


def _make_dataset(d, n=3, size=64):
    import cv2
    rng = np.random.default_rng(0)
    for i in range(n):
        img = (rng.random((size, size, 3)) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(d, f"img{i}.png"), img)


def test_train_custom_recipe():
    """custom:<名> 能被训练环接受; 配方尺寸覆盖表单尺寸; ckpt 带结构快照。"""
    import torch
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        crec.save("trainme", enc.spec, dec.spec, title="可训配方")
        with tempfile.TemporaryDirectory() as td:
            ds = os.path.join(td, "ds")
            os.makedirs(ds)
            _make_dataset(ds, 3, 64)
            runs = os.path.join(td, "runs")
            old = T.RUNS_DIR
            T.RUNS_DIR = runs
            try:
                T.STATE.__init__()
                T.STATE.running = True
                T.STATE.started_at = T.time.time()
                run_dir = T.train({"framework": "custom:trainme",
                                   "dataset_path": ds, "n_images": 3,
                                   "epochs": 1, "batch_size": 2, "lr": 1e-3,
                                   "strength": 1.0, "w_img": 1.0, "w_msg": 1.0,
                                   "seed": 0, "size": 32,     # 故意与配方不符
                                   "attacks": ["none"]})
                log = list(T.STATE.log)
            finally:
                T.RUNS_DIR = old
            cks = sorted(p for p in os.listdir(run_dir) if p.endswith(".pt"))
            assert cks, "没产出 ckpt"
            ck = torch.load(os.path.join(run_dir, cks[-1]),
                            map_location="cpu", weights_only=False)
            assert ck["size"] == 128, f"配方尺寸应覆盖表单 32, got {ck['size']}"
            assert ck["msg_len"] == 30
            assert ck["spec_enc"] and ck["spec_dec"], "ckpt 必须带结构快照"
            assert ck["recipe"] == "custom:trainme", ck.get("recipe")
            assert any("梯度自检通过" in x for x in log), log[-3:]
            assert any("已忽略表单尺寸" in x for x in log), "应提示尺寸被覆盖"
            print(f"[ok] train custom:<name>: 1 epoch done, size 32->128, "
                  f"ckpt 带 spec 快照, 梯度自检通过")


def test_ckpt_weights_actually_applied():
    """回放必须用 ckpt 的真权重, 不是随机权重 (旧 bug: 每次 as_module() 新建实例)。"""
    import torch
    import webui as W
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        crec.save("wp", enc.spec, dec.spec)
        e = enc.build().as_module()
        d = dec.build().as_module()
        ck = {"enc": e.state_dict(), "dec": d.state_dict(),
              "spec_enc": enc.spec, "spec_dec": dec.spec,
              "size": 128, "msg_len": 30,
              "recipe": "custom:wp", "framework": "custom:wp"}
        with tempfile.TemporaryDirectory() as td:
            old = T.RUNS_DIR
            T.RUNS_DIR = td
            try:
                ckpt_path = os.path.join(td, "fake.pt")
                torch.save(ck, ckpt_path)
                demo = W.op_run_demo({"framework": "custom", "custom_recipe": "wp",
                                      "ckpt_path": ckpt_path, "strength": 1.0,
                                      "seed": 0, "layer": "none", "preset": "none"})
            finally:
                T.RUNS_DIR = old
        assert demo["ok"], demo.get("error")
        assert demo["trained"] is True
        # 手工用同一份权重复算 BER, 必须与接口返回完全一致
        cover = W._demo_cover(128)
        bits = np.random.default_rng(0).integers(0, 2, size=30).astype(np.uint8)
        e2 = NetSpec(ck["spec_enc"]).build().as_module()
        e2.load_state_dict(ck["enc"]); e2.eval()
        d2 = NetSpec(ck["spec_dec"]).build().as_module()
        d2.load_state_dict(ck["dec"]); d2.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(
                cover.transpose(2, 0, 1)))[None].float()
            m = torch.from_numpy(bits.astype(np.float32))[None]
            y = e2({"image": x, "message": m})[0].permute(1, 2, 0).numpy()
            emb = np.clip(y, 0, 1)
            logit = d2({"image": torch.from_numpy(np.ascontiguousarray(
                emb.transpose(2, 0, 1)))[None].float()})[0]
            dbits = (logit > 0.5).numpy().astype(np.uint8)
        ber = round(int(np.sum(dbits != bits)) / 30, 4)
        assert abs(ber - demo["nodes"]["metrics"]["ber"]) < 1e-9, \
            (ber, demo["nodes"]["metrics"]["ber"], "权重没真正载入")
        print(f"[ok] ckpt 回放用真权重: BER 与手工复算一致 ({ber})")


def test_ckpt_structure_mismatch_errors():
    """ckpt 与所选结构不符时明确报错, 不静默用随机权重。"""
    import torch
    import webui as W
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        crec.save("wp2", enc.spec, dec.spec)
        enc_v, dec_v, _ = RECIPES["videoseal"]()
        ck = {"enc": enc.build().as_module().state_dict(),
              "dec": dec.build().as_module().state_dict(),
              "spec_enc": enc.spec, "spec_dec": dec.spec,
              "size": 128, "msg_len": 30, "recipe": "custom:wp2"}
        with tempfile.TemporaryDirectory() as td:
            old = T.RUNS_DIR
            T.RUNS_DIR = td
            try:
                p = os.path.join(td, "m.pt")
                torch.save(ck, p)
                r = W.op_run_demo({"framework": "custom",
                                   "enc_spec": enc_v.spec, "dec_spec": dec_v.spec,
                                   "ckpt_path": p, "strength": 1.0, "seed": 0,
                                   "layer": "none", "preset": "none"})
            finally:
                T.RUNS_DIR = old
        assert not r["ok"], "结构不匹配应报错"
        assert "不匹配" in r["error"], r["error"]
        print("[ok] ckpt 与结构不符 -> 明确报错, 不静默随机权重")


def test_ckpt_outside_runs_rejected():
    import torch
    import webui as W
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        crec.save("wp3", enc.spec, dec.spec)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "evil.pt")
            torch.save({"enc": {}, "dec": {}, "spec_enc": enc.spec,
                        "spec_dec": dec.spec}, p)
            r = W.op_run_demo({"framework": "custom", "custom_recipe": "wp3",
                               "ckpt_path": p, "strength": 1.0, "seed": 0,
                               "layer": "none", "preset": "none"})
        assert not r["ok"] and "runs/" in r["error"], r
        print("[ok] runs/ 之外的 ckpt 被拒绝")


if __name__ == "__main__":
    test_train_custom_recipe()
    test_ckpt_weights_actually_applied()
    test_ckpt_structure_mismatch_errors()
    test_ckpt_outside_runs_rejected()
    print("\nALL TRAIN-CUSTOM TESTS PASSED")
