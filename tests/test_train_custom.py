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


def _save_ckpt(enc_spec, dec_spec, path, recipe="custom:x", size=128, msg_len=30):
    """用当前随机权重存一个 ckpt (结构快照 + 权重)。"""
    import torch
    ck = {"enc": enc_spec.build().as_module().state_dict(),
          "dec": dec_spec.build().as_module().state_dict(),
          "spec_enc": enc_spec.spec, "spec_dec": dec_spec.spec,
          "size": size, "msg_len": msg_len, "recipe": recipe}
    torch.save(ck, path)
    return ck


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
                            map_location="cpu", weights_only=True)
            assert ck["size"] == 128, f"配方尺寸应覆盖表单 32, got {ck['size']}"
            assert ck["msg_len"] == 30
            assert ck["spec_enc"] and ck["spec_dec"], "ckpt 必须带结构快照"
            assert ck["recipe"] == "custom:trainme", ck.get("recipe")
            assert any("梯度自检通过" in x for x in log), log[-3:]
            assert any("已忽略表单尺寸" in x for x in log), "应提示尺寸被覆盖"
            print("[ok] train custom:<name>: 1 epoch, size 32->128, ckpt 带 spec, 梯度自检过")


def test_strength_zero_rejected_by_grad_check():
    """strength=0 时 encoder 输出被丢弃 -> 梯度全 0, 自检必须拦下 (Codex 指出的漏报)。"""
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        crec.save("sz", enc.spec, dec.spec)
        with tempfile.TemporaryDirectory() as td:
            ds = os.path.join(td, "ds")
            os.makedirs(ds)
            _make_dataset(ds, 2, 64)
            old = T.RUNS_DIR
            T.RUNS_DIR = td
            try:
                T.STATE.__init__()
                T.STATE.running = True
                T.STATE.started_at = T.time.time()
                try:
                    T.train({"framework": "custom:sz", "dataset_path": ds,
                             "n_images": 2, "epochs": 1, "batch_size": 2,
                             "strength": 0.0, "size": 128, "attacks": ["none"]})
                    raise AssertionError("strength=0 应被梯度自检拦下")
                except ValueError as e:
                    assert "梯度全为 0" in str(e), e
            finally:
                T.RUNS_DIR = old
            print("[ok] strength=0 -> 梯度自检拦下 (encoder 全零梯度)")


def test_model_weights_actually_applied():
    """比嵌入图浮点值 (不是比 BER): 回放必须与手工同权重逐元素一致。"""
    import torch
    import webui as W
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        crec.save("wp", enc.spec, dec.spec)
        with tempfile.TemporaryDirectory() as td:
            old = T.RUNS_DIR
            T.RUNS_DIR = td
            try:
                p = os.path.join(td, "fake.pt")
                ck = _save_ckpt(enc, dec, p, recipe="custom:wp")
                m = W._resolve_demo_model({"framework": "custom",
                                           "custom_recipe": "wp", "ckpt_path": p})
            finally:
                T.RUNS_DIR = old
        assert m["status"] == "trained", m["status"]
        cover = W._demo_cover(128)
        bits = np.random.default_rng(3).integers(0, 2, size=30).astype(np.uint8)
        got = m["encode"](cover, bits, 1.0)
        # 手工用同一份权重算
        e2 = NetSpec(ck["spec_enc"]).build().as_module()
        e2.load_state_dict(ck["enc"]); e2.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(
                cover.transpose(2, 0, 1)))[None].float()
            mm = torch.from_numpy(bits.astype(np.float32))[None]
            y = e2({"image": x, "message": mm})[0].permute(1, 2, 0).numpy()
            want = np.clip(y, 0, 1)
        assert np.allclose(got, want, atol=0, rtol=0), "嵌入图与手工同权重不一致: 权重没真载入"
        # 解码阈值也必须是 logit>0
        d2 = NetSpec(ck["spec_dec"]).build().as_module()
        d2.load_state_dict(ck["dec"]); d2.eval()
        with torch.no_grad():
            logit = d2({"image": torch.from_numpy(np.ascontiguousarray(
                got.transpose(2, 0, 1)))[None].float()})[0].numpy()
        assert np.array_equal(m["decode"](got), (logit > 0).astype(np.uint8)), \
            "解码阈值不是 logit>0"
        print("[ok] 权重真载入: 嵌入图逐元素一致; 解码用 logit>0")


def test_deleted_recipe_still_replays():
    """配方删除后, ckpt 仍按结构快照回放 (不能因查配方失败而报错)。"""
    import webui as W
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        crec.save("gone", enc.spec, dec.spec)
        with tempfile.TemporaryDirectory() as td:
            old = T.RUNS_DIR
            T.RUNS_DIR = td
            try:
                p = os.path.join(td, "g.pt")
                _save_ckpt(enc, dec, p, recipe="custom:gone")
                crec.delete("gone")
                m = W._resolve_demo_model({"framework": "custom:gone", "ckpt_path": p})
            finally:
                T.RUNS_DIR = old
        assert m["status"] == "trained" and m["msg_len"] == 30
        assert "已不存在" in (m["note"] or ""), m["note"]
        print("[ok] 配方删除后 ckpt 仍按快照回放, 并提示配方已不存在")


def test_overwritten_recipe_warns():
    """同名配方被改成别的结构后, 仍按 ckpt 旧结构跑, 但要给出提示。"""
    import webui as W
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        crec.save("ov", enc.spec, dec.spec)
        with tempfile.TemporaryDirectory() as td:
            old = T.RUNS_DIR
            T.RUNS_DIR = td
            try:
                p = os.path.join(td, "o.pt")
                _save_ckpt(enc, dec, p, recipe="custom:ov")
                enc_v, dec_v, _ = RECIPES["videoseal"]()
                crec.save("ov", enc_v.spec, dec_v.spec, overwrite=True)
                m = W._resolve_demo_model({"framework": "custom:ov", "ckpt_path": p})
            finally:
                T.RUNS_DIR = old
        assert m["status"] == "trained" and m["msg_len"] == 30, "应按 ckpt 的 30bit 结构"
        assert "结构已被改动" in (m["note"] or ""), m["note"]
        print("[ok] 配方被覆盖成别的结构 -> 仍按 ckpt 快照跑 + 明确提示")


def test_ckpt_structure_mismatch_errors():
    """显式混搭 spec 与 ckpt 结构不符时明确报错, 不静默用随机权重。"""
    import webui as W
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        enc_v, dec_v, _ = RECIPES["videoseal"]()
        with tempfile.TemporaryDirectory() as td:
            old = T.RUNS_DIR
            T.RUNS_DIR = td
            try:
                p = os.path.join(td, "m.pt")
                _save_ckpt(enc, dec, p, recipe="custom:m")
                try:
                    W._resolve_demo_model({"framework": "custom",
                                           "enc_spec": enc_v.spec,
                                           "dec_spec": dec_v.spec, "ckpt_path": p})
                    raise AssertionError("结构不匹配应报错")
                except ValueError as e:
                    assert "不匹配" in str(e), e
            finally:
                T.RUNS_DIR = old
        print("[ok] 显式混搭与 ckpt 结构不符 -> 明确报错")


def test_ckpt_outside_runs_rejected():
    import webui as W
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        crec.save("wp3", enc.spec, dec.spec)
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "evil.pt")
            _save_ckpt(enc, dec, p)
            try:
                W._resolve_demo_model({"framework": "custom", "custom_recipe": "wp3",
                                       "ckpt_path": p})
                raise AssertionError("runs/ 外的 ckpt 应被拒绝")
            except ValueError as e:
                assert "runs/" in str(e), e
        print("[ok] runs/ 之外的 ckpt 被拒绝")


def test_lsb_not_labelled_trained():
    """lsb_toy 是确定性算法嵌入, 标 algorithmic, 不能冒充 trained。"""
    import webui as W
    r = W.op_run_demo({"framework": "lsb_toy", "msg_len": 64, "strength": 1.0,
                       "seed": 0, "layer": "none", "preset": "none"})
    assert r["ok"] and r["model_status"] == "algorithmic", r.get("model_status")
    assert r["trained"] is False, "lsb_toy 不应标 trained"
    print("[ok] lsb_toy 标 algorithmic (不冒充训练权重)")


if __name__ == "__main__":
    test_train_custom_recipe()
    test_strength_zero_rejected_by_grad_check()
    test_model_weights_actually_applied()
    test_deleted_recipe_still_replays()
    test_overwritten_recipe_warns()
    test_ckpt_structure_mismatch_errors()
    test_ckpt_outside_runs_rejected()
    test_lsb_not_labelled_trained()
    print("\nALL TRAIN-CUSTOM TESTS PASSED")
