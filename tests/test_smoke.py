"""wmflow 冒烟测试: 引擎机制 + 端到端 LSB 干净往返。python tests/test_smoke.py 即可跑。"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmflow import Executor, save_report, all_node_info
from wmflow.registry import get_node


def make_cover(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.5 + 0.25 * np.sin(xx / 3.0) * np.cos(yy / 5.0)
    img = np.stack([base] * 3, axis=-1) + rng.normal(0, 0.02, (h, w, 3))
    return np.clip(img, 0, 1).astype(np.float32)


def test_registry():
    info = all_node_info()
    assert len(info) == 14, f"预期 14 节点, got {len(info)}"
    assert "Embed" in info and "BERCompare" in info
    assert "NetSpecModel" in info and "NetSpecPrompt" in info
    print(f"[ok] registry: {len(info)} nodes")


def test_clean_roundtrip():
    """干净往返 BER 必须为 0 (LSB 玩具模型, 无攻击)。"""
    with tempfile.TemporaryDirectory() as td:
        import cv2
        cover = make_cover()
        cv2.imwrite(f"{td}/cover.png", (cover[:, :, ::-1] * 255).astype(np.uint8))
        wf = {
            "meta": {"name": "clean"},
            "nodes": {
                "1": {"class_type": "CoverLoader", "inputs": {"path": f"{td}/cover.png"}},
                "2": {"class_type": "MessageGen", "inputs": {"length": 64, "seed": 1}},
                "3": {"class_type": "WMModelLoader", "inputs": {"framework": "lsb_toy", "msg_len": 64}},
                "4": {"class_type": "Embed", "inputs": {"model": ["3", 0], "cover": ["1", 0], "message": ["2", 0], "strength": 1.0}},
                "5": {"class_type": "Decode", "inputs": {"model": ["3", 0], "image": ["4", 0]}},
                "6": {"class_type": "BERCompare", "inputs": {"ref": ["2", 0], "dec": ["5", 0]}},
            },
            "outputs": ["6"],
        }
        assert Executor.validate(wf) == [], Executor.validate(wf)
        res = Executor().run(wf)
        ber = res.metrics["6:BERCompare"]["ber"]
        assert ber == 0.0, f"干净往返 BER={ber}, 预期 0"
        print(f"[ok] clean roundtrip: BER={ber}")


def test_cache_and_seed():
    """同图重跑应全缓存命中; 改 seed 应失效。"""
    with tempfile.TemporaryDirectory() as td:
        import cv2
        cover = make_cover()
        cv2.imwrite(f"{td}/cover.png", (cover[:, :, ::-1] * 255).astype(np.uint8))
        wf = {
            "meta": {"name": "cache"},
            "nodes": {
                "1": {"class_type": "CoverLoader", "inputs": {"path": f"{td}/cover.png"}},
                "2": {"class_type": "MessageGen", "inputs": {"length": 64, "seed": 1}},
                "3": {"class_type": "NoiseLayer", "inputs": {"image": ["1", 0], "layer": "blur", "seed": 5}},
            },
            "outputs": ["3"],
        }
        cache = {}
        e = Executor(cache=cache)
        e.run(wf)
        res2 = Executor(cache=cache).run(wf)
        assert all(r.from_cache for r in res2.runs), "重跑未全命中缓存"
        wf["nodes"]["2"]["inputs"]["seed"] = 2
        res3 = Executor(cache=cache).run(wf)
        assert not all(r.from_cache for r in res3.runs), "换 seed 未失效缓存"
        print(f"[ok] cache: 重跑 {res2.cache_hits}/{len(res2.runs)} 命中, 换seed后 {res3.cache_hits}/{len(res3.runs)}")


def test_type_check():
    """类型错线必须被校验拒绝。"""
    wf = {
        "meta": {},
        "nodes": {
            "2": {"class_type": "MessageGen", "inputs": {"length": 64, "seed": 1}},
            "4": {"class_type": "Embed", "inputs": {
                "model": ["2", 0], "cover": ["2", 0], "message": ["2", 0]}},
        },
    }
    errs = Executor.validate(wf)
    assert any("类型不匹配" in e for e in errs), f"类型错误未拦截: {errs}"
    print(f"[ok] type check: 拦截 {len(errs)} 处")


def test_full_pipeline():
    """smoke_ablation 全图: 嵌入→攻击→提取→度量→报告落盘。"""
    with tempfile.TemporaryDirectory() as td:
        import cv2
        cover = make_cover(128, 128, seed=3)
        cv2.imwrite(f"{td}/cover.png", (cover[:, :, ::-1] * 255).astype(np.uint8))
        wf_path = Path(__file__).resolve().parents[1] / "workflows" / "smoke_ablation.json"
        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        wf["nodes"]["1"]["inputs"]["path"] = f"{td}/cover.png"
        wf["nodes"]["10"]["inputs"]["out_dir"] = f"{td}/out"
        assert Executor.validate(wf) == [], Executor.validate(wf)
        res = Executor().run(wf, only_outputs=wf["outputs"])
        save_report(res, wf, f"{td}/out")
        ber = res.metrics["8:BERCompare"]["ber"]
        assert 0.0 <= ber <= 0.5, f"攻击后 BER={ber} 异常"
        assert (Path(td) / "out" / "report.json").exists()
        print(f"[ok] full pipeline: BER={ber:.3f} (combo_builtin+wechat_lite), 报告落盘")


if __name__ == "__main__":
    test_registry()
    test_clean_roundtrip()
    test_cache_and_seed()
    test_type_check()
    test_full_pipeline()
    print("\nALL SMOKE TESTS PASSED")
