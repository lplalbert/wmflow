"""组件库测试: 切段 / 拣段重组 / 跨框架混段。
python tests/test_components.py 即可跑。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmflow.components import segment_spec, single_module_segments, assemble
from wmflow.netspec import NetSpec
from wmflow.recipes import RECIPES


def test_segment_recipes():
    for fw in ("hidden", "stegastamp", "mbrs", "videoseal"):
        enc, dec, _ = RECIPES[fw]()
        segs_e = segment_spec(enc.spec)
        segs_d = segment_spec(dec.spec)
        assert len(segs_e) >= 3, f"{fw} enc 段数 {len(segs_e)}"
        # 重组回原顺序必须仍是合法网络
        spec = assemble(segs_e, "encoder", enc.inputs["message"][0], 128)
        built = NetSpec(spec).build()
        assert built.out_shape == (3, 128, 128)
        labels = [s["label"] for s in segs_e]
        print(f"[ok] {fw}: enc {len(segs_e)} 段 / dec {len(segs_d)} 段 | "
              f"重组输出 {built.out_shape} | {' + '.join(labels)}")


def test_cross_framework_segments():
    """hidden 消息融合段 + videoseal 残差组 + hidden 残差输出 → 应能构建。"""
    enc_h, _, _ = RECIPES["hidden"]()
    enc_v, _, _ = RECIPES["videoseal"]()
    segs_h = segment_spec(enc_h.spec)
    segs_v = segment_spec(enc_v.spec)
    fusion = next(s for s in segs_h if s["family"] == "fusion")
    res_v = next(s for s in segs_v if s["family"] == "res")
    out_h = next(s for s in segs_h if s["family"] == "out")
    spec = assemble([fusion, res_v, out_h], "encoder", 30, 128)
    built = NetSpec(spec).build()
    assert built.out_shape == (3, 128, 128)
    print(f"[ok] 跨框架拣段: hidden.fusion + videoseal.res + hidden.out → "
          f"{built.out_shape}, 自动适配 {len(built.auto_adapters)} 处")


def test_empty_and_single():
    try:
        assemble([], "encoder")
        raise AssertionError("空组装台应报错")
    except ValueError as e:
        pass
    singles = single_module_segments()
    assert len(singles) >= 12
    spec = assemble([singles[0], singles[1]], "decoder", 30, 64)
    # 允许构建失败 (随机单积木未必成链) 但 assemble 本身应产出结构
    assert spec["modules"] and spec["output"] == spec["modules"][-1]["id"]
    print(f"[ok] 空台报错 / 单积木 {len(singles)} 个可拣")


if __name__ == "__main__":
    test_segment_recipes()
    test_cross_framework_segments()
    test_empty_and_single()
    print("\nALL COMPONENTS TESTS PASSED")
