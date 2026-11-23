"""积木系统测试: NetSpec 搭建/自动适配/换积木/LLM prompt/补丁闭环。
python tests/test_netspec.py 即可跑 (需 torch)。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmflow.modules import MODULES
from wmflow.netspec import NetSpec, NetSpecError
from wmflow.recipes import RECIPES, describe_framework
from wmflow import llm


def test_module_catalog():
    assert len(MODULES) >= 12, f"积木数 {len(MODULES)}"
    for name in ("conv_block", "res_block", "msg_map", "bit_head",
                 "residual_add", "adapter1x1"):
        assert name in MODULES
        d = MODULES[name].describe()
        assert "层组成" in d  # 显示模块内部层构成, 不外露单层
    print(f"[ok] 积木目录: {len(MODULES)} 个模块")


def test_recipes_build_and_forward():
    import torch
    for fw in ("hidden", "stegastamp", "mbrs"):
        enc, dec, meta = RECIPES[fw]()
        eb, db = enc.build(), dec.build()
        em, dm = eb.as_module(), db.as_module()
        msg_len = enc.inputs["message"][0]
        x = torch.rand(2, 3, 128, 128)
        m = (torch.rand(2, msg_len) > 0.5).float()
        with torch.no_grad():
            y = em({"image": x, "message": m})
            logit = dm({"image": y})
        assert y.shape == (2, 3, 128, 128), f"{fw} encoder out {y.shape}"
        assert logit.shape == (2, msg_len), f"{fw} decoder out {logit.shape}"
        assert 0.0 <= float(y.min()) and float(y.max()) <= 1.0
        print(f"[ok] {fw}: enc->dec forward 128x128, msg_len={msg_len}, "
              f"modules {len(eb.modules)}/{len(db.modules)}")


def test_auto_adapt():
    """显式通道与上游不符 → 自动插 adapter1x1。"""
    spec = NetSpec({
        "name": "adapt_test",
        "input": {"image": [3, 32, 32]},
        "modules": [
            {"id": "a", "type": "conv_block", "params": {"out_ch": 16}},
            {"id": "b", "type": "conv_block", "params": {"in_ch": 24, "out_ch": 8}},
        ],
        "output": "b"})
    built = spec.build()
    ids = [bm.id for bm in built.modules]
    assert any(i.endswith("__adapt0") for i in ids), f"未自动插适配: {ids}"
    assert built.auto_adapters, "auto_adapters 未记录"
    import torch
    fn = built.as_module()
    with torch.no_grad():
        y = fn({"image": torch.rand(1, 3, 32, 32)})
    assert y.shape == (1, 8, 32, 32)
    print(f"[ok] 自动适配: 插入 {built.auto_adapters} 后形状 {tuple(y.shape)}")


def test_swap_module():
    """换积木: conv_block→se_block 同样能跑 (通道自动继承)。"""
    base = {
        "name": "swap", "input": {"image": [3, 32, 32]},
        "modules": [
            {"id": "a", "type": "conv_block", "params": {"out_ch": 16}},
            {"id": "b", "type": "res_block"},
            {"id": "c", "type": "bit_head", "params": {"msg_len": 10}},
        ], "output": "c"}
    s1 = NetSpec(base)
    new = s1.apply_patch([{"id": "b", "replace_type": "se_block"}])
    import torch
    with torch.no_grad():
        y = new.build().as_module()({"image": torch.rand(1, 3, 32, 32)})
    assert y.shape == (1, 10)
    print("[ok] 换积木: res_block→se_block 前向通过")


def test_llm_prompt_and_patch_loop():
    """prompt 渲染 + 模拟模型回复 → 解析补丁 → 自动重适配套用。"""
    enc, dec, meta = RECIPES["hidden"]()
    p = enc.render_prompt()
    for sec in ("模块目录", "积木清单", "形状与自动适配解析表", "约定"):
        assert sec in p, f"prompt 缺段: {sec}"
    assert "conv_block" in p and "层组成" in p

    fw_desc = describe_framework("stegastamp")
    assert "StegaStamp" in fw_desc and "encoder" in fw_desc

    fake_reply = ('分析如下...\n```json\n{"analysis": "加深残差", "patch": '
                  '[{"id": "r4", "replace_type": "se_block"}], "risk": "低"}\n```')
    new_spec = llm.apply_response(enc, fake_reply)
    assert new_spec._by_id["r4"]["type"] == "se_block"
    # 回显同源回归: to_json 的 spec 必须反映补丁 (曾因 modules/spec 双源丢失)
    assert json.loads(new_spec.to_json())["modules"][7]["type"] == "se_block"
    import torch
    with torch.no_grad():
        y = new_spec.build().as_module()(
            {"image": torch.rand(1, 3, 128, 128),
             "message": (torch.rand(1, 30) > 0.5).float()})
    assert y.shape == (1, 3, 128, 128)
    tp = llm.tune_prompt(enc, {"ber": 0.2, "psnr": 28}, "ber<0.05")
    assert "输出格式" in tp and "patch" in tp
    print("[ok] LLM prompt 渲染 + 补丁闭环 (替换 r4→se_block 自动重适配)")


def test_debug_prompt():
    enc, _, _ = RECIPES["mbrs"]()
    p = llm.debug_prompt(enc, "训练后 BER=0.5", "RuntimeError: size mismatch")
    assert "RuntimeError" in p and "问题现象" in p
    print("[ok] debug prompt 含现象与 traceback")


if __name__ == "__main__":
    test_module_catalog()
    test_recipes_build_and_forward()
    test_auto_adapt()
    test_swap_module()
    test_llm_prompt_and_patch_loop()
    test_debug_prompt()
    print("\nALL NETSPEC TESTS PASSED")
