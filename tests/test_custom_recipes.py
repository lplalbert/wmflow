"""自定义配方库测试: 存/列/取/导出/导入/删 + 消息长自动对齐 + 校验拦截 + 流水线直用。

python tests/test_custom_recipes.py 即可跑 (印刷文案全 ASCII, 防 Windows GBK 控制台炸)。
"""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmflow import custom_recipes as crec
from wmflow.components import assemble, segment_spec
from wmflow.netspec import NetSpec
from wmflow.recipes import RECIPES


class temp_store:
    """把配方目录临时指到 tempdir, 不污染仓库 custom_recipes/。"""

    def __enter__(self):
        self._td = tempfile.TemporaryDirectory()
        self._old = crec.CUSTOM_DIR
        crec.CUSTOM_DIR = Path(self._td.name)
        return crec.CUSTOM_DIR

    def __exit__(self, *exc):
        crec.CUSTOM_DIR = self._old
        self._td.cleanup()


def _mixed_encoder():
    """跨框架混段 encoder: hidden.fusion + videoseal.res + hidden.out (同 test_components)。"""
    enc_h, _, _ = RECIPES["hidden"]()
    enc_v, _, _ = RECIPES["videoseal"]()
    segs_h, segs_v = segment_spec(enc_h.spec), segment_spec(enc_v.spec)
    fusion = next(s for s in segs_h if s["family"] == "fusion")
    res_v = next(s for s in segs_v if s["family"] == "res")
    out_h = next(s for s in segs_h if s["family"] == "out")
    return assemble([fusion, res_v, out_h], "encoder", 30, 128)


def test_save_assemble_and_load():
    with temp_store():
        enc_spec = _mixed_encoder()
        _, dec, _ = RECIPES["hidden"]()
        s = crec.save("my_mix", enc_spec, dec.spec,
                      title="hidden.fusion + videoseal.res",
                      note="cross-framework mix", source={"kind": "assemble"})
        assert s["name"] == "my_mix" and s["msg_len"] == 30, s
        assert s["out_shape"] == [3, 128, 128], s
        listing = crec.list_all()
        assert len(listing) == 1 and listing[0]["name"] == "my_mix", listing
        rec = crec.get("my_mix")
        assert NetSpec(rec["encoder"]).build().out_shape == (3, 128, 128)
        assert NetSpec(rec["decoder"]).build().out_shape == (30,)
        enc2, dec2, meta = crec.pair("my_mix")
        assert meta["framework"] == "custom:my_mix" and meta["msg_len"] == 30
        assert enc2.build().out_shape == (3, 128, 128)
        print("[ok] save/list/get/pair: assembled enc + built-in dec -> custom:my_mix")


def test_msg_len_auto_align():
    """hidden(30bit) enc + videoseal(256bit) dec: 存盘时自动把 bit_head 对齐到 30。"""
    with temp_store():
        enc, _, _ = RECIPES["hidden"]()
        _, dec_v, _ = RECIPES["videoseal"]()
        assert dec_v.build().out_shape == (256,)
        s = crec.save("mix_align", enc.spec, dec_v.spec)
        assert s["msg_len"] == 30, s
        rec = crec.get("mix_align")
        assert rec["meta"].get("msg_len_aligned") is True, rec["meta"]
        heads = [m for m in rec["decoder"]["modules"] if m["type"] == "bit_head"]
        assert heads and heads[-1]["params"]["msg_len"] == 30, heads
        assert NetSpec(rec["decoder"]).build().out_shape == (30,)
        print("[ok] msg_len align: videoseal dec 256bit -> 30bit (flag msg_len_aligned)")


def test_export_import_roundtrip():
    with temp_store():
        enc, dec, _ = RECIPES["mbrs"]()
        crec.save("orig", enc.spec, dec.spec)
        fname, text = crec.export("orig")
        assert fname == "orig.json" and '"encoder"' in text
        s = crec.import_json(text, name="copy1")
        assert s["name"] == "copy1" and s["msg_len"] == 30
        assert len(crec.list_all()) == 2
        # 同名导入: 默认拒绝, 勾 overwrite 才写
        try:
            crec.import_json(text, name="copy1")
            raise AssertionError("same-name import should raise")
        except FileExistsError:
            pass
        crec.import_json(text, name="copy1", overwrite=True)
        # 裸 {encoder,decoder} 也接受
        bare = '{"encoder": %s, "decoder": %s}' % (
            json.dumps(enc.spec), json.dumps(dec.spec))
        s3 = crec.import_json(bare, name="bare")
        assert s3["msg_len"] == 30
        print("[ok] export/import: roundtrip + overwrite guard + bare pair")


def test_validation_rejects_bad():
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        # decoder 没有 bit_head: 输出不是 (30,) -> 必须拒绝
        bad = {"name": "nobit", "input": {"image": [3, 128, 128]},
               "modules": [dict(m) for m in dec.spec["modules"]
                           if m["type"] != "bit_head"],
               "output": None}
        bad["output"] = bad["modules"][-1]["id"]
        try:
            crec.save("nobit", enc.spec, bad)
            raise AssertionError("decoder without bit_head should be rejected")
        except ValueError as e:
            assert "decoder" in str(e), e
        # encoder 缺 message 输入
        enc_nomsg = {"name": "nomsg", "input": {"image": [3, 128, 128]},
                     "modules": enc.spec["modules"], "output": enc.spec["output"]}
        try:
            crec.save("nomsg", enc_nomsg, dec.spec)
            raise AssertionError("encoder without message should be rejected")
        except ValueError as e:
            assert "message" in str(e) or "构建失败" in str(e), e
        # 空名
        try:
            crec.save("  ", enc.spec, dec.spec)
            raise AssertionError("empty name should be rejected")
        except ValueError:
            pass
        print("[ok] validation: reject decoder-without-head / encoder-without-msg / empty name")


def test_name_safety_and_delete():
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        s = crec.save("../evil", enc.spec, dec.spec)
        p = Path(s["path"]).resolve()
        assert p.parent == Path(crec.CUSTOM_DIR).resolve(), (p, crec.CUSTOM_DIR)
        assert p.name == "evil.json", p
        s2 = crec.save("中文 名", enc.spec, dec.spec)
        assert s2["name"] == "中文_名", s2["name"]
        # 覆盖语义: 默认拒绝, overwrite=True 保留 created
        try:
            crec.save("中文 名", enc.spec, dec.spec)
            raise AssertionError("duplicate save should raise")
        except FileExistsError:
            pass
        before = crec.get("中文_名")["created"]
        crec.save("中文 名", enc.spec, dec.spec, overwrite=True, note="v2")
        after = crec.get("中文_名")
        assert after["created"] == before and after["meta"]["note"] == "v2"
        crec.delete("中文_名")
        assert not crec.exists("中文_名")
        try:
            crec.delete("中文_名")
            raise AssertionError("delete missing should raise")
        except FileNotFoundError:
            pass
        print("[ok] name safety: ../evil -> evil.json, chinese kept, overwrite keeps created")


def test_webui_endpoints():
    """走 webui 的 op 层: 存(内置两侧) -> 列表 -> 导出 -> 流水线直跑自定义配方。"""
    import webui as W
    with temp_store():
        r = W.op_recipe_save({
            "name": "ui_mix", "title": "ui test",
            "encoder": {"kind": "framework", "framework": "hidden"},
            "decoder": {"kind": "framework", "framework": "videoseal"},
            "source": {"kind": "assemble", "segments": ["fusion", "res", "out"]}})
        assert r["ok"], r
        assert r["recipe"]["msg_len"] == 30, r
        lst = W.op_recipe_list({})
        assert lst["ok"] and any(x["name"] == "ui_mix" for x in lst["recipes"])
        got = W.op_recipe_get({"name": "ui_mix"})
        assert got["ok"] and got["segments"]["encoder"], got.get("error")
        exp = W.op_recipe_export({"name": "ui_mix"})
        assert exp["ok"] and exp["filename"] == "ui_mix.json"
        imp = W.op_recipe_import({"json": exp["json"], "name": "ui_copy"})
        assert imp["ok"], imp
        demo = W.op_run_demo({"framework": "custom", "custom_recipe": "ui_mix",
                              "strength": 1.0, "seed": 1,
                              "layer": "none", "preset": "none"})
        assert demo["ok"], demo.get("error")
        assert demo["msg_len"] == 30
        assert 0.0 <= demo["nodes"]["metrics"]["ber"] <= 1.0
        dele = W.op_recipe_delete({"name": "ui_mix"})
        assert dele["ok"]
        print("[ok] webui ops: save/list/get/export/import/delete + run_demo(custom_recipe) "
              "BER=%.3f" % demo["nodes"]["metrics"]["ber"])


if __name__ == "__main__":
    test_save_assemble_and_load()
    test_msg_len_auto_align()
    test_export_import_roundtrip()
    test_validation_rejects_bad()
    test_name_safety_and_delete()
    test_webui_endpoints()
    print("\nALL CUSTOM RECIPE TESTS PASSED")
