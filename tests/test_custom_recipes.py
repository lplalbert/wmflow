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


def test_size_mismatch_rejected():
    """encoder 输出尺寸与 decoder 声明尺寸不一致必须拒绝 (Codex 评审阻断项 1)。"""
    with temp_store():
        enc, _, _ = RECIPES["hidden"]()                 # 128x128
        _, dec64, _ = RECIPES["videoseal"](size=64)     # 声明收 64x64
        try:
            crec.save("sizemix", enc.spec, dec64.spec)
            raise AssertionError("尺寸错配应被拒绝")
        except ValueError as e:
            assert "尺寸不一致" in str(e), e
        assert crec.list_all() == [], "拒绝后不应留下任何配方文件"
        print("[ok] size mismatch rejected: 128x128 enc + 64x64 dec -> refused, no file")


def test_enc_in_out_size_mismatch():
    """encoder 输入 128 但输出 32 的怪结构也要拒 (流水线要把输出加回同尺寸载体)。"""
    with temp_store():
        weird = {"name": "weird",
                 "input": {"image": [3, 128, 128], "message": [30]},
                 "modules": [
                     {"id": "mm", "type": "msg_map", "params": {"out_ch": 1},
                      "from": ["message"]},
                     {"id": "cat", "type": "concat", "from": ["image", "mm"]},
                     {"id": "d1", "type": "conv_block",
                      "params": {"out_ch": 32, "stride": 2}},
                     {"id": "d2", "type": "conv_block",
                      "params": {"out_ch": 3, "stride": 2}},
                 ], "output": "d2"}
        dec32 = {"name": "dec32", "input": {"image": [3, 32, 32]},
                 "modules": [
                     {"id": "d1", "type": "conv_block", "params": {"out_ch": 32}},
                     {"id": "hd", "type": "bit_head", "params": {"msg_len": 30}},
                 ], "output": "hd"}
        try:
            crec.save("weird", weird, dec32)
            raise AssertionError("encoder 输入/输出尺寸不一致应被拒绝")
        except ValueError as e:
            assert "与输入" in str(e), e
        print("[ok] enc in/out size mismatch rejected (128 in -> 32 out)")


def test_meta_cannot_override_derived():
    """meta 不许覆盖已校验的 msg_len/size/out_shape。"""
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        s = crec.save("metaover", enc.spec, dec.spec,
                      meta={"msg_len": 999, "size": 32, "note": "keep",
                            "msg_len_aligned": True})
        assert s["msg_len"] == 30 and s["size"] == 128, s
        rec = crec.get("metaover")
        assert rec["meta"]["msg_len"] == 30 and rec["meta"]["size"] == 128
        assert rec["meta"]["note"] == "keep"
        assert "msg_len_aligned" not in rec["meta"], "未真对齐时不许伪造对齐标记"
        print("[ok] meta cannot override derived: msg_len/size kept, aligned flag not forged")


def test_concurrent_same_name():
    """同名并发 save: 非覆盖写恰好一个成功, 不留临时文件。"""
    import threading
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        results, barrier = [], threading.Barrier(6)

        def worker():
            barrier.wait()
            try:
                crec.save("race", enc.spec, dec.spec)   # overwrite=False
                results.append("ok")
            except FileExistsError:
                results.append("exists")
            except Exception as e:                      # 任何其他异常都算失败
                results.append(f"err:{type(e).__name__}")

        ts = [threading.Thread(target=worker) for _ in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert results.count("ok") == 1, results
        assert not any(r.startswith("err:") for r in results), results
        rec = crec.get("race")                          # 文件必须完整可读
        assert NetSpec(rec["encoder"]).build().out_shape == (3, 128, 128)
        leftover = [p.name for p in Path(crec.CUSTOM_DIR).iterdir()
                    if p.name.endswith(".tmp")]
        assert leftover == [], f"残留临时文件: {leftover}"
        print(f"[ok] concurrent same-name: {results.count('ok')} ok / "
              f"{results.count('exists')} exists, no .tmp left")


def test_reserved_and_symlink():
    with temp_store():
        enc, dec, _ = RECIPES["hidden"]()
        for bad in ("CON", "NUL", "com1", "PRN"):
            try:
                crec.save(bad, enc.spec, dec.spec)
                raise AssertionError(f"{bad} 应被拒绝")
            except ValueError as e:
                assert "保留" in str(e), e
        crec.save("real", enc.spec, dec.spec)
        # symlink 逃逸 (Windows 无权限时跳过)
        outside = Path(crec.CUSTOM_DIR).parent / "outside.json"
        outside.write_text('{"encoder":{},"decoder":{}}', encoding="utf-8")
        link = Path(crec.CUSTOM_DIR) / "link.json"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            print("[ok] reserved names rejected (symlink case skipped: no privilege)")
            return
        try:
            crec.get("link")
            raise AssertionError("symlink 应被拒绝")
        except ValueError as e:
            assert "符号链接" in str(e), e
        print("[ok] reserved names + symlink escape rejected")


def test_source_field_ambiguity():
    """_recipe_side 同时给 spec/framework/name 必须拒绝 (不再静默忽略)。"""
    import webui as W
    with temp_store():
        r = W.op_recipe_save({
            "name": "amb", "encoder": {"kind": "framework", "framework": "hidden",
                                       "spec": {"x": 1}},
            "decoder": {"kind": "framework", "framework": "hidden"}})
        assert not r["ok"] and "只能选一种来源" in r["error"], r
        print("[ok] ambiguous source fields rejected")


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
        # 单层 jpeg 曾因负 stride 视图喂 torch 而崩, 这里连同 combo 一起守着
        for layer in ("jpeg", "blur", "combo_builtin"):
            dl = W.op_run_demo({"framework": "custom", "custom_recipe": "ui_mix",
                                "strength": 1.0, "seed": 2,
                                "layer": layer, "preset": "none"})
            assert dl["ok"], (layer, dl.get("error"))
        # 两种调用风格都应生效: custom_recipe=<名> 与 framework='custom:<名>'
        for payload in ({"framework": "custom", "custom_recipe": "ui_mix"},
                        {"framework": "custom:ui_mix"}):
            dl2 = W.op_run_demo({**payload, "strength": 1.0, "seed": 3,
                                 "layer": "none", "preset": "none"})
            assert dl2["ok"] and dl2["msg_len"] == 30, (payload, dl2.get("error"))
        dele = W.op_recipe_delete({"name": "ui_mix"})
        assert dele["ok"]
        print("[ok] webui ops: save/list/get/export/import/delete + run_demo(custom_recipe) "
              "BER=%.3f" % demo["nodes"]["metrics"]["ber"])


if __name__ == "__main__":
    test_save_assemble_and_load()
    test_msg_len_auto_align()
    test_export_import_roundtrip()
    test_validation_rejects_bad()
    test_size_mismatch_rejected()
    test_enc_in_out_size_mismatch()
    test_meta_cannot_override_derived()
    test_concurrent_same_name()
    test_reserved_and_symlink()
    test_source_field_ambiguity()
    test_name_safety_and_delete()
    test_webui_endpoints()
    print("\nALL CUSTOM RECIPE TESTS PASSED")
