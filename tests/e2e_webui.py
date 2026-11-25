#!/usr/bin/env python
"""wmflow WebUI 端到端点测 (可选, 需 Playwright)。

覆盖: 流水线(5 框架 / 攻击层 / 转码) · 积木链(组件库/载入/构建/追踪/节点编辑/
存配方/自定义直跑) · LLM Prompt(describe/debug/tune/套补丁/无结构容错) · 训练面板。

用法:
  python tests/e2e_webui.py
  # 自定义 chromium / 依赖库 (特殊环境, 如缺 libatomic 的镜像):
  WMFLOW_CHROME=/path/to/chrome WMFLOW_PW_LIB=/path/to/libdir python tests/e2e_webui.py
  # 验证 token 鉴权 (服务以 --token 启动, 并断言无 token 返回 401):
  WMFLOW_E2E_TOKEN=secret python tests/e2e_webui.py

未装 playwright 时打印 [skip] 并以 0 退出 (不阻塞常规测试)。
服务由脚本自行在随机空闲端口启动并负责关闭。
"""
import glob
import os
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

RESULTS = []
JS_ERRORS = []


def rec(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(("[PASS] " if ok else "[FAIL] ") + name + ((" :: " + detail) if detail else ""))


def step(name, fn):
    try:
        detail = fn()
        rec(name, True, detail or "")
    except Exception as e:
        rec(name, False, f"{type(e).__name__}: {e}")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_server(base, timeout=60, token=""):
    url = base + "/api/modules" + (
        ("?token=" + urllib.parse.quote(token)) if token else "")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _run_tests(base, token=""):
    from playwright.sync_api import sync_playwright

    if token:   # 先验: 开启鉴权后, 无 token 的 API 必须 401
        import urllib.error
        try:
            urllib.request.urlopen(base + "/api/modules", timeout=5).read()
            rec("无 token 请求应 401", False, "未返回 401")
        except urllib.error.HTTPError as e:
            rec("无 token 请求应 401", e.code == 401, f"HTTP {e.code}")
        except Exception as e:
            rec("无 token 请求应 401", False, str(e))

    def wait_pipe(page, timeout=90000):
        page.wait_for_function(
            "() => { const e=document.querySelector('#pipeStatus');"
            " return e && (/完成/.test(e.textContent) || e.className.includes('err')); }",
            timeout=timeout)

    def wait_blocks(page, text, timeout=90000):
        page.wait_for_function(
            "(t) => { const e=document.querySelector('#blockStatus');"
            " return e && e.textContent.includes(t); }",
            arg=text, timeout=timeout)

    def run_pipe(page, fw=None, layer=None, preset=None):
        if fw is not None:
            page.select_option("#fw", fw)
        if layer is not None:
            page.select_option("#layer", layer)
        if preset is not None:
            page.select_option("#preset", preset)
        page.click("#ctrl-pipe button.primary")
        wait_pipe(page)
        st = page.locator("#pipeStatus").inner_text()
        ok = ("完成" in st) and ("err" not in (page.locator("#pipeStatus").get_attribute("class") or ""))
        return ok, st

    chrome = (os.environ.get("WMFLOW_CHROME")
              or (sorted(glob.glob(os.path.expanduser(
                  "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome"))) or [None])[-1])
    launch_kw = dict(headless=True, args=["--no-sandbox"])
    if chrome:
        launch_kw["executable_path"] = chrome

    with sync_playwright() as p:
        browser = p.chromium.launch(**launch_kw)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.on("console", lambda m: JS_ERRORS.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)
        page.on("pageerror", lambda e: JS_ERRORS.append(f"pageerror: {e}"))
        page.on("dialog", lambda d: d.accept())

        page_url = base + ("/?token=" + urllib.parse.quote(token) if token else "/")
        step("页面加载", lambda: (page.goto(page_url),
                                  page.wait_for_load_state("networkidle"),
                                  page.wait_for_timeout(1500),
                                  "loaded")[3])

        # ---------------- 流水线 ----------------
        try:
            wait_pipe(page)
            rec("默认流水线自动运行", "完成" in page.locator("#pipeStatus").inner_text(),
                page.locator("#pipeStatus").inner_text()[:100])
        except Exception as e:
            rec("默认流水线自动运行", False, str(e))

        n = page.locator("#canvas .node").count()
        rec("画布节点渲染", n >= 6, f"nodes={n}")

        for fw in ["lsb_toy", "hidden", "stegastamp", "mbrs", "videoseal"]:
            def _f(fw=fw):
                ok, st = run_pipe(page, fw=fw)
                if not ok:
                    raise AssertionError(st[:120])
                return st[:80]
            step(f"流水线框架 {fw}", _f)

        for layer in ["combo_builtin", "jpeg", "blur", "noise"]:
            def _l(layer=layer):
                ok, st = run_pipe(page, fw="lsb_toy", layer=layer, preset="none")
                if not ok:
                    raise AssertionError(st[:120])
                return st[:70]
            step(f"退化层 {layer}", _l)

        for preset in ["wechat_lite", "chroma420", "jpeg75", "resize_half"]:
            def _p(preset=preset):
                ok, st = run_pipe(page, fw="lsb_toy", layer="none", preset=preset)
                if not ok:
                    raise AssertionError(st[:120])
                return st[:70]
            step(f"转码 {preset}", _p)

        # ---------------- 积木链 ----------------
        def _open_blocks():
            page.click("#view-blocks")
            page.wait_for_function(
                "() => document.querySelectorAll('#palette .seg-card').length > 20",
                timeout=90000)
            return (f"pal-groups={page.locator('#palette .pal-group').count()}, "
                    f"seg-cards={page.locator('#palette .seg-card').count()}")
        step("积木链组件库渲染", _open_blocks)

        def _load_whole():
            page.locator("#palette .pal-group h4", has_text="hidden").first \
                .get_by_text("整段载入 enc").click()
            wait_blocks(page, "构建通过")
            return page.locator("#blockStatus").inner_text()[:100]
        step("整段载入 hidden.encoder + 构建", _load_whole)

        def _build():
            page.click("#ctrl-blocks button.primary")
            wait_blocks(page, "构建通过")
            return page.locator("#blockStatus").inner_text()[:100]
        step("构建校验", _build)

        def _trace():
            page.get_by_text("逐积木追踪").click()
            wait_blocks(page, "逐积木输出已显示")
            return page.locator("#blockStatus").inner_text()[:100]
        step("逐积木追踪", _trace)

        def _edit():
            page.locator("#canvas .node").filter(has_text="msg_map").first.click()
            page.wait_for_selector("#editParams", timeout=20000)
            page.fill("#editParams", '{"out_ch": 1}')
            page.get_by_text("套用 (自动适配)").click()
            page.wait_for_function(
                "() => { const e=document.querySelector('#editStatus');"
                " return e && (e.textContent.includes('已套用') || e.className.includes('err')); }",
                timeout=60000)
            st = page.locator("#editStatus").inner_text()
            if "已套用" not in st:
                raise AssertionError(st[:120])
            return st[:80]
        step("节点编辑套用", _edit)

        def _edit_bad_json():
            js_before = len(JS_ERRORS)
            page.locator("#canvas .node").filter(has_text="msg_map").first.click()
            page.wait_for_selector("#editParams", timeout=20000)
            page.fill("#editParams", "{bad json")
            page.get_by_text("套用 (自动适配)").click()
            page.wait_for_timeout(800)
            st = page.locator("#editStatus").inner_text()
            new_err = JS_ERRORS[js_before:]
            if new_err or "已套用" in st:
                raise AssertionError(f"st={st!r} js={new_err[:2]}")
            return st[:80]
        step("节点编辑非法 JSON 容错", _edit_bad_json)

        def _save_recipe():
            page.locator("#ctrl-blocks button", has_text="存为配方").first.click()
            page.wait_for_selector("#svName", timeout=20000)
            page.fill("#svName", "e2e_auto")
            page.fill("#svTitle", "E2E 自动配方")
            page.click("#dlgOk")
            page.wait_for_function(
                "() => { const e=document.querySelector('#dlgStatus');"
                " return e && (e.textContent.includes('已保存') || e.className.includes('err')); }",
                timeout=90000)
            st = page.locator("#dlgStatus").inner_text()
            if "已保存" not in st:
                raise AssertionError(st[:150])
            page.wait_for_timeout(1200)
            return st[:80]
        step("存为自定义配方", _save_recipe)

        def _recipe_card():
            page.wait_for_function(
                "() => [...document.querySelectorAll('.rec-card')]"
                ".some(c => c.textContent.includes('E2E 自动配方'))", timeout=30000)
            return f"rec-cards={page.locator('.rec-card').count()}"
        step("我的配方货架出现", _recipe_card)

        def _custom_pipe():
            page.click("#tab-lib")
            page.wait_for_timeout(200)
            page.locator(".rec-card").filter(has_text="E2E 自动配方").first \
                .get_by_text("流水线").click()
            page.wait_for_timeout(300)
            wait_pipe(page)
            st = page.locator("#pipeStatus").inner_text()
            if "完成" not in st:
                raise AssertionError(st[:150])
            return st[:100]
        step("自定义配方直跑流水线", _custom_pipe)

        # ---------------- LLM Prompt ----------------
        def _llm_describe():
            page.click("#view-blocks")
            page.wait_for_timeout(300)
            page.click("#tab-llm")
            page.select_option("#pmode", "describe")
            page.get_by_text("生成").first.click()
            page.wait_for_function(
                "() => (document.querySelector('#prompt')?.value || '').length > 100",
                timeout=30000)
            return f"prompt_len={len(page.locator('#prompt').input_value())}"
        step("LLM describe 生成", _llm_describe)

        def _llm_debug():
            page.select_option("#pmode", "debug")
            page.fill("#problem", "训练后 BER 一直很高")
            page.get_by_text("生成").first.click()
            page.wait_for_function(
                "() => (document.querySelector('#prompt')?.value || '').includes('BER 一直很高')",
                timeout=30000)
            return "debug prompt ok"
        step("LLM debug 生成", _llm_debug)

        def _llm_tune():
            page.select_option("#pmode", "tune")
            page.fill("#metrics", '{"ber":0.21,"psnr":28.5}')
            page.get_by_text("生成").first.click()
            page.wait_for_function(
                "() => (document.querySelector('#prompt')?.value || '').includes('调参任务')",
                timeout=30000)
            return "tune prompt ok"
        step("LLM tune 生成", _llm_tune)

        def _apply_reply():
            rid = page.evaluate(
                "() => { const ms=(blocksSpec&&blocksSpec.modules)||[];"
                " const r=ms.find(m=>m.type==='res_block')||ms[0]; return r&&r.id; }")
            if not rid:
                raise AssertionError("当前没有可打补丁的模块")
            reply = "\n".join([
                "```json",
                '{"analysis":"t","patch":[{"id":"%s","replace_type":"se_block"}],'
                '"risk":"low"}' % rid,
                "```",
            ])
            page.fill("#reply", reply)
            page.get_by_text("解析并套用补丁").click()
            page.wait_for_function(
                "() => { const e=document.querySelector('#replyStatus');"
                " return e && (e.textContent.includes('已套用') || e.className.includes('err')); }",
                timeout=60000)
            st = page.locator("#replyStatus").inner_text()
            if "已套用" not in st:
                raise AssertionError(st[:150])
            return st[:80]
        step("LLM 回复套补丁", _apply_reply)

        def _llm_pipe_no_spec():
            page.click("#view-blocks")
            page.get_by_text("清空组装台").click()
            page.click("#view-pipe")
            page.select_option("#fw", "lsb_toy")
            page.click("#tab-llm")
            page.select_option("#pmode", "describe")
            js_before = len(JS_ERRORS)
            page.get_by_text("生成").first.click()
            page.wait_for_timeout(1200)
            val = page.locator("#prompt").input_value()
            new_err = JS_ERRORS[js_before:]
            if val.strip() in ("", "undefined") or new_err:
                raise AssertionError(f"prompt={val[:40]!r}, js={new_err[:2]}")
            return val[:60]
        step("流水线视图无结构时 LLM 生成 (容错)", _llm_pipe_no_spec)

        # ---------------- 训练面板 ----------------
        def _train_ui():
            page.click("#view-train")
            page.wait_for_function(
                "() => document.querySelectorAll('#atkChecks input').length === 7",
                timeout=20000)
            opts = page.locator("#tFw option").count()
            page.wait_for_function(
                "() => { const e=document.querySelector('#fsList');"
                " return e && e.textContent.length > 0; }", timeout=60000)
            return f"atk=7, tFw_opts={opts}, fs={page.locator('#fsList').inner_text()[:40]!r}"
        step("训练面板渲染 + 目录扫描", _train_ui)

        def _train_status():
            page.wait_for_timeout(2500)
            return f"tProg={page.locator('#tProg').inner_text()!r}"
        step("训练状态轮询", _train_status)

        def _train_compare():
            page.wait_for_function(
                "() => [...document.querySelectorAll('#cmpRuns label')]"
                ".some(l => l.textContent.includes('_e2e_cmp_1125'))",
                timeout=20000)
            page.locator("#cmpRuns label", has_text="_e2e_cmp_1125"
                         ).locator("input").check()
            page.wait_for_function(
                "() => document.querySelectorAll('#cmpChart path').length >= 1 "
                "&& document.querySelectorAll('#cmpTable tr').length === 2",
                timeout=10000)
            return (f"paths={page.locator('#cmpChart path').count()}, "
                    f"legend={page.locator('#cmpLegend').inner_text()[:60]!r}")
        step("run 对比面板 (勾选叠加曲线+表格)", _train_compare)

        def _back_pipe():
            page.click("#view-pipe")
            page.wait_for_timeout(500)
            return "back to pipe"
        step("视图切回流水线", _back_pipe)

        def _cleanup():
            page.click("#view-blocks")
            page.wait_for_timeout(800)
            card = page.locator(".rec-card").filter(has_text="E2E 自动配方").first
            if card.count() == 0:
                return "no card"
            card.get_by_text("删除").click()
            page.wait_for_timeout(1500)
            return f"left={page.locator('.rec-card').filter(has_text='E2E 自动配方').count()}"
        step("删除自定义配方 (清理)", _cleanup)

        browser.close()

    print("\n================ SUMMARY ================")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = [(n, d) for n, ok, d in RESULTS if not ok]
    print(f"PASS {passed}/{len(RESULTS)}")
    for nme, det in failed:
        print(f"  FAILED: {nme} :: {det}")
    print(f"JS errors captured: {len(JS_ERRORS)}")
    for e in JS_ERRORS[:20]:
        print("  -", e)
    return 0 if (not failed and not JS_ERRORS) else 1


def main():
    try:
        import playwright  # noqa: F401
    except Exception:
        print("[skip] 未安装 playwright, 跳过 WebUI e2e "
              "(pip install playwright && playwright install chromium)")
        return 0
    lib = os.environ.get("WMFLOW_PW_LIB")
    if lib:
        old = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = lib + ((":" + old) if old else "")
    token = os.environ.get("WMFLOW_E2E_TOKEN", "")
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    # run 对比面板需要至少一个带 history.json 的 run (CI 的 runs/ 是空的)
    cmp_dir = REPO / "runs" / "_e2e_cmp_1125"
    try:
        cmp_dir.mkdir(parents=True, exist_ok=True)
        import json as _json
        (cmp_dir / "history.json").write_text(_json.dumps({
            "framework": "hidden", "recipe": "hidden", "size": 48,
            "msg_len": 30, "dataset_path": "/tmp/ds", "n_images": 10,
            "attacks": ["none"], "epochs": 2, "batch_size": 4, "lr": 1e-3,
            "strength": 1.0, "val_ratio": 0.1, "seed": 0,
            "started_at": 1732500000.0, "updated_at": 1732500060.0,
            "finished": True,
            "history": [
                {"epoch": 1, "loss": 0.9, "loss_img": 0.5, "loss_msg": 0.4,
                 "ber": 0.4, "val_n": 1, "train_n": 9, "val_ber": 0.4},
                {"epoch": 2, "loss": 0.7, "loss_img": 0.4, "loss_msg": 0.3,
                 "ber": 0.3, "val_n": 1, "train_n": 9, "val_ber": 0.3},
            ]}), encoding="utf-8")
    except Exception as e:
        print(f"[warn] 无法构造对比 run: {e}")
    cmd = [sys.executable, "webui.py", "--host", "127.0.0.1", "--port", str(port)]
    if token:
        cmd += ["--token", token]
    proc = subprocess.Popen(
        cmd, cwd=str(REPO), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        if not _wait_server(base, token=token):
            print("[FAIL] webui 未能在 60s 内启动")
            return 1
        return _run_tests(base, token)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        try:
            (cmp_dir / "history.json").unlink(missing_ok=True)
            cmp_dir.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())