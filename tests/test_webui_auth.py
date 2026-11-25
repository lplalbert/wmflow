#!/usr/bin/env python
"""WebUI 访问控制回归测试 (P0)。

覆盖:
- token 鉴权: 带 ?token= 的首页可打开 (曾经因未剥 query 而 404); API 无 token -> 401,
  错误 token -> 401, 正确 token -> 200。
- FS_ROOTS 约束同样作用于「启动训练」: 数据集路径不在白名单内被拒。
- val_ratio 非法 (>=1) 被拒, 且不真的启动训练。

不依赖 torch 也能跑 (webui 里 torch 是函数内 import), 但服务启动会 import wmflow 包。
纯 ASCII 文案, 防 Windows GBK 控制台炸。
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _req(base, path, method="GET", body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers=headers)
    try:
        r = urllib.request.urlopen(req, timeout=8)
        return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _wait(base, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(base + "/api/modules?token=secret",
                                   timeout=2).read()
            return True
        except Exception:
            time.sleep(0.4)
    return False


def test_webui_auth_and_fsroots():
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    root = tempfile.mkdtemp(prefix="wmflow_fs_")
    env = dict(os.environ, WMFLOW_WEBUI_TOKEN="secret", WMFLOW_FS_ROOTS=root)
    proc = subprocess.Popen(
        [sys.executable, "webui.py", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(REPO), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        assert _wait(base), "webui 未能在限定时间内启动"

        # 1) 首页: 裸 / 与 ?token= 都应 200 (回归: 带 query 曾 404)
        st, _ = _req(base, "/")
        assert st == 200, f"GET / -> {st}"
        st, _ = _req(base, "/?token=secret")
        assert st == 200, f"GET /?token=secret -> {st} (token 页面入口 404 回归)"
        st, _ = _req(base, "/favicon.ico")
        assert st in (200, 204), f"favicon -> {st}"

        # 2) API 鉴权
        st, _ = _req(base, "/api/modules?token=secret")
        assert st == 200, f"api 带 token -> {st}"
        st, _ = _req(base, "/api/modules")
        assert st == 401, f"api 无 token -> {st} (应 401)"
        st, _ = _req(base, "/api/modules?token=wrong")
        assert st == 401, f"api 错误 token -> {st} (应 401)"

        # 3) FS_ROOTS 也约束训练启动 (白名单外目录被拒)
        st, b = _req(base, "/api/train_start?token=secret", "POST",
                     {"dataset_path": "/etc", "framework": "hidden"})
        d = json.loads(b)
        assert d.get("ok") is False and "须在" in d.get("error", ""), \
            f"白名单外数据集未被拒: {d}"

        # 4) val_ratio 非法被拒, 且不启动训练
        st, b = _req(base, "/api/train_start?token=secret", "POST",
                     {"dataset_path": root, "framework": "hidden",
                      "val_ratio": 2})
        d = json.loads(b)
        assert d.get("ok") is False and "val_ratio" in d.get("error", ""), \
            f"非法 val_ratio 未被拒: {d}"

        print("[ok] webui token routing + FS_ROOTS/val_ratio guards")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        import shutil
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_webui_auth_and_fsroots()
    print("ALL WEBUI-AUTH TESTS PASSED")
