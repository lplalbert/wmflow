"""自定义配方库: 把组装台/混搭的结果存成可复用配方 (JSON 落盘), 支持导出/导入。

一条自定义配方 = encoder NetSpec + decoder NetSpec + 元信息, 与内置四配方同构
(`RECIPES[fw]()` 返回 `(enc, dec, meta)`), 因此可直接被流水线 / 训练 / LLM prompt
复用, 无需再走一次组装。

落盘: `<repo>/custom_recipes/<slug>.json` (环境变量 `WMFLOW_CUSTOM_RECIPES` 可覆盖)。
文件即接口: 导出的 .json 可直接导回; 手写 JSON 也能入库 (入库前一律 build 校验)。

约定:
  - 存盘前把 decoder 的 bit_head.msg_len 对齐 encoder 的消息长 (混搭 30↔256bit 常见),
    对齐与否记在 meta.msg_len_aligned, 出问题一律报错不静默写坏文件。
  - 写入走 tmp + os.replace, 半截文件不会留在目录里。
  - 名字允许中文 (文件名保留中文), 但路径分隔符/空白一律折成下划线, 防越权写盘。
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .netspec import NetSpec

FORMAT = "wmflow.custom_recipe/1"
ROOT = Path(__file__).resolve().parent.parent
CUSTOM_DIR = Path(os.environ.get("WMFLOW_CUSTOM_RECIPES")
                  or (ROOT / "custom_recipes"))

_SLUG_BAD = re.compile(r"[^\w\u4e00-\u9fff\-]+")
# Windows 保留设备名 (含带扩展名的 CON.json 也非法), 一律拒
_RESERVED = {"CON", "PRN", "AUX", "NUL",
             *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}

_LOCKS: Dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(name: str) -> threading.Lock:
    """同名操作串行化用的进程内锁 (webui 是单进程多线程)。"""
    key = str(name).casefold()
    with _LOCKS_GUARD:
        lk = _LOCKS.get(key)
        if lk is None:
            lk = _LOCKS[key] = threading.Lock()
        return lk



# --------------------------------------------------------------------- 工具
def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(name: str) -> str:
    """配方名 → 安全文件名主干 (去空白/路径分隔符, 限长 48)。"""
    s = _SLUG_BAD.sub("_", (name or "").strip()).strip("_").strip(".")
    if not s:
        raise ValueError("配方名不能为空, 也不能只含非法字符")
    if s.split(".")[0].upper() in _RESERVED:
        raise ValueError(f"'{s}' 是 Windows 保留设备名, 换个名字")
    return s[:48]


def _path(name: str) -> Path:
    p = CUSTOM_DIR / f"{slugify(name)}.json"
    # 防目录逃逸/symlink: 解析后的父目录必须就是配方目录
    base = CUSTOM_DIR.resolve()
    if p.parent.resolve() != base:
        raise ValueError(f"配方路径越界: {p}")
    return p


def _read(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write(p: Path, rec: Dict[str, Any]) -> None:
    """独立临时文件 + 原子替换; 失败清理临时文件 (同名并发也不会互相踩)。"""
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.stem + ".",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        tmp = None
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


# --------------------------------------------------------------- 校验/对齐
def align_decoder(enc_spec: Dict[str, Any],
                  dec_spec: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """把 decoder 的 bit_head.msg_len 对齐 encoder 的消息长。返回 (spec, 是否改过)。"""
    enc, dec = NetSpec(enc_spec), NetSpec(dec_spec)
    if "message" not in enc.inputs:      # 不是编码器: 不猜, 交给 _validate 报清晰错
        return dec_spec, False
    want = enc.inputs["message"][0]
    patch = []
    for m in dec.spec["modules"]:
        if m["type"] == "bit_head":
            cur = (m.get("params") or {}).get("msg_len")
            if cur is None or int(cur) != int(want):
                patch.append({"id": m["id"], "params": {"msg_len": int(want)}})
    if not patch:
        return dec_spec, False
    return dec.apply_patch(patch).spec, True


def _validate(enc_spec: Dict[str, Any],
              dec_spec: Dict[str, Any]) -> Tuple[NetSpec, NetSpec, Dict[str, Any]]:
    """入库前硬校验: 两侧能 build、图像形状首尾相接、消息长自洽, 并真跑一次前向。"""
    enc, dec = NetSpec(enc_spec), NetSpec(dec_spec)
    try:
        eb, db = enc.build(), dec.build()
    except Exception as e:                       # NetSpecError 等
        raise ValueError(f"配方构建失败: {e}") from e
    if "message" not in enc.inputs:
        raise ValueError("encoder 缺 message 输入, 不是图文水印编码器")
    if "image" not in enc.inputs or "image" not in dec.inputs:
        raise ValueError("encoder 与 decoder 都要有 image 输入")
    ein = tuple(enc.inputs["image"])
    din = tuple(dec.inputs["image"])
    if len(ein) != 3 or ein[0] != 3:
        raise ValueError(f"encoder 输入图像应为 (3,H,W), 实得 {ein}")
    msg_len = int(enc.inputs["message"][0])
    out = tuple(eb.out_shape)
    if len(out) != 3 or out[0] != 3:
        raise ValueError(f"encoder 输出应为 (3,H,W), 实得 {out}")
    # 关键: encoder 输出尺寸必须与 decoder 声明接收的尺寸一致 (否则存进去也串不起来)
    if out != din:
        raise ValueError(f"encoder 输出 {out} 与 decoder 输入 {din} 尺寸不一致")
    if tuple(db.out_shape) != (msg_len,):
        raise ValueError(f"decoder 输出 {tuple(db.out_shape)} 与消息长 {msg_len} 不一致")

    # 真跑一次 enc→dec: 抓 build 查不出的运行期不兼容 (通道/广播/算子)
    forward_checked = False
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        _, H, W = out
        try:
            with torch.no_grad():
                x = torch.rand(1, 3, H, W)
                m = (torch.rand(1, msg_len) > 0.5).float()
                y = eb.forward({"image": x, "message": m})
                if tuple(y.shape) != (1, *out):
                    raise ValueError(f"encoder 实跑输出 {tuple(y.shape)} ≠ (1,{out})")
                z = db.forward({"image": y})
            if tuple(z.shape) != (1, msg_len):
                raise ValueError(f"前向输出 {tuple(z.shape)} ≠ (1,{msg_len})")
            forward_checked = True
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"enc→dec 串接前向失败 (结构拼不起来): {e}") from e
    return enc, dec, {"msg_len": msg_len, "size": int(ein[1]),
                      "out_shape": list(out), "forward_checked": forward_checked}


# ------------------------------------------------------------------- 增删查
def _summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    meta = rec.get("meta") or {}
    return {
        "name": rec.get("name") or "",
        "title": rec.get("title") or rec.get("name") or "",
        "msg_len": meta.get("msg_len"),
        "size": meta.get("size"),
        "out_shape": meta.get("out_shape"),
        "created": rec.get("created", ""),
        "updated": rec.get("updated", ""),
        "note": meta.get("note", ""),
        "source": meta.get("source"),
        "n_modules": [len((rec.get("encoder") or {}).get("modules") or []),
                      len((rec.get("decoder") or {}).get("modules") or [])],
        "path": str(_path(rec.get("name") or "x")),
    }


_DERIVED = ("msg_len", "size", "out_shape", "forward_checked")


def save(name: str, encoder: Dict[str, Any], decoder: Dict[str, Any], *,
         title: Optional[str] = None, note: str = "",
         source: Any = None, overwrite: bool = False,
         meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """存一条自定义配方 (先对齐 decoder 消息长, 再硬校验, 最后原子写盘)。"""
    p = _path(name)                          # 先把名字/路径问题挡掉
    dec_fixed, aligned = align_decoder(encoder, decoder)
    enc, dec, info = _validate(encoder, dec_fixed)
    with _lock_for(p.stem):                  # 存在检查与写入同一临界区
        if p.exists() and not overwrite:
            raise FileExistsError(f"配方 '{p.stem}' 已存在 (勾选覆盖才重写)")
        now = _now()
        m: Dict[str, Any] = dict(info)
        for k, v in (meta or {}).items():    # 派生字段不许被 meta 覆盖
            if k not in _DERIVED:
                m[k] = v
        if note:
            m["note"] = note
        if source:
            m["source"] = source
        if aligned:
            m["msg_len_aligned"] = True
        rec = {"format": FORMAT, "name": p.stem,
               "title": (title or p.stem).strip() or p.stem,
               "created": now, "updated": now, "meta": m,
               "encoder": enc.spec, "decoder": dec.spec}
        if p.exists():                       # 覆盖时保留首次创建时间
            try:
                rec["created"] = _read(p).get("created") or now
            except Exception:
                pass
        _write(p, rec)
    return _summary(rec)


def list_all() -> List[Dict[str, Any]]:
    """列出全部自定义配方 (最近更新在前; 坏文件/并发删除都不让整表失败)。"""
    if not CUSTOM_DIR.is_dir():
        return []
    entries = []
    for p in CUSTOM_DIR.glob("*.json"):
        try:
            entries.append((p.stat().st_mtime, p))
        except OSError:                 # 并发删除: 直接跳过
            continue
    out: List[Dict[str, Any]] = []
    for _mt, p in sorted(entries, reverse=True):
        try:
            out.append(_summary(_read(p)))
        except Exception as e:
            out.append({"name": p.stem, "title": p.stem, "error": str(e)})
    return out


def get(name: str) -> Dict[str, Any]:
    p = _path(name)
    if p.is_symlink():          # 防止目录被放入指向外部的链接
        raise ValueError(f"配方文件是符号链接, 拒绝读取: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"没有这条自定义配方: {p.stem}")
    rec = _read(p)
    if not rec.get("encoder") or not rec.get("decoder"):
        raise ValueError(f"配方文件不完整 (缺 encoder/decoder): {p}")
    return rec


def pair(name: str) -> Tuple[NetSpec, NetSpec, Dict[str, Any]]:
    """取一条配方 → (enc, dec, meta), 与内置 RECIPES[fw]() 同形, 可直接喂流水线/训练。"""
    rec = get(name)
    meta = dict(rec.get("meta") or {})
    meta.update({"framework": f"custom:{rec['name']}",
                 "title": rec.get("title") or rec["name"],
                 "custom": True})
    return NetSpec(rec["encoder"]), NetSpec(rec["decoder"]), meta


def exists(name: str) -> bool:
    return _path(name).is_file()


def delete(name: str) -> bool:
    p = _path(name)
    with _lock_for(p.stem):
        if not p.exists() or p.is_symlink():
            raise FileNotFoundError(f"没有这条自定义配方: {p.stem}")
        p.unlink()
    return True


# --------------------------------------------------------------- 导出/导入
def export(name: str) -> Tuple[str, str]:
    """导出为可分享的 JSON 文本。返回 (filename, text)。"""
    rec = get(name)
    return f"{rec['name']}.json", json.dumps(rec, ensure_ascii=False, indent=1)


def import_json(payload: Any, *, name: Optional[str] = None,
                overwrite: bool = False) -> Dict[str, Any]:
    """从 JSON (文本或 dict) 导入一条配方。

    接受三种写法: 本库导出的完整记录 / {"encoder":…,"decoder":…} /
    {"recipe": {…}} (webui 存盘载荷)。名字优先级: 显式 name > 记录里的 name。
    """
    if isinstance(payload, str):
        try:
            d = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}") from e
    else:
        d = payload
    if not isinstance(d, dict):
        raise ValueError("导入内容应为 JSON 对象")
    if "recipe" in d and isinstance(d["recipe"], dict):
        d = d["recipe"]
    enc = d.get("encoder") or d.get("enc")
    dec = d.get("decoder") or d.get("dec")
    if not isinstance(enc, dict) or not isinstance(dec, dict):
        raise ValueError('导入 JSON 需要 {"encoder": {...}, "decoder": {...}} 两段 NetSpec')
    src_name = name or d.get("name") or "imported"
    return save(src_name, enc, dec,
                title=d.get("title") or src_name,
                note=(d.get("meta") or {}).get("note", ""),
                source={"kind": "import"}, overwrite=overwrite)
