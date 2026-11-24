"""wmflow 训练器: 可微攻击 + 服务器数据集 + 训练环 (webui 后台线程调用)。

与推理图同构: encoder/decoder 来自 NetSpec (recipes), 训练环在两者之间插
**可微攻击层** (梯度可回传, 与评测 LAYERS 同名思想):
  none / gaussian_noise / blur / jpeg_approx (DCT+STE) / crop_resize /
  dropout / combo (随机组合)

损失: w_img * MSE(emb, cover) + w_msg * BCE(logits, bits)
emb = (1-strength)*cover + strength*enc(cover, bits)   (与 run_demo 同口径)

数据集: 服务器目录扫图 (jpg/png/bmp/webp), 预载内存, 数量可限。
ckpt 落 runs/ (enc+dec state_dict + spec + config), run_demo 可加载。
"""
from __future__ import annotations

import glob
import os
import threading
import time
from typing import Any, Dict, List

import numpy as np

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "runs")

ATTACKS = ["none", "gaussian_noise", "blur", "jpeg_approx",
           "crop_resize", "dropout", "combo"]


# ---------------------------------------------------------------- 可微攻击
def _ste_round(x):
    import torch

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t):
            return t.round()

        @staticmethod
        def backward(ctx, g):
            return g

    return _F.apply(x)


_DCT_CACHE: Dict[int, Any] = {}


def _dct_mat(n=8, device="cpu"):
    import torch
    key = n
    if key not in _DCT_CACHE:
        k = torch.arange(n, dtype=torch.float32)
        m = torch.cos((2 * k[None] + 1) * k[:, None] * np.pi / (2 * n))
        m[0] *= 1 / np.sqrt(2)
        _DCT_CACHE[key] = m * np.sqrt(2 / n)
    return _DCT_CACHE[key].to(device)


def atk_jpeg_approx(x, rng, quality=None):
    """8x8 DCT + STE 量化近似 JPEG (可微)。"""
    import torch
    q = quality or int(rng.integers(45, 90))
    B, C, H, W = x.shape
    D = _dct_mat(8, x.device)
    # 标准 JPEG 亮度量化表 (缩放到 q)
    base = torch.tensor([
        [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
    ], dtype=torch.float32, device=x.device)
    scale = (100 - q) / 50.0 + 0.5
    qtab = (base * scale).clamp(min=1)

    def blocks(t):
        return (t.reshape(B * C, H // 8, 8, W // 8, 8)
                .permute(0, 1, 3, 2, 4).reshape(-1, 8, 8))

    def unblocks(t):
        return (t.reshape(B * C, H // 8, W // 8, 8, 8)
                .permute(0, 1, 3, 2, 4).reshape(B, C, H, W))

    xb = blocks(x)
    coef = D @ xb @ D.T
    coef = _ste_round(coef / qtab) * qtab
    out = D.T @ coef @ D
    return unblocks(out).clamp(0, 1)


def atk_gaussian_noise(x, rng):
    import torch
    sigma = float(rng.uniform(0.01, 0.05))
    return (x + torch.randn_like(x) * sigma).clamp(0, 1)


def atk_blur(x, rng):
    import torch
    import torch.nn.functional as F
    k = int(rng.choice([3, 5]))
    sig = float(rng.uniform(0.6, 1.6))
    g = torch.arange(k, dtype=torch.float32, device=x.device) - k // 2
    ker = torch.exp(-(g ** 2) / (2 * sig ** 2))
    ker = (ker / ker.sum())
    k2 = ker[None, :] * ker[:, None]
    k2 = k2.expand(x.shape[1], 1, k, k)
    return F.conv2d(x, k2, padding=k // 2, groups=x.shape[1]).clamp(0, 1)


def atk_crop_resize(x, rng):
    import torch.nn.functional as F
    _, _, H, W = x.shape
    frac = float(rng.uniform(0.85, 0.99))
    h, w = int(H * frac), int(W * frac)
    top = int(rng.integers(0, H - h + 1))
    left = int(rng.integers(0, W - w + 1))
    crop = x[:, :, top:top + h, left:left + w]
    return F.interpolate(crop, size=(H, W), mode="bilinear",
                         align_corners=False)


def atk_dropout(x, rng):
    import torch
    out = x.clone()
    _, _, H, W = x.shape
    for _ in range(int(rng.integers(1, 4))):
        h = int(rng.integers(H // 16, max(H // 4, H // 16 + 1)))
        w = int(rng.integers(W // 16, max(W // 4, W // 16 + 1)))
        t = int(rng.integers(0, H - h + 1))
        l = int(rng.integers(0, W - w + 1))
        out[:, :, t:t + h, l:l + w] = 0
    return out


_ATOMIC = {"none": lambda x, r: x, "gaussian_noise": atk_gaussian_noise,
           "blur": atk_blur, "jpeg_approx": atk_jpeg_approx,
           "crop_resize": atk_crop_resize, "dropout": atk_dropout}


def apply_attacks(x, attacks: List[str], rng: np.random.Generator):
    import random as _r
    names = [a for a in attacks if a in _ATOMIC] or ["none"]
    if "combo" in attacks:
        picked = _r.sample([a for a in _ATOMIC if a != "none"],
                           k=int(rng.integers(1, 3)))
    else:
        picked = names
    for name in picked:
        x = _ATOMIC[name](x, rng)
    return x, picked


# ---------------------------------------------------------------- 数据集
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _count_images(path: str) -> int:
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += sum(1 for f in files
                 if os.path.splitext(f)[1].lower() in IMG_EXTS)
        if n > 20000:
            break
    return n


def scan_dataset(path: str):
    """目录里的图片数 (递归) 与子目录 (一层, 含各自递归图片数)。"""
    out = {"path": path, "dirs": []}
    try:
        entries = sorted(os.scandir(path), key=lambda e: e.name)
    except OSError as e:
        return {"error": str(e)}
    n_here = 0
    for e in entries:
        if e.is_dir():
            out["dirs"].append({"name": e.name, "path": e.path,
                                "n_images": _count_images(e.path)})
        elif e.is_file() and os.path.splitext(e.name)[1].lower() in IMG_EXTS:
            n_here += 1
    out["n_images"] = n_here or _count_images(path)
    return out


def load_dataset(path: str, n: int, size: int, seed: int):
    import cv2
    files = sorted(f for f in glob.glob(os.path.join(path, "*"))
                   if os.path.splitext(f)[1].lower() in IMG_EXTS)
    if not files:
        # 递归回退 (如 DIV2K/DIV2K_train/*.png 两级结构)
        for _root, _dirs, fs in os.walk(path):
            files.extend(os.path.join(_root, f) for f in fs
                         if os.path.splitext(f)[1].lower() in IMG_EXTS)
        files = sorted(files)
    if not files:
        raise ValueError(f"目录无图片 (含子目录): {path}")
    rng = np.random.default_rng(seed)
    if n and n < len(files):
        idx = rng.choice(len(files), n, replace=False)
        files = sorted(files[i] for i in idx)
    imgs = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (size, size))
        imgs.append(img[:, :, ::-1].astype(np.float32) / 255.0)
    if not imgs:
        raise ValueError("没有可解码的图片")
    return np.stack(imgs), files


# ---------------------------------------------------------------- 训练状态
class TrainState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.stop_flag = False
        self.config: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.log: List[str] = []
        self.progress = {"epoch": 0, "step": 0, "total_steps": 0}
        self.previews: Dict[str, str] = {}
        self.ckpts: List[str] = []
        self.error = None
        self.started_at = None

    def say(self, msg):
        with self.lock:
            self.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            self.log = self.log[-200:]

    def snapshot(self):
        with self.lock:
            return {
                "running": self.running,
                "error": self.error,
                "config": self.config,
                "history": list(self.history),
                "log": list(self.log[-60:]),
                "progress": dict(self.progress),
                "previews": dict(self.previews),
                "ckpts": list(self.ckpts[-5:]),
                "elapsed": (time.time() - self.started_at) if self.started_at else 0,
            }


STATE = TrainState()


# ---------------------------------------------------------------- 训练环
def train(config: Dict[str, Any]):
    import cv2
    import torch
    import torch.nn.functional as F
    from .netspec import NetSpec
    from .recipes import RECIPES

    st = STATE
    fw = str(config["framework"])
    if fw.startswith("custom:"):
        from . import custom_recipes as crec
        name = fw.split(":", 1)[1]
        st.say(f"构建模型: 自定义配方 {name}")
        enc_spec, dec_spec, meta = crec.pair(name)
        size_eff = int(enc_spec.inputs["image"][1])
        if int(config.get("size") or size_eff) != size_eff:
            st.say(f"注意: 配方固定 {size_eff}², 已忽略表单尺寸 {config.get('size')}")
        config = dict(config)
        config["size"] = size_eff          # 配方尺寸优先, 下游一律用 config["size"]
        st.say(f"配方结构: {size_eff}² · {enc_spec.inputs['message'][0]}bit · "
               f"{meta.get('title', name)}")
    else:
        st.say(f"构建模型: {fw}")
        if config.get("msg_len"):
            enc_spec, dec_spec, meta = RECIPES[fw](
                msg_len=int(config["msg_len"]), size=config["size"])
        else:
            enc_spec, dec_spec, meta = RECIPES[fw](size=config["size"])
    enc_b, dec_b = enc_spec.build(), dec_spec.build()
    enc, dec = enc_b.as_module(), dec_b.as_module()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc.to(device).train()
    dec.to(device).train()
    st.say(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})"
                               if device == "cuda" else ""))

    # 训练前梯度自检 (Codex 计划要求): 一次前向+反向, 确认两侧参数都参与优化且梯度有限。
    # 跨框架混搭/自定义配方最怕某侧参数没接上, 训练白跑几十轮才发现。
    msg_len_chk = int(enc_spec.inputs["message"][0])
    sz_chk = int(config["size"])
    x = torch.rand(2, 3, sz_chk, sz_chk, device=device)
    b = (torch.rand(2, msg_len_chk, device=device) > 0.5).float()
    emb = (1 - float(config.get("strength", 1.0))) * x + \
          float(config.get("strength", 1.0)) * enc({"image": x, "message": b})
    chk = F.mse_loss(emb, x) + F.binary_cross_entropy_with_logits(
        dec({"image": emb}), b)
    chk.backward()
    bad = []
    for nm, mod in (("encoder", enc), ("decoder", dec)):
        gs = [p.grad for p in mod.parameters() if p.requires_grad]
        if not gs or all(g is None for g in gs):
            bad.append(f"{nm} 全部参数没有梯度")
        elif any(g is not None and not torch.isfinite(g).all() for g in gs):
            bad.append(f"{nm} 梯度含 inf/nan")
        elif all(g is not None and float(g.abs().sum()) == 0.0 for g in gs
                 if g is not None):
            # 全零梯度 = 该侧不进优化 (典型: strength=0 时 encoder 输出被完全丢弃)
            bad.append(f"{nm} 梯度全为 0 (该侧不会更新: 检查 strength 与连接)")
    enc.zero_grad(set_to_none=True)
    dec.zero_grad(set_to_none=True)
    if bad:
        raise ValueError("训练前梯度自检失败: " + "; ".join(bad) +
                         " (该结构这一侧可能是死分支, 训不动)")
    st.say("梯度自检通过: encoder/decoder 两侧均有有限梯度")

    data, files = load_dataset(config["dataset_path"], int(config.get("n_images", 200)),
                               int(config["size"]), int(config.get("seed", 0)))
    st.say(f"数据集: {len(files)} 张 (目录 {config['dataset_path']})")
    n = data.shape[0]
    data_t = torch.from_numpy(data).permute(0, 3, 1, 2).to(device)

    msg_len = enc_spec.inputs["message"][0]
    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=float(config.get("lr", 1e-3)))
    bs = int(config.get("batch_size", 16))
    epochs = int(config.get("epochs", 5))
    strength = float(config.get("strength", 1.0))
    w_img = float(config.get("w_img", 1.0))
    w_msg = float(config.get("w_msg", 1.0))
    attacks = list(config.get("attacks", ["combo"])) or ["none"]
    steps_per = (n + bs - 1) // bs

    os.makedirs(RUNS_DIR, exist_ok=True)
    run_name = f"{fw.replace(':', '-')}_{time.strftime('%m%d_%H%M%S')}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    rng = np.random.default_rng(int(config.get("seed", 0)))
    st.say(f"开训: {epochs} epoch × {steps_per} step, bs={bs}, "
           f"lr={config.get('lr', 1e-3)}, 攻击={attacks}")

    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        ep_loss = ep_img = ep_msg = 0.0
        for si in range(0, n, bs):
            if st.stop_flag:
                st.say("收到停止信号, 中断")
                break
            idx = perm[si:si + bs]
            cover = data_t[idx]
            bits = (torch.rand(cover.shape[0], msg_len, device=device) > 0.5).float()
            emb_raw = enc({"image": cover, "message": bits})
            emb = (1 - strength) * cover + strength * emb_raw
            attacked, picked = apply_attacks(emb, attacks, rng)
            logits = dec({"image": attacked})
            loss_msg = F.binary_cross_entropy_with_logits(logits, bits)
            loss_img = F.mse_loss(emb, cover)
            loss = w_img * loss_img + w_msg * loss_msg
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach())
            ep_img += float(loss_img.detach())
            ep_msg += float(loss_msg.detach())
            st.progress.update({"epoch": ep, "step": si // bs + 1,
                                "total_steps": steps_per})
        if st.stop_flag:
            break
        # epoch 末 BER (攻击后)
        with torch.no_grad():
            bits_v = (torch.rand(n, msg_len, device=device) > 0.5).float()
            emb_v = (1 - strength) * data_t + strength * enc(
                {"image": data_t, "message": bits_v})
            atk_v, _ = apply_attacks(emb_v, attacks, rng)
            logit_v = dec({"image": atk_v})
            # bit_head 出 logits, BCEWithLogits 的判决边界是 0 (不是 0.5)
            ber = float(((logit_v > 0).float() != bits_v).float().mean())
        rec = {"epoch": ep, "loss": round(ep_loss / steps_per, 5),
               "loss_img": round(ep_img / steps_per, 5),
               "loss_msg": round(ep_msg / steps_per, 5),
               "ber": round(ber, 4)}
        st.history.append(rec)
        st.say(f"epoch {ep}: loss={rec['loss']} img={rec['loss_img']} "
               f"msg={rec['loss_msg']} ber={rec['ber']}")
        # 预览图 (首样本)
        with torch.no_grad():
            one = data_t[:1]
            b1 = (torch.rand(1, msg_len, device=device) > 0.5).float()
            e1 = (1 - strength) * one + strength * enc({"image": one, "message": b1})
            a1, _ = apply_attacks(e1, attacks, rng)

            def png(t):
                u8 = (t[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                      ).astype(np.uint8)
                ok, buf = cv2.imencode(".png", u8[:, :, ::-1])
                import base64
                return base64.b64encode(buf.tobytes()).decode("ascii")

            st.previews = {"cover": png(one), "watermarked": png(e1),
                           "attacked": png(a1)}
        # ckpt
        ckpt = os.path.join(run_dir, f"ckpt_ep{ep}.pt")
        torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                    "framework": fw, "recipe": fw, "msg_len": msg_len,
                    "size": config["size"], "spec_enc": enc_spec.spec,
                    "spec_dec": dec_spec.spec, "epoch": ep,
                    "config": config}, ckpt)
        st.ckpts.append(ckpt)

    st.say("训练结束" + (", 已提前停止" if st.stop_flag else ""))
    return run_dir


def start_training(config: Dict[str, Any]) -> str:
    if STATE.running:
        raise RuntimeError("已有训练在跑, 先停止")
    for k in ("dataset_path", "framework"):
        if not config.get(k):
            raise ValueError(f"缺参数 {k}")
    if not os.path.isdir(config["dataset_path"]):
        raise ValueError(f"数据集目录不存在: {config['dataset_path']}")
    STATE.__init__()
    STATE.config = dict(config)
    STATE.running = True
    STATE.started_at = time.time()

    def _run():
        try:
            train(config)
        except Exception as e:
            STATE.error = str(e)
            STATE.say(f"出错: {e}")
        finally:
            STATE.running = False

    threading.Thread(target=_run, daemon=True).start()
    return "started"
