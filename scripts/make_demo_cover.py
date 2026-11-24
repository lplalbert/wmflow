#!/usr/bin/env python
"""生成示例载体图 demo/cover.png —— 让 workflows/smoke_ablation.json 开箱即跑。

demo/ 被 .gitignore 忽略 (运行时资产), 所以新 clone 的仓库里没有封面图;
先跑本脚本再跑 run.py 即可复现冒烟流水线。图像是确定性合成纹理, 不依赖外部素材。

用法:
  python scripts/make_demo_cover.py            # 已存在则跳过
  python scripts/make_demo_cover.py --force    # 覆盖
  python scripts/make_demo_cover.py --size 128 --out demo/cover.png
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="生成 demo 载体图")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="demo/cover.png")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"已存在, 跳过: {out} (加 --force 覆盖)")
        return

    import cv2
    import numpy as np

    size = args.size
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    img = np.stack([
        0.5 + 0.4 * np.sin(6 * xx + 3 * yy),
        0.5 + 0.4 * np.sin(9 * yy + np.cos(4 * xx)),
        0.5 + 0.4 * np.cos(7 * (xx + yy)),
    ], axis=-1)
    rng = np.random.default_rng(11)
    img += rng.normal(0, 0.03, img.shape)
    img = np.ascontiguousarray(np.clip(img, 0, 1))

    bgr = (img * 255).astype(np.uint8)[:, :, ::-1].copy()
    cv2.circle(bgr, (size // 3, size // 2), size // 5, (50, 60, 230), -1)
    cv2.rectangle(bgr, (size // 2, size // 4), (size * 5 // 6, size * 3 // 4),
                  (220, 80, 50), -1)

    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), bgr):
        raise SystemExit(f"写文件失败: {out}")
    print(f"已生成: {out} ({size}x{size})")


if __name__ == "__main__":
    main()