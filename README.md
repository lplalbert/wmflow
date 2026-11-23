# wmflow

水印嵌入 / 攻击 / 提取 / 评测的**节点图工作流引擎**。工作流 = 一份 JSON，实验可复现、可组合、可缓存重放。

> 设计取自 ComfyUI 的工作流模式（类型化节点契约 + 可序列化 DAG + 惰性执行缓存），
> **clean-room 自研实现，未使用 ComfyUI 任何代码**（其为 GPL-3.0；本仓不受传染）。
> 调研与设计文档见论文调研仓 `ComfyUI工作流模式调研_水印节点化设计_20261123.md`。

## 核心概念

- **节点** = 声明式端口契约的 Python 类（`INPUT_TYPES` / `RETURN_TYPES` / 执行方法 + 可选 `VALIDATE_INPUTS` 预检 / `fingerprint_inputs` 缓存指纹）。
- **工作流** = API 格式 JSON：`{节点id: {class_type, inputs}}`，inputs 里 `[节点id, 槽]` 是连线，其余是字面量。
- **引擎**：拓扑执行 + 环检测 + **类型检查**（错线直接拒） + **输入签名缓存**（未变子图不重算） + 种子记账 + report.json 落盘。

## 类型系统

`IMAGE` (H,W,3 f32 RGB[0,1]) / `MESSAGE` (bits L 或 B×L) / `WM_MODEL` (框架句柄) / `PARAMS` / `METRICS` / `REPORT` + 标量 INT/FLOAT/STRING。

## 节点（12 核心节点）

| 节点 | 类别 | 作用 |
|---|---|---|
| CoverLoader | wm/io | 载体加载（fingerprint=文件 mtime，文件变了才重跑） |
| MessageGen | wm/io | 显式 seed 的消息比特 |
| SaveReport | wm/io | METRICS 聚合落盘 |
| WMModelLoader | wm/model | 框架适配：`lsb_toy`（冒烟）/ `s2r`（权重加载） |
| Embed | wm/model | model+cover+message+strength → 水印图 |
| Decode | wm/model | model+image → message |
| NoiseLayer | wm/attack | 可组合退化层（内置 5 层；scw 在 PYTHONPATH 时自动扩到 LAYERS 全表） |
| TranscodeChain | wm/attack | 转码链预设 none/jpeg/420/resize/wechat_lite |
| BERCompare | wm/metrics | BER/bit_errors |
| ImageProfile | wm/metrics | PSNR/对比度（scw 在位时追加 moiré/glare 族） |
| Rectify | wm/post | （P0 占位，P1 接 scw 双轨） |
| PhysicsChannel | wm/attack | （P0 占位，P1 接 scw 可微物理信道） |

框架适配路线：`lsb_toy` ✅ → `s2r` ✅(接口) → `stegastamp` / `hidden` / `mbrs` / `videoseal`（P1，适配器接口已留好，服务器上各框架仓在 `/home/lpl2025/lpl/`）。

## 快速开始

```bash
python tests/test_smoke.py                            # 冒烟：机制 + 干净往返 BER=0
python run.py workflows/smoke_ablation.json --out out/smoke
python run.py workflows/smoke_ablation.json --set 5.layer=noise --set 6.preset=jpeg75
python run.py workflows/smoke_ablation.json --validate-only
```

扫参 = 外层 driver 用 `--set` 批量改写节点输入；图本身只描述单次执行（与冻结协议对齐：每张图一份 report.json，含全部种子与逐节点缓存键）。

## 与 scw_research 的关系

`wmflow` 是工程组织层；`scw_research/scw` 的噪声层/物理信道/度量/校正通过 **PYTHONPATH 适配**（import，不复制代码）：
`PYTHONPATH=/path/to/scw_research python run.py ...`。
论文实验口径仍以 scw_research 的冻结协议（E2 预注册）为准。

## 验证状态

- [x] 12 节点注册与 schema 自省
- [x] 干净往返 BER=0（LSB 玩具模型）
- [x] 输入签名缓存（重跑全命中 / 换 seed 失效）
- [x] 类型检查拦截错线
- [x] 全链路冒烟（嵌入→combo攻击→wechat_lite→提取→BER→报告落盘）
