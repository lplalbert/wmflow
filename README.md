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

## 网络积木 (NetSpec) —— 模块粒度约定

三层粒度（需求：显示模块、不显示单层、像搭积木一样换）：

| 粒度 | 可见性 | 例子 |
|---|---|---|
| 层 (Conv/BN/ReLU) | ❌ 不外露，只在模块的"层组成"描述串里显示 | `Conv3x3→BN→ReLU` |
| **模块 (积木)** | ✅ 主粒度：显示/可换/自动适配 | conv_block, res_block, se_block, msg_map, bit_head... |
| 网络 | 模块串成 encoder/decoder (NetSpec JSON) | hidden_encoder |

- **搭积木**：NetSpec = JSON 积木清单，`{"id","type","params","from"}` 逐块声明，隐式串联。
- **随意换模块**：改 `type` 或 `apply_patch([{"id":"r4","replace_type":"se_block"}])`。
- **自动适配参数**：`in_ch` 不填自动继承上游；通道不符**自动插 1×1 adapter**（记录在 `auto_adapters`）；`msg_map` 目标尺寸、`bit_head` 消息长自动注入；补丁套用后自动重算适配。
- **保真口径**：经典框架为"模块级结构对照"（层宽/深度合理化，差异在 `recipes.py` 逐条注明），不称官方等价复现。

已配配方（`wmflow/recipes.py`）：`hidden`（HiDDeN ECCV'18）、`stegastamp`（CVPR'20）、`mbrs`（ACM MM'21）——三者训练侧攻击（Denoiser/可微扰动/真 JPEG mini-batch）记在 `train_note`，不进推理积木。

## LLM 专用 prompt（找 bug / 调参 / 自动适配闭环）

每个 NetSpec 自带**系统化描述 prompt**：任务+模块目录（含层组成）+积木清单+形状解析表+约束，可直接贴任意大模型 API：

```bash
python prompt_tool.py describe hidden            # 框架系统化描述
python prompt_tool.py debug hidden --problem "BER=0.5 不降" --tb err.txt
python prompt_tool.py tune hidden --metrics '{"ber":0.21,"psnr":28.5}' --goal "BER<0.05 且 PSNR>30"
python prompt_tool.py apply hidden --from-reply reply.txt   # 模型回复→补丁→自动重适配→新 NetSpec JSON
```

- 模型回复格式约定：JSON `{"analysis":..., "patch":[{"id":..., "params":{...}} | {"id":..., "replace_type":...}], "risk":...}`。
- `apply` 闭环：解析→逐项套用→**通道/尺寸自动重适配**→build 兜底校验，非法积木/形状直接拒。
- `prompt_tool.py ask` 可直连 API（OpenAI 兼容），凭据只走 `WMFLOW_LLM_BASE_URL / API_KEY / MODEL` 环境变量，不落代码。
- 节点形态：`NetSpecPrompt` 节点（describe/debug/tune 三模式）可在图上出 prompt。

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

- [x] 14 节点注册与 schema 自省
- [x] 干净往返 BER=0（LSB 玩具模型）
- [x] 输入签名缓存（重跑全命中 / 换 seed 失效）
- [x] 类型检查拦截错线
- [x] 全链路冒烟（嵌入→combo攻击→wechat_lite→提取→BER→报告落盘）
- [x] 积木系统：13 模块目录 / 三配方 enc→dec 前向 128² / 自动插 adapter / 换积木 / LLM 补丁闭环 / debug prompt（tests/test_netspec.py 8 项）
