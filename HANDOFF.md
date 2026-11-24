# wmflow 交接简报（2026-11-23，供新会话接手）

> 一句话定位：**水印嵌入/攻击/提取/评测的节点图工作流引擎**——ComfyUI 工作流模式的
> clean-room 化（借模式不抄码，避 GPL-3.0），支持搭积木式网络组装、服务器训练、LLM 辅助调参。
> 设计依据：论文调研仓 `ComfyUI工作流模式调研_水印节点化设计_20261123.md`。

## 1. 资产地图

| 东西 | 位置 |
|---|---|
| 代码仓（远端） | `github.com/lplalbert/wmflow`（Public，main） |
| 服务器工作副本 | `lpl2025@222.20.98.60:/home/lpl2025/lpl/wmflow`（**主要开发地**） |
| 本地镜像 | `c:/Users/24976/Desktop/code/wmflow` |
| webui 服务 | `http://222.20.98.60:7860`（无鉴权，**勿映射公网**） |
| 训练产出 | 仓库下 `runs/<框架>_<时间戳>/ckpt_epN.pt` |
| 服务器数据集 | `/home/lpl2025/lpl/Datasets/`（DIV2K 900 图、document_ds 30000 图） |
| 流水账 | 论文调研仓 `handoff/status.md`（wmflow 各条目，最新在下） |

## 2. 架构速览（文件地图）

```
wmflow/
  engine.py        # clean-room 执行器: API 格式 JSON 图/拓扑+环检测/类型检查/
                   # 输入签名缓存(张量走 bytes 摘要)/种子记账/report.json
  node.py          # 节点契约 (INPUT_TYPES/RETURN_TYPES + VALIDATE_INPUTS/fingerprint)
  registry.py      # NODE_CLASS_MAPPINGS + /object_info 式自省
  types.py         # IMAGE/MESSAGE/WM_MODEL/TENSOR/METRICS... + Message/WMModel
  modules.py       # 13 个模块级积木 (conv/res/se/dense/down/up/msg_map/concat/
                   # residual_add/bit_head/adapter1x1...)——层不外露只显示层组成串
  netspec.py       # NetSpec: 搭积木 JSON/自动适配(通道继承/插 adapter/参数注入)/
                   # LLM prompt 渲染/apply_patch 补丁闭环/forward_trace 逐块追踪
  components.py    # 自动切段器(积木段=组件) + 组装器(悬空引用回落上一块)
  custom_recipes.py # 自定义配方库: 组装/混搭结果存 JSON (dec msg_len 自动对齐 + build
                   # 校验 + 原子写) / 列表 / 导出导入 / pair() 给流水线·训练用
  recipes.py       # hidden/stegastamp/mbrs/videoseal 四配方 (模块级结构对照口径,
                   # 训练侧攻击记 train_note 不进推理积木)
  llm.py           # debug/tune prompt + 模型回复 JSON 补丁解析 + 可选 API 钩子
                   # (凭据只走 WMFLOW_LLM_BASE_URL/API_KEY/MODEL 环境变量)
  train.py         # 训练环: 支持 framework='custom:<名>' (尺寸由配方锁定) + 训练前梯度自检;
                   # 可微攻击(gaussian_noise/blur/jpeg_approx(DCT+STE)/
                   # crop_resize/dropout/combo) + BCE/MSE 加权 + epoch BER + ckpt
  nodes_core.py / nodes_models.py / nodes_netspec.py   # 14 个图节点
  webui.py         # 零依赖 http.server 的 web 服务 (21 个 /api/*)
webui_index.html   # 单页前端: 节点图画布(SVG 贝塞尔连线/类型彩色端口/节点内嵌
                   # 图像·比特格·热图·指标预览) 四视图: 流水线/积木链(组件库+组装台+
                   # 我的配方)/训练 + 通用对话框(存配方/导出/导入)
prompt_tool.py     # LLM prompt CLI (describe/debug/tune/apply/ask)
scripts/webui.sh   # 服务管理 (pidfile /tmp/wmflow_webui.pid, 避 pkill 自匹配)
tests/             # test_smoke(6)/test_netspec(9)/test_components(6)/
                   # test_custom_recipes(12)/test_train_custom(8) 五套
workflows/         # 示例图 JSON
custom_recipes/    # 运行时资产: 自定义配方 JSON (gitignore, 不随仓走)
```

## 3. 已完成（按时间序）

1. **ComfyUI 调研** → 设计文档（工作区根，含 15 条可借鉴清单、GPL 红线、
   "层→模块→网络"粒度约定、LLM prompt 设计、训练不进图裁决）。
2. **P0 引擎**：14 节点 + 执行器 + 双端冒烟全过。
3. **P1-1 积木系统**：13 积木 + NetSpec（自动适配）+ 四经典框架配方 + LLM prompt
   三模式 + 补丁闭环。
4. **webui v1→v2**：网页工作台，ComfyUI 式节点图，输入输出直视（图像 PNG/比特格
   错位标红/特征图热图/指标 KV），上传载体图，服务器 GPU 真跑。
5. **训练能力**：训练面板（服务器路径浏览含递归计数/图片数量/尺寸/噪声多选/
   epochs·batch·lr·strength·损失权重），后台训练+进度+loss/BER 表+逐 epoch 预览+
   ckpt 一键回灌流水线。**服务器 GPU 实测通过**（DIV2K×stegastamp×RTX 3090）。
6. **组件库+组装台**：按框架分组的积木段货架（每框架 enc 7 段/dec 3 段 +
   13 自由单积木），拣段→排序→构建校验→追踪→作 enc/dec 跑流水线；跨框架混段
   实测通过（hidden.fusion + videoseal.res + hidden.out）。
7. **自定义配方**：组装/混搭结果存盘 → 流水线/节点按名直用 → 导出/导入 JSON。
   存盘前 dec bit_head msg_len 自动对齐 enc 并 build 校验（不过不落盘），原子写
   `custom_recipes/<名>.json`（gitignore）；两侧来源可各选（组装台当前 / 内置整段 /
   已存配方），故"混搭 enc + 内置 dec"也能一条指令落库。组件库「我的配方」货架可
   载入 enc/dec（拆回积木段续改）/ ▶流水线 / 导出 / 删除 / 导入 JSON。
8. **自定义配方进训练**（最新）：训练面板框架下拉可选 `custom:<名>`，配方的 size/msg_len
   锁定（表单冲突尺寸被忽略并记日志）；开训前跑一次 enc→dec 前向+反向做**梯度自检**
   （含"全零梯度"判定）；ckpt 存结构快照 + `recipe` 字段。流水线选 ckpt 后按快照渲染，
   配方被删/被覆盖仍可回放（有提示）；顺带修了"权重从未真正载入却标 trained"的老 bug，
   以及 BER 判决阈值 `logit>0.5` 的错误口径（应为 `logit>0`）。

## 4. 操作手册

```bash
# 服务 (服务器上)
cd /home/lpl2025/lpl/wmflow
bash scripts/webui.sh start|stop|restart|status 7860
# 解释器自动挑 (WMFLOW_PY 优先 → /data/miniconda_dir/envs/lpl/bin/python → 系统
# python3); 系统 python3 没 numpy, 显式 WMFLOW_PY 仍可强制

# 部署流 (改完代码)
# 本地: git add -A && git commit && git push origin main
# 服务器: git fetch origin && git merge --ff-only origin/main
#         bash scripts/webui.sh restart 7860   # fetch 与 merge 分开跑!

# 测试
python tests/test_smoke.py && python tests/test_netspec.py && python tests/test_components.py && python tests/test_custom_recipes.py && python tests/test_train_custom.py
```

- 自定义配方用法：组装台搭好 →「💾 存为配方」（两侧来源可各选）→ 流水线「框架」下拉
  的「我的配方」组里直接选 → ▶ 运行；组件库「我的配方」可导出 `.json` / 导入。
  节点内引用写 `framework="custom:<配方名>"`。配方目录 `custom_recipes/` 不进 git。

- **服务器 git 已配 HTTPS fetch + SSH push**（deploy key
  `~/.ssh/id_ed25519_github_wmflow`）——SSH fetch 会挂起，勿改回。
- 训练用法：webui「训练」页签选数据集目录（可逐级浏览）→ 勾噪声层 → 参数 →
  开始训练；ckpt 自动落 runs/，训练完点「在流水线中使用」。
- LLM 闭环：右栏生成 prompt → 贴给任意大模型 → 回复贴回「贴回复·套补丁」。

## 5. 已知坑（合集，改代码前先看）

1. **NetSpec 补丁回显双源**：`__init__` 里 modules 与 spec 必须同源（deepcopy 后取），
   否则补丁改了不回显——已有回归断言守着。
2. **`type("M",(),{...})()` 类字典挂函数 = 绑定方法吃 self**——临时对象传 callable
   用实例属性。
3. **pkill/pgrep 自匹配**会杀掉 ssh 会话/自己——用 pidfile（scripts/webui.sh）或
   `pgrep -x` 精确名。
4. **git fetch 与 merge 同条 ssh 命令竞态**（merge 先报"已最新"）——分开跑。
5. **DIV2K 类两级目录**要递归扫图（load/scan 已做递归回退）。
6. **跨框架混搭消息长不齐**（30↔256bit）：dec bit_head 自动对齐 enc + BER 短边
   防御比较，缺一 numpy 广播就炸。
7. **residual_add 形状**取 cover 形（x+cover 广播）；切段后其跨段 from 悬空 →
   组装器"回落上一块"语义。
8. **中文文件写入**一律 Write 工具或 Python `open(encoding='utf-8')` + 事后数 `?`
   核验，勿走 shell 管道（GBK 坑）。
9. Windows GBK 控制台打不出 `²` 等字符会炸 print——测试文案用 ASCII。
10. webui **无鉴权**，别把 7860 映射公网；要公网先加 token。
11. **POST 体编码**：浏览器发 UTF-8，但 Windows 下 curl/PowerShell 手搓请求可能是 GBK
    ——`do_POST` 先 UTF-8 解、失败再按 gb18030 兜一次，否则中文配方名/备注直接 400。
    自己写压测脚本时也别用 `ConvertTo-Json` 默认编码直接当 body，要转 UTF-8 字节。
12. **自定义配方是运行时资产**：`custom_recipes/` 已 gitignore，服务器上攒的配方不会
    随 push 走；要跨端搬用「导出 JSON」，别指望 git。
13. **噪声层输出必须 C 连续**：通道翻转 `arr[:, :, ::-1]` 是负 stride 视图，直接
    `torch.from_numpy` 必报 "At least one stride ... is negative"。`_jpg` 已
    ascontiguousarray 修掉（单独 layer=jpeg 曾必崩，combo 因 np.clip 落了新数组才没
    暴露）；webui/nodes_netspec 入口也统一兜底，新增层（含 scw 物理层）照此办理。
    test_smoke 里有 5 层连续性断言守着。
14. **服务器 git fetch 偶发 `GnuTLS recv error (-110)`**（HTTPS，网络抖动）：重试即通；
    但重试失败时 `merge` 会误报"已经是最新的"——先 `git rev-parse origin/main` 核对 sha
    再 merge。
15. **`encodeURIComponent` 不编码单引号**：`onclick="f('${encodeURIComponent(p)}')"` 仍能被
    文件名 `x');alert(1);//` 逃逸；凡把**外部字符串**（目录名/路径/用户标题）塞进
    innerHTML 或内联事件，一律用 `textContent` + `addEventListener` 闭包，别拼串。
    配方名因为过了 slugify（只剩 `\w` 与中文）才幸免。
16. **前端异步竞态会写回旧 spec**：构建/补丁请求在途时改台子，旧响应回来会覆盖缓存，
    接着被存成配方。做法：台子状态计数器 `benchVer` + 缓存标记 `benchSideVer`，
    响应后比对版本/段快照/参数三元组，丢弃过期结果；存盘前再核对一次版本。
17. **BER 判决边界是 logit>0，不是 >0.5**：`bit_head` 是 `Linear` 直出 logits，训练用
    `BCEWithLogits`，所以判决阈值是 0。曾一律用 `>0.5`（等价于把 logit 当概率），
    导致 BER 系统性偏高/失真。已统一改 `logit > 0`（train/webui/nodes_netspec），
    改动后 BER 数值与旧记录不可直接对比。`nodes_models` 里 `sigmoid>0.5` 等价写法保留。
18. **ckpt 必须绑结构快照，且回放要用同一实例**：训练出的 ckpt 存 `spec_enc/spec_dec`
    （结构快照）+ 权重，回放时**以快照为准**，配方被删/被覆盖也能按原结构跑（并提示）。
    另一个坑：`eb.as_module().load_state_dict(...)` 后再 `eb.as_module().eval()` 是
    **两个不同实例**，权重从未生效却标 trained=True——模块只建一次、权重载入同一实例。
    现在 `_resolve_demo_model()` 统一做这件事，测试用**逐元素比对嵌入图**抓这类回归
    （只比 BER 抓不住）。
19. **训练前必须做梯度自检**：自定义/混搭结构可能出现某侧是死分支（如 `strength=0`
    时 encoder 输出被完全丢弃 → 梯度全 0）。只查 `None`/`inf` 会漏，必须查"全零梯度"，
    否则白训数十轮才发现。

## 6. 未完成 / 下一步（按优先级候选）

- 四配方**真训练权重**（现在都是 untrained 结构演示，lsb_toy 才是真嵌入）
- 损失曲线图、多 run 对比、val 集划分
- 7860 **访问控制**（无鉴权，见 §5 #10）；建议反向代理单用户认证 + 限公网入口
- scw 物理层（pimog/esmn，numpy 不可微）进训练的 **STE 直通方案**待设计
- 服务守护化（systemd）；段级 trace 缩略图；画布拖拽/缩放
- 自定义配方的**段级复用**（现在整网载入后拆回组装台；还没做到"把别人配方里的
  某一段拣进我的台"）
- LICENSE 选型（建议 MIT，未放文件）；**旧 fine-grained PAT 应吊销**（曾明文出现在
  对话里，用户已知）

## 7. 口径纪律（论文相关，勿破）

- 四配方 = **模块级结构对照**，不称官方等价复现（层宽/深度差异在 recipes.py 注明）
- 训练侧攻击（HiDDeN Denoiser/MBRS 真 JPEG mini-batch 等）记 train_note 不进推理积木
- 引擎 clean-room，不抄 ComfyUI 代码（GPL-3.0）
- scw_research（论文线代码）经 PYTHONPATH 适配，不复制进本仓；E2 预注册口径以
  论文调研仓为准，wmflow 是工程组织层不动科学路线
