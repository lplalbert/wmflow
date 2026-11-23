"""LLM 辅助层: 专用 prompt 生成 (找 bug / 调参) + 响应解析 + 补丁自动适配。

prompt 的三种形态 (都以 NetSpec 的系统化描述为底):
  1. describe_prompt   —— 静态框架描述 (积木目录/清单/形状表/约束), 直接贴任何大模型
  2. debug_prompt      —— 描述 + 报错/现象 → 问诊断与最小修改
  3. tune_prompt       —— 描述 + 指标 + 目标 → 问 JSON 补丁

闭环: parse_patch_response 提取 JSON → NetSpec.apply_patch 校验+自动重适配。
API 调用留了极简可选钩子 (call_llm), 凭据只走环境变量, 不落代码。

环境变量 (仅 call_llm 用, 可选):
  WMFLOW_LLM_BASE_URL  e.g. https://api.openai.com/v1
  WMFLOW_LLM_API_KEY   你自己的 key (不写进代码/仓库)
  WMFLOW_LLM_MODEL     e.g. gpt-4o / claude-... / 本地 vllm 模型名
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .netspec import NetSpec


def debug_prompt(spec: NetSpec, problem: str, traceback_text: str = "",
                 extra_context: str = "") -> str:
    """找 bug prompt: 现象 + (可选) traceback + 结构描述 → 诊断建议。"""
    ctx = extra_context
    if traceback_text:
        ctx += ("\n\n### 报错/日志\n```\n" + traceback_text.strip() + "\n```")
    base = spec.render_prompt(ctx)
    return "\n".join([
        base,
        "",
        "## 问题现象",
        problem,
        "",
        "## 请回答",
        "1. 最可能的根因 (逐条给依据, 指出对应积木 id)",
        "2. 最小修改建议 (给出 patch JSON, 格式同调参任务; 或指出改哪积木)",
        "3. 如涉及形状/通道问题, 明确说自动适配是否会兜住, 哪里兜不住",
    ])


def tune_prompt(spec: NetSpec, metrics: Dict[str, Any], goal: str,
                constraints: str = "", history: Optional[List[str]] = None) -> str:
    base = spec.generate_tune_prompt(metrics, goal, constraints)
    if history:
        base += "\n\n## 既往尝试 (避免重复)\n" + "\n".join(f"- {h}" for h in history)
    return base


def parse_patch_response(text: str) -> Dict[str, Any]:
    """从模型回复里提取 patch JSON。容忍 ```json 围栏与前后杂文。"""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    blob = m.group(1) if m else None
    if blob is None:
        m = re.search(r"\{[^{}]*\"patch\"[^{}]*\[.*?\][^{}]*\}", text, re.S)
        blob = m.group(0) if m else None
    if blob is None:
        raise ValueError("回复中找不到 patch JSON")
    d = json.loads(blob)
    if "patch" not in d or not isinstance(d["patch"], list):
        raise ValueError(f"JSON 缺少 patch 列表: {d}")
    return d


def apply_response(spec: NetSpec, text: str) -> NetSpec:
    """解析回复并套用补丁 (自动重适配 + build 兜底)。返回新 NetSpec。"""
    d = parse_patch_response(text)
    return spec.apply_patch(d["patch"])


# ---------------------------------------------------------------------------
# 可选: 真实 API 调用 (OpenAI 兼容 chat 接口; 凭据只走环境变量)
# ---------------------------------------------------------------------------
def call_llm(prompt: str, system: str = "", timeout: int = 120) -> str:
    """调用 chat completions。未配环境变量则报错说明用法 (不静默降级)。"""
    base = os.environ.get("WMFLOW_LLM_BASE_URL")
    key = os.environ.get("WMFLOW_LLM_API_KEY")
    model = os.environ.get("WMFLOW_LLM_MODEL")
    if not (base and key and model):
        raise RuntimeError(
            "call_llm 需要环境变量 WMFLOW_LLM_BASE_URL / WMFLOW_LLM_API_KEY / "
            "WMFLOW_LLM_MODEL; 也可不调 API, 直接把 prompt 贴任意模型。")
    import urllib.request
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "messages": msgs}).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"]
