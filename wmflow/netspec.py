"""NetSpec: 搭积木式网络描述 + 自动适配 + LLM 专用 prompt 生成。

一份 NetSpec = 一个 encoder 或 decoder 的模块积木清单, JSON 可序列化:
{
  "name": "hidden_encoder",
  "input": {"image": [3,128,128], "message": [30]},
  "modules": [
    {"id": "m1", "type": "msg_map", "params": {"out_ch": 1}},
    {"id": "c1", "type": "concat", "from": ["image", "m1"]},
    {"id": "e1", "type": "conv_block", "params": {"out_ch": 32}},
    ...
    {"id": "out", "type": "residual_add", "from": ["e9", "image"]}
  ],
  "output": "out"
}

自动适配 (build 时逐模块解析):
  - params 里没写的 in_ch/ch → 继承上游 out_ch
  - 上游 out_ch ≠ 本模块显式 in_ch → 自动插 adapter1x1 (记录 auto_adapters)
  - msg_map.target_hw / bit_head.msg_len 自动注入
  - 全程形状推理, 不符即报错并给最近合法建议

LLM prompt (render_prompt):
  系统化描述本网络: 模块目录+积木清单+形状表+参数表+已知约束,
  直接可贴给大模型 API 做 (a) 找 bug (b) 调参建议 (c) 换模块建议;
  generate_tune_prompt 额外附上指标与期望, 让模型输出 JSON 补丁,
  apply_patch 校验后套用 (含自动重适配) —— 参数自动适配闭环。
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .modules import MODULES, Shape, register_module  # noqa: F401 (确保注册)

_MISSING = object()


@dataclass
class BuiltModule:
    id: str
    type: str
    params: Dict[str, Any]
    in_shapes: List[Shape]
    out_shapes: List[Shape]
    auto_adapters: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BuildResult:
    name: str
    modules: List[BuiltModule]           # 已解析参数与形状的积木序列
    callable_chain: List[Tuple[str, Any, List[str]]]  # (id, callable, from_ids)
    output_id: str
    auto_adapters: List[Dict[str, Any]]

    @property
    def out_shape(self) -> Shape:
        return self.modules[-1].out_shapes[0]

    def forward(self, inputs: Dict[str, Any]) -> Any:
        """按 chain 求值。inputs: {输入名: 张量}。"""
        env = dict(inputs)
        for mid, fn, frm in self.callable_chain:
            args = [env[f] for f in frm]
            env[mid] = fn(args[0] if len(args) == 1 else args)
        return env[self.output_id]

    def forward_trace(self, inputs: Dict[str, Any]):
        """同 forward, 但返回全部中间张量 (逐积木输入输出可视化用)。"""
        env = dict(inputs)
        for mid, fn, frm in self.callable_chain:
            args = [env[f] for f in frm]
            env[mid] = fn(args[0] if len(args) == 1 else args)
        return env[self.output_id], env

    def as_module(self):
        """包装成单个 nn.Module (可进 torch 训练/推理)。"""
        import torch.nn as nn
        chain = self.callable_chain
        out_id = self.output_id

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleDict(
                    {mid: fn for mid, fn, _ in chain})

            def forward(self, inputs):
                env = dict(inputs)
                for mid, fn, frm in chain:
                    args = [env[f] for f in frm]
                    env[mid] = self.blocks[mid](args[0] if len(args) == 1 else args)
                return env[out_id]

        return _Net()


class NetSpecError(ValueError):
    pass


class NetSpec:
    def __init__(self, spec: Dict[str, Any]):
        self.spec = copy.deepcopy(spec)
        self.name = self.spec.get("name", "net")
        self.inputs: Dict[str, Shape] = {k: tuple(v)
                                         for k, v in self.spec["input"].items()}
        # 注意: modules 必须取自 self.spec (deepcopy 之后), 否则补丁改的与回显的不同源
        self.modules: List[Dict[str, Any]] = self.spec["modules"]
        self.output: str = self.spec.get("output", self.modules[-1]["id"])
        self._by_id = {m["id"]: m for m in self.modules}
        if len(self._by_id) != len(self.modules):
            raise NetSpecError("模块 id 重复")

    # ------------------------------------------------------------------ build
    def build(self) -> BuildResult:
        """解析自动适配并搭出可调用链。返回 BuildResult (含逐模块形状表)。"""
        shapes: Dict[str, List[Shape]] = {k: [v] for k, v in self.inputs.items()}
        built: List[BuiltModule] = []
        chain: List[Tuple[str, Any, List[str]]] = []
        all_adapters: List[Dict[str, Any]] = []

        prev: Optional[str] = None
        for m in self.modules:
            mid, mtype = m["id"], m["type"]
            if mtype not in MODULES:
                raise NetSpecError(
                    f"模块 {mid}: 未知类型 '{mtype}', 可用: {sorted(MODULES)}")
            md = MODULES[mtype]
            frm: Optional[List[str]] = m.get("from")
            if not frm:
                # 隐式串联: 单输入模块默认接上一个模块的输出
                frm = [prev] if prev is not None else [next(iter(self.inputs))]
            if md.n_inputs == 2 and len(frm) != 2:
                raise NetSpecError(f"模块 {mid} ({mtype}) 需要恰好 2 个输入")
            if md.n_inputs == 1 and len(frm) != 1:
                raise NetSpecError(f"模块 {mid} ({mtype}) 需要恰好 1 个输入")

            in_lists: List[List[Shape]] = []
            for src in frm:
                if src in shapes:
                    in_lists.append(shapes[src])
                elif src in self.inputs:
                    in_lists.append([self.inputs[src]])
                else:
                    raise NetSpecError(f"模块 {mid}: 输入 '{src}' 未定义")
            in_shapes = [lst[0] for lst in in_lists]

            params = dict(m.get("params") or {})
            params, adapters = self._adapt(mid, mtype, md, in_shapes, params)
            # 自动适配: 通道不符处真正插入 adapter1x1 前置块
            for ad in adapters:
                aid = f"{mid}__adapt{ad['input_index']}"
                a_fn, a_shapes = MODULES["adapter1x1"].build(
                    [in_shapes[ad["input_index"]]],
                    in_ch=ad["from_ch"], out_ch=ad["to_ch"])
                chain.append((aid, a_fn, [frm[ad["input_index"]]]))
                shapes[aid] = [tuple(s) for s in a_shapes]
                built.append(BuiltModule(aid, "adapter1x1",
                                         {"in_ch": ad["from_ch"],
                                          "out_ch": ad["to_ch"]},
                                         [in_shapes[ad["input_index"]]],
                                         [tuple(s) for s in a_shapes], []))
                frm[ad["input_index"]] = aid
                in_shapes[ad["input_index"]] = tuple(a_shapes[0])
                ad["adapter_id"] = aid
            # msg_map 自动注入 target_hw (取同 'from' 组里含 H,W 的输入)
            if mtype == "msg_map" and params.get("target_hw") is None:
                spatial = [s for s in in_shapes if len(s) == 3]
                ref = next((s for s in self._all_shapes(shapes) if len(s) == 3), None)
                params["target_hw"] = (ref[1], ref[2]) if ref else (128, 128)
            if mtype == "bit_head" and params.get("msg_len") is None:
                ref = next((s for s in self._all_shapes(shapes) if len(s) == 1), None)
                if ref:
                    params["msg_len"] = ref[0]

            fn, out_shapes = md.build(in_shapes, **params)
            built.append(BuiltModule(mid, mtype, params, in_shapes,
                                     [tuple(s) for s in out_shapes], adapters))
            chain.append((mid, fn, frm))
            shapes[mid] = [tuple(s) for s in out_shapes]
            all_adapters.extend(adapters)
            prev = mid

        if self.output not in shapes:
            raise NetSpecError(f"output '{self.output}' 不是已构建模块")
        return BuildResult(self.name, built, chain, self.output, all_adapters)

    @staticmethod
    def _all_shapes(shapes: Dict[str, List[Shape]]):
        for lst in shapes.values():
            yield from lst

    def _adapt(self, mid, mtype, md, in_shapes, params):
        """通道自动继承/自动插 adapter。返回 (params, adapters)。"""
        adapters = []
        for i, s in enumerate(in_shapes):
            if len(s) != 3:
                continue
            want = params.get("in_ch") or params.get("ch")
            if mtype == "concat":
                continue  # concat 自身合并, 不适配
            if want is None:
                if "in_ch" in md.params:
                    params["in_ch"] = s[0]
                elif "ch" in md.params:
                    params["ch"] = s[0]
            elif want != s[0]:
                adapters.append({"for_module": mid, "input_index": i,
                                 "from_ch": s[0], "to_ch": want})
        return params, adapters

    # ------------------------------------------------------------------ LLM prompt
    def render_prompt(self, extra_context: str = "") -> str:
        """系统化框架描述 prompt (可直接贴大模型 API)。"""
        try:
            built = self.build()
            build_ok, build_err = True, ""
        except Exception as e:
            built, build_ok, build_err = None, False, str(e)

        L: List[str] = []
        L.append("# wmflow 网络积木图 (NetSpec) —— 系统化描述")
        L.append("")
        L.append("## 1. 任务")
        L.append("这是一段水印网络 (encoder 或 decoder) 的积木式结构描述。"
                 "模块是若干网络层组成的积木 (内部层构成见模块目录), 单层不外露。"
                 "请基于此回答结构/调试/调参问题。")
        L.append("")
        L.append("## 2. 网络标识与输入")
        L.append(f"- name: `{self.name}`")
        for k, v in self.inputs.items():
            L.append(f"- 输入 `{k}`: shape {v}")
        L.append(f"- 输出节点: `{self.output}`")
        L.append("")
        L.append("## 3. 模块目录 (全部可用积木)")
        for name in sorted(MODULES):
            L.append(MODULES[name].describe())
        L.append("")
        L.append("## 4. 积木清单 (本网络实际结构)")
        L.append("| # | id | 模块 | from | 显式参数 |")
        L.append("|---|----|------|------|----------|")
        for i, m in enumerate(self.modules):
            frm = ",".join(m.get("from") or ["<first>"])
            ps = json.dumps(m.get("params") or {}, ensure_ascii=False)
            L.append(f"| {i} | {m['id']} | {m['type']} | {frm} | {ps} |")
        L.append("")
        if build_ok:
            L.append("## 5. 形状与自动适配解析表")
            L.append("| id | 模块 | in_shapes | out_shapes | 解析后参数 |")
            L.append("|----|------|-----------|------------|------------|")
            for b in built.modules:
                L.append(f"| {b.id} | {b.type} | {b.in_shapes} | {b.out_shapes} "
                         f"| {json.dumps(b.params, default=str)} |")
            if built.auto_adapters:
                L.append("")
                L.append(f"自动插入的 1x1 适配器: "
                         f"`{json.dumps(built.auto_adapters, default=str)}`")
        else:
            L.append("## 5. 形状解析失败 (这本身就是要找的 bug)")
            L.append(f"```{build_err}```")
        L.append("")
        L.append("## 6. 约定与已知约束")
        L.append("- 图像张量 (B,C,H,W) float32 [0,1]; 消息 (B,msg_len) {0,1}")
        L.append("- encoder 输出经 residual_add 与 cover 相加并 clamp(0,1)")
        L.append("- decoder 输出 logits (B,msg_len), 判决 bit = logit>0 (bit_head 出 logits)")
        L.append("- 换积木 = 改 modules[].type, 通道会自动适配; 空间尺寸不自动, 须自洽")
        if extra_context:
            L.append("")
            L.append("## 7. 附加上下文")
            L.append(extra_context)
        return "\n".join(L)

    def generate_tune_prompt(self, metrics: Dict[str, Any],
                             goal: str = "降低 BER 同时保持 PSNR",
                             constraints: str = "") -> str:
        """调参 prompt: 描述+当前指标+目标 → 期望模型输出 JSON 补丁。"""
        base = self.render_prompt()
        L = [base, "", "## 调参任务", f"- 目标: {goal}",
             f"- 当前指标: {json.dumps(metrics, ensure_ascii=False)}"]
        if constraints:
            L.append(f"- 约束: {constraints}")
        L += ["", "## 输出格式 (只输出 JSON, 不要其它)",
              '{"analysis": "...", "patch": [',
              '  {"id": "模块id", "params": {"键": 值}},',
              '  {"id": "模块id", "replace_type": "新积木类型", "params": {}}',
              "], \"risk\": \"...\"}",
              "patch 逐项套用 (参数自动重适配); 也可用 replace_type 换积木。"]
        return "\n".join(L)

    # ------------------------------------------------------------------ 补丁
    def apply_patch(self, patch: List[Dict[str, Any]]) -> "NetSpec":
        """套用 LLM/人工补丁, 返回新 NetSpec (含自动重适配, validate 兜底)。"""
        new = NetSpec(copy.deepcopy(self.spec))
        for item in patch:
            mid = item.get("id")
            if mid not in new._by_id:
                raise NetSpecError(f"patch: 未知模块 id '{mid}'")
            mod = new._by_id[mid]
            if "replace_type" in item:
                t = item["replace_type"]
                if t not in MODULES:
                    raise NetSpecError(f"patch: 未知积木 '{t}'")
                mod["type"] = t
            if "params" in item:
                mod.setdefault("params", {}).update(item["params"])
        # 兜底: 必须还能 build (自动适配在此重算)
        new.build()
        return new

    def to_json(self) -> str:
        return json.dumps(self.spec, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, s: str) -> "NetSpec":
        return cls(json.loads(s))
