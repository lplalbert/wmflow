"""wmflow 执行引擎 (clean-room 实现)。

工作流 JSON (API 格式, 节点即字典):
{
  "meta": {"name": "...", "seed": 42},
  "nodes": {
    "1": {"class_type": "CoverLoader", "inputs": {"path": "a.png"}},
    "5": {"class_type": "Embed", "inputs": {"model": ["3", 0], "cover": ["1", 0], ...}}
  },
  "outputs": ["12"]          # 可选; 缺省跑全部节点
}
连线约定: inputs 值为 [节点id, 输出槽] 是连线, 否则是字面量。

缓存: 键 = sha256(class_type + 字面量输入 + 上游缓存键 + fingerprint_inputs)。
与 ComfyUI 的 CacheKeySetInputSignature 同思路——图中未变化的子图不重算。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .node import collect_input_specs
from .registry import get_node


class WorkflowError(RuntimeError):
    pass


def _is_link(v: Any) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and isinstance(v[0], (str, int)))


def _stable_hash(obj: Any) -> str:
    def conv(o):
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o
        if isinstance(o, (list, tuple)):
            return [conv(x) for x in o]
        if isinstance(o, dict):
            return {k: conv(o[k]) for k in sorted(o)}
        if hasattr(o, "tolist"):  # numpy
            return {"__ndarray__": conv(o.tolist())}
        return {"__repr__": repr(o)}
    blob = json.dumps(conv(obj), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class NodeRun:
    node_id: str
    class_type: str
    outputs: Tuple[Any, ...]
    cache_key: str
    seconds: float
    from_cache: bool


@dataclass
class RunResult:
    outputs: Dict[str, Tuple[Any, ...]]
    runs: List[NodeRun] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def cache_hits(self) -> int:
        return sum(1 for r in self.runs if r.from_cache)


class Executor:
    """一次运行一个 Executor 实例; cache 可跨实例复用 (扫参场景)。"""

    def __init__(self, cache: Optional[Dict[str, Tuple[Any, ...]]] = None,
                 collect_metrics: bool = True):
        self.cache = cache if cache is not None else {}
        self.collect_metrics = collect_metrics

    # ---------- 加载与校验 ----------
    @staticmethod
    def load(path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            wf = json.load(f)
        if "nodes" not in wf:
            raise WorkflowError("工作流缺少 'nodes' 字段")
        return wf

    @staticmethod
    def validate(wf: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        nodes = wf["nodes"]
        for nid, nd in nodes.items():
            try:
                cls = get_node(nd["class_type"])
            except KeyError as e:
                errors.append(f"节点 {nid}: {e}")
                continue
            specs = collect_input_specs(cls)
            for pname, pval in (nd.get("inputs") or {}).items():
                if pname not in specs:
                    errors.append(f"节点 {nid} ({nd['class_type']}): 未知输入 '{pname}'")
                    continue
                if _is_link(pval):
                    up_id, slot = str(pval[0]), int(pval[1])
                    if up_id not in nodes:
                        errors.append(f"节点 {nid}: 连线指向不存在的节点 '{up_id}'")
                        continue
                    try:
                        up_cls = get_node(nodes[up_id]["class_type"])
                    except KeyError:
                        continue  # 上游已报
                    ups = getattr(up_cls, "RETURN_TYPES", ())
                    if slot >= len(ups):
                        errors.append(f"节点 {nid}: 上游 {up_id} 无输出槽 {slot}")
                    elif ups[slot] != specs[pname]["type"]:
                        errors.append(
                            f"节点 {nid}: 类型不匹配 —— 输入 '{pname}' 要 "
                            f"{specs[pname]['type']}, 上游 {up_id}[{slot}] 给 {ups[slot]}")
            for pname, spec in specs.items():
                if spec["required"] and pname not in (nd.get("inputs") or {}) \
                        and spec["default"] is None:
                    errors.append(f"节点 {nid} ({nd['class_type']}): 缺必需输入 '{pname}'")
        # 环检测
        try:
            Executor._topo(wf)
        except WorkflowError as e:
            errors.append(str(e))
        return errors

    @staticmethod
    def _topo(wf: Dict[str, Any]) -> List[str]:
        nodes = wf["nodes"]
        order: List[str] = []
        state: Dict[str, int] = {}  # 0=未 1=访问中 2=完成

        def visit(nid: str, chain: Tuple[str, ...]):
            st = state.get(nid, 0)
            if st == 2:
                return
            if st == 1:
                raise WorkflowError(f"图中有环: {' -> '.join(chain + (nid,))}")
            state[nid] = 1
            for pval in (nodes[nid].get("inputs") or {}).values():
                if _is_link(pval):
                    visit(str(pval[0]), chain + (nid,))
            state[nid] = 2
            order.append(nid)

        for nid in nodes:
            visit(nid, ())
        return order

    # ---------- 执行 ----------
    def run(self, wf: Dict[str, Any],
            only_outputs: Optional[List[str]] = None) -> RunResult:
        nodes = wf["nodes"]
        result = RunResult(outputs={})
        needed = set(only_outputs or nodes.keys())

        # 反向可达集裁剪: 只算输出节点的祖先
        reach: set = set()

        def mark(nid: str):
            if nid in reach:
                return
            reach.add(nid)
            for pval in (nodes[nid].get("inputs") or {}).values():
                if _is_link(pval):
                    mark(str(pval[0]))

        for nid in needed:
            if nid not in nodes:
                raise WorkflowError(f"输出节点不存在: {nid}")
            mark(nid)

        for nid in self._topo(wf):
            if nid not in reach:
                continue
            nd = nodes[nid]
            cls = get_node(nd["class_type"])
            specs = collect_input_specs(cls)

            resolved: Dict[str, Any] = {}
            literals: Dict[str, Any] = {}
            link_keys: List[str] = []
            for pname, spec in specs.items():
                raw = (nd.get("inputs") or {}).get(pname, None)
                if raw is None:
                    if not spec["required"] and spec["default"] is not None:
                        raw = spec["default"]
                    elif not spec["required"]:
                        continue
                if _is_link(raw):
                    up_id, slot = str(raw[0]), int(raw[1])
                    up = result.outputs[up_id]
                    resolved[pname] = up[slot]
                    link_keys.append(f"{pname}<-{up_id}#{slot}:{self._key_of(up_id, result)}")
                else:
                    resolved[pname] = raw
                    literals[pname] = raw

            errs = cls.VALIDATE_INPUTS(**{k: v for k, v in resolved.items()
                                          if not hasattr(v, "shape")})
            if errs:
                raise WorkflowError(f"节点 {nid} ({nd['class_type']}) 预检失败: " + "; ".join(errs))

            fp = cls.fingerprint_inputs(**resolved)
            key = _stable_hash({
                "class": nd["class_type"], "literals": literals,
                "links": sorted(link_keys), "fingerprint": fp,
            })

            t0 = time.perf_counter()
            if key in self.cache:
                outs = self.cache[key]
                result.runs.append(NodeRun(nid, nd["class_type"], outs, key,
                                           time.perf_counter() - t0, True))
            else:
                fn = getattr(cls(), cls.FUNCTION)
                outs = fn(**resolved)
                if not isinstance(outs, tuple):
                    outs = (outs,)
                self.cache[key] = outs
                result.runs.append(NodeRun(nid, nd["class_type"], outs, key,
                                           time.perf_counter() - t0, False))
            result.outputs[nid] = outs

            # METRICS 自动聚合进报告
            if self.collect_metrics:
                for i, rtype in enumerate(getattr(cls, "RETURN_TYPES", ())):
                    if rtype == "METRICS" and isinstance(outs[i], dict):
                        result.metrics[f"{nid}:{nd['class_type']}"] = outs[i]
        return result

    def _key_of(self, nid: str, result: RunResult) -> str:
        for r in reversed(result.runs):
            if r.node_id == nid:
                return r.cache_key
        return "?"


def save_report(result: RunResult, wf: Dict[str, Any], out_dir) -> Path:
    """运行清单落盘: report.json (度量 + 逐节点缓存键/耗时 + 全部种子)。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = {}
    for nid, nd in wf["nodes"].items():
        for k, v in (nd.get("inputs") or {}).items():
            if k.endswith("seed") and not _is_link(v):
                seeds[f"{nid}:{nd['class_type']}.{k}"] = v
    report = {
        "meta": wf.get("meta", {}),
        "metrics": result.metrics,
        "seeds": seeds,
        "nodes": [{"node_id": r.node_id, "class_type": r.class_type,
                   "cache_key": r.cache_key, "seconds": round(r.seconds, 4),
                   "from_cache": r.from_cache} for r in result.runs],
    }
    path = out / "report.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    return path
