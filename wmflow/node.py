"""节点契约 (clean-room, 思想借自 ComfyUI 的声明式端口契约; 未使用其代码)。

一个节点 = 一个类:
  INPUT_TYPES()  -> {"required": {参数名: (类型, 选项dict)}, "optional": {...}}
  RETURN_TYPES  -> 元组 (输出槽类型名...)
  FUNCTION      -> 执行方法名 (默认 "execute")
  可选钩子:
    VALIDATE_INPUTS(**literals) -> [错误字符串...]   出数前预检
    fingerprint_inputs(**inputs) -> 可哈希对象        缓存指纹 (文件内容变了才重跑)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


class Node:
    CATEGORY = "wm"
    RETURN_TYPES: Tuple[str, ...] = ()
    RETURN_NAMES: Tuple[str, ...] = ()
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, tuple]]:
        return {"required": {}}

    # ---- 可选钩子 ----
    @classmethod
    def VALIDATE_INPUTS(cls, **literals) -> List[str]:
        return []

    @classmethod
    def fingerprint_inputs(cls, **inputs) -> Any:
        """默认 None = 只按输入签名缓存; 覆盖后参与缓存键。"""
        return None


def collect_input_specs(cls) -> Dict[str, Dict[str, Any]]:
    """展开 required+optional 的 {参数名: {type, default, ...}}。"""
    specs: Dict[str, Dict[str, Any]] = {}
    decl = cls.INPUT_TYPES()
    for bucket in ("required", "optional"):
        for pname, ptype in decl.get(bucket, {}).items():
            if isinstance(ptype, tuple):
                ptype_name, opts = ptype[0], (ptype[1] if len(ptype) > 1 else {})
            else:
                ptype_name, opts = ptype, {}
            specs[pname] = {
                "type": ptype_name,
                "required": bucket == "required",
                "default": opts.get("default"),
                "choices": opts.get("choices"),
                "combo": opts.get("combo"),
            }
    return specs
