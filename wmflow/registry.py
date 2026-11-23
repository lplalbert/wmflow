"""节点注册表: 装饰器注册 + 类型签名自省 (对应 ComfyUI 的 NODE_CLASS_MAPPINGS 与 /object_info)。"""
from __future__ import annotations

from typing import Any, Dict, Type

NODE_CLASS_MAPPINGS: Dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: Dict[str, str] = {}


def register(cls):
    """把节点类注册进全局映射表。用作类装饰器。"""
    name = getattr(cls, "NODE_ID", cls.__name__)
    if name in NODE_CLASS_MAPPINGS:
        raise ValueError(f"节点重名: {name}")
    NODE_CLASS_MAPPINGS[name] = cls
    disp = getattr(cls, "DISPLAY_NAME", None)
    if disp:
        NODE_DISPLAY_NAME_MAPPINGS[name] = disp
    return cls


def get_node(name: str) -> type:
    if name not in NODE_CLASS_MAPPINGS:
        raise KeyError(f"未知节点: {name}")
    return NODE_CLASS_MAPPINGS[name]


def node_info(name: str) -> Dict[str, Any]:
    """单节点 schema 自省 (ComfyUI /object_info 的极简版)。"""
    cls = get_node(name)
    return {
        "class_type": name,
        "category": getattr(cls, "CATEGORY", ""),
        "description": (cls.__doc__ or "").strip(),
        "input_types": cls.INPUT_TYPES(),
        "return_types": getattr(cls, "RETURN_TYPES", ()),
        "return_names": getattr(cls, "RETURN_NAMES", ()),
    }


def all_node_info() -> Dict[str, Dict[str, Any]]:
    return {n: node_info(n) for n in sorted(NODE_CLASS_MAPPINGS)}
