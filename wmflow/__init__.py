"""wmflow: 水印嵌入/攻击/提取/评测的节点图工作流引擎 (clean-room, GPL 无涉)。"""
from .types import (IMAGE, MASK, WM_MODEL, MESSAGE, ATTACK, PARAMS,
                    METRICS, REPORT, Message, WMModel, Attack)
from .node import Node, collect_input_specs
from .registry import register, get_node, node_info, all_node_info
from .engine import Executor, WorkflowError, RunResult, save_report

# 导入即注册节点
from . import nodes_core  # noqa: F401
from . import nodes_models  # noqa: F401
from . import nodes_netspec  # noqa: F401

__version__ = "0.1.0"
