#!/usr/bin/env bash
# wmflow webui 服务管理 (pidfile 方式, 避免 pkill 自匹配误杀)
# 用法: bash scripts/webui.sh {start|stop|restart|status} [端口]
set -u
PORT="${2:-7860}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="/tmp/wmflow_webui.pid"
LOG="/tmp/wmflow_webui.log"

# 解释器: 显式 WMFLOW_PY 优先; 否则在常见 env 里挑一个真能 import numpy+torch 的
# (服务器系统 python3 没 numpy, 踩过一次 —— 启动失败只留一行 ModuleNotFoundError)。
pick_py() {
  for c in "${WMFLOW_PY:-}" /data/miniconda_dir/envs/lpl/bin/python \
           "$HOME/miniconda3/envs/lpl/bin/python" python3; do
    [ -z "$c" ] && continue
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c "import numpy, torch" >/dev/null 2>&1 && { echo "$c"; return 0; }
  done
  echo python3
}
PY="$(pick_py)"

case "${1:-status}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "已在运行 (pid $(cat "$PIDFILE"))"; exit 0
    fi
    cd "$DIR"
    setsid nohup "$PY" webui.py --host 0.0.0.0 --port "$PORT" > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 2
    if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "已启动 pid $(cat "$PIDFILE") 端口 $PORT (python: $PY)"
    else
      echo "启动失败 (python: $PY), 看 $LOG"; tail -5 "$LOG"; exit 1
    fi
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      kill "$(cat "$PIDFILE")" 2>/dev/null && echo "已停止 $(cat "$PIDFILE")"
      rm -f "$PIDFILE"
    else
      echo "无 pidfile"
    fi
    ;;
  restart)
    # 用 bash 显式调起, 不依赖文件可执行位 (git 不保留 exec 位, 曾报"权限不够")
    bash "$0" stop "$PORT"; sleep 1; bash "$0" start "$PORT" ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "运行中 pid $(cat "$PIDFILE")"; curl -s -m 5 "http://127.0.0.1:$PORT/api/modules" | head -c 60
    else
      echo "未运行"
    fi
    ;;
esac
