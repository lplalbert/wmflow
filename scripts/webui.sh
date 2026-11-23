#!/usr/bin/env bash
# wmflow webui 服务管理 (pidfile 方式, 避免 pkill 自匹配误杀)
# 用法: bash scripts/webui.sh {start|stop|restart|status} [端口]
set -u
PORT="${2:-7860}"
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY="${WMFLOW_PY:-python3}"
PIDFILE="/tmp/wmflow_webui.pid"
LOG="/tmp/wmflow_webui.log"

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
      echo "已启动 pid $(cat "$PIDFILE") 端口 $PORT"
    else
      echo "启动失败, 看 $LOG"; tail -5 "$LOG"; exit 1
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
    "$0" stop "$PORT"; sleep 1; "$0" start "$PORT" ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "运行中 pid $(cat "$PIDFILE")"; curl -s -m 5 "http://127.0.0.1:$PORT/api/modules" | head -c 60
    else
      echo "未运行"
    fi
    ;;
esac
