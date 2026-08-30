#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
usd_path="${1:-${project_root}/lab_scene.usda}"
web_root="${project_root}/web_client/so101-web-viewer"

if [[ ! -f "${usd_path}" ]]; then
    echo "USD file not found: ${usd_path}" >&2
    exit 1
fi

if [[ ! -d "${web_root}/node_modules" ]]; then
    echo "Web client dependencies are missing. Run: npm --prefix ${web_root} install" >&2
    exit 1
fi

if [[ -n "$(ss -H -ltn 'sport = :49100')" ]]; then
    echo "TCP 49100 is already in use. Stop the existing Isaac Sim stream first." >&2
    exit 1
fi

if [[ -n "$(ss -H -ltn 'sport = :5173')" ]]; then
    echo "TCP 5173 is already in use. Stop the existing web viewer first." >&2
    exit 1
fi

isaac_pid=""
web_pid=""

cleanup() {
    if [[ -n "${web_pid}" ]]; then
        kill -TERM -- "-${web_pid}" 2>/dev/null || true
    fi
    if [[ -n "${isaac_pid}" ]]; then
        kill -TERM -- "-${isaac_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

setsid conda run --no-capture-output -n isaacsim \
    isaacsim isaacsim.exp.full.streaming --no-window "${usd_path}" &
isaac_pid=$!

setsid npm --prefix "${web_root}" run dev &
web_pid=$!

echo "Browser viewer: http://localhost:5173"
echo "LAN viewer:     http://$(hostname -I | awk '{print $1}'):5173"
echo "The first Isaac Sim start can take several minutes while caches are built."
echo "Press Ctrl+C here to stop both services."

wait -n "${isaac_pid}" "${web_pid}"
