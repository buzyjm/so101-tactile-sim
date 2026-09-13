#!/usr/bin/env bash
# Start Isaac Sim's WebRTC stream, a TURN relay and the browser client.
#
# The Kit WebRTC stream negotiates over TCP (signalling, port 49100) but
# carries the media over UDP (port 47998).  An SSH tunnel or VS Code port
# forward only carries TCP, so a remote browser could connect the signalling
# and then wait for media forever.  This script therefore also starts a TURN
# server (coturn, in Docker on the host network).  A browser that reaches the
# machine through TCP tunnels only can relay the media through TURN over TCP;
# the relay hands it to Kit over UDP on this host.
#
# Usage:  start_browser_viewer.sh [scene.usda]
#         STREAM_APP_CMD="python scripts/xarm5_dh116/record_gestures.py --stream" start_browser_viewer.sh
#
# With STREAM_APP_CMD set, that command (run inside the isaacsim conda env)
# replaces the stock Isaac Sim streaming app; it must enable the livestream
# itself (Isaac Lab scripts: --livestream 2).  The script exports
# STREAM_KIT_ARGS with the publicIp/GPU settings the app should forward as
# --kit_args.
#
# Environment overrides:
#   STREAM_GPU        physical GPU index to render on              (default 0)
#   STREAM_PUBLIC_IP  address advertised for the media stream      (default: first LAN IP)
#   STREAM_DEBUG=1    verbose StreamSDK logging (ICE, encoder, QoS)
#   TURN_PORT         TURN listening port                          (default 3478)
#   TURN_USER/TURN_PASS TURN credentials, must match the web client (default so101/so101)
#   NO_TURN=1         skip the TURN relay (direct UDP only)
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
usd_path="${1:-${project_root}/lab_scene.usda}"
web_root="${project_root}/web_client/so101-web-viewer"

stream_gpu="${STREAM_GPU:-0}"
signal_port="${SIGNAL_PORT:-49100}"
media_port="${MEDIA_PORT:-47998}"
web_port="${WEB_PORT:-5173}"
turn_port="${TURN_PORT:-3478}"
turn_user="${TURN_USER:-so101}"
turn_pass="${TURN_PASS:-so101}"
turn_relay_min="${TURN_RELAY_MIN:-49160}"
turn_relay_max="${TURN_RELAY_MAX:-49199}"
turn_container="so101-turn"
public_ip="${STREAM_PUBLIC_IP:-$(hostname -I | awk '{print $1}')}"

if [[ -z "${STREAM_APP_CMD:-}" && ! -f "${usd_path}" ]]; then
    echo "USD file not found: ${usd_path}" >&2
    exit 1
fi

if [[ ! -d "${web_root}/node_modules" ]]; then
    echo "Web client dependencies are missing. Run: npm --prefix ${web_root} install" >&2
    exit 1
fi

if [[ -z "${public_ip}" ]]; then
    echo "Could not determine a LAN IP; set STREAM_PUBLIC_IP." >&2
    exit 1
fi

check_tcp_free() {
    if [[ -n "$(ss -H -ltn "sport = :$1")" ]]; then
        echo "TCP $1 is already in use ($2). Stop the existing process first." >&2
        exit 1
    fi
}

check_tcp_free "${signal_port}" "Isaac Sim signalling"
check_tcp_free "${web_port}" "web viewer"
if [[ -n "$(ss -H -lun "sport = :${media_port}")" ]]; then
    echo "UDP ${media_port} is already in use (Isaac Sim media). Stop the existing stream first." >&2
    exit 1
fi

use_turn=1
turn_started=0
if [[ "${NO_TURN:-0}" == "1" ]]; then
    use_turn=0
elif ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    echo "Docker is not available; running without a TURN relay (direct UDP only)." >&2
    use_turn=0
elif docker ps --filter "name=^${turn_container}\$" --filter status=running -q | grep -q .; then
    # A second stream (other SIGNAL_PORT/MEDIA_PORT/WEB_PORT) shares the
    # relay that the first one started; it is left running on exit.
    echo "Reusing the running TURN relay container ${turn_container}."
else
    check_tcp_free "${turn_port}" "TURN"
    turn_started=1
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
    if [[ "${turn_started}" == "1" ]]; then
        docker stop -t 2 "${turn_container}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "${turn_started}" == "1" ]]; then
    docker rm -f "${turn_container}" >/dev/null 2>&1 || true
    # Listens on loopback (SSH/VS Code tunnels land there) and on the LAN
    # address; relays from the LAN address so Kit can reach the relay port.
    docker run -d --rm --name "${turn_container}" --network host \
        coturn/coturn:latest \
        -n --log-file=stdout --no-tls --fingerprint \
        --lt-cred-mech --realm=so101 --user="${turn_user}:${turn_pass}" \
        --listening-port="${turn_port}" \
        --listening-ip=127.0.0.1 --listening-ip="${public_ip}" \
        --relay-ip="${public_ip}" \
        --min-port="${turn_relay_min}" --max-port="${turn_relay_max}" \
        --no-multicast-peers >/dev/null
fi

# Render on one GPU only.  With both cards enabled the app also initialises
# CUDA/OptiX on the second one and fails when that card is busy
# (cudaErrorMemoryAllocation at startup, no frames encoded).  The fixed
# media address matters too: without it StreamSDK picks an interface
# itself, which on this machine may be a Docker bridge.
kit_settings=(
    "--/renderer/activeGpu=${stream_gpu}"
    "--/renderer/multiGpu/enabled=false"
    "--/physics/cudaDevice=${stream_gpu}"
    "--/exts/omni.kit.livestream.app/primaryStream/publicIp=${public_ip}"
    "--/exts/omni.kit.livestream.app/primaryStream/signalPort=${signal_port}"
    "--/exts/omni.kit.livestream.app/primaryStream/streamPort=${media_port}"
)
if [[ "${STREAM_DEBUG:-0}" == "1" ]]; then
    kit_settings+=("--/log/channels/omni.kit.livestream.streamsdk=info")
fi

if [[ -n "${STREAM_APP_CMD:-}" ]]; then
    export STREAM_KIT_ARGS="${kit_settings[*]}"
    export PUBLIC_IP="${public_ip}"
    setsid conda run --no-capture-output -n isaacsim \
        bash -c "cd '${project_root}' && ${STREAM_APP_CMD}" &
    isaac_pid=$!
else
    setsid conda run --no-capture-output -n isaacsim \
        isaacsim isaacsim.exp.full.streaming --no-window "${kit_settings[@]}" "${usd_path}" &
    isaac_pid=$!
fi

setsid npm --prefix "${web_root}" run dev -- --port "${web_port}" &
web_pid=$!

cat <<MSG

Isaac Sim stream:  signalling TCP ${signal_port}, media UDP ${media_port}, advertised as ${public_ip}
MSG
if [[ "${use_turn}" == "1" ]]; then
    echo "TURN relay:        TCP/UDP ${turn_port} (container ${turn_container}, user ${turn_user})"
fi
cat <<MSG

On this machine:   http://localhost:${web_port}
On the LAN:        http://${public_ip}:${web_port}
Through SSH:       ssh -L ${web_port}:localhost:${web_port} -L ${signal_port}:localhost:${signal_port} -L ${turn_port}:localhost:${turn_port} $(id -un)@${public_ip}
                   then open http://localhost:${web_port} (media is relayed through TURN over TCP)
                   VS Code: forward the same three ports from the Ports panel.

The first Isaac Sim start can take several minutes while caches are built.
Press Ctrl+C here to stop everything.
MSG

wait -n "${isaac_pid}" "${web_pid}"
