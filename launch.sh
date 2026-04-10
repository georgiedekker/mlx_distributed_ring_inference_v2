#!/bin/bash

# Simple distributed inference launcher for mini1 + mini2
# Auto-detects which machine is local and configures accordingly

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ACTION=${1:-start}

# Detect which machine we're running on
detect_machine() {
    if ifconfig 2>/dev/null | grep -q "inet 192.168.5.2 "; then
        # We're on mini2
        LOCAL_IP="192.168.5.2"
        REMOTE_IP="192.168.5.1"
        REMOTE_SSH="mini1@192.168.5.1"
        REMOTE_CWD="/Users/mini1/Movies/mlx_distributed_ring_inference_v2"
        LOCAL_NAME="mini2"
        REMOTE_NAME="mini1"
    elif ifconfig 2>/dev/null | grep -q "inet 192.168.5.1 "; then
        # We're on mini1
        LOCAL_IP="192.168.5.1"
        REMOTE_IP="192.168.5.2"
        REMOTE_SSH="mini2@192.168.5.2"
        REMOTE_CWD="/Users/mini2/Movies/mlx_distributed_ring_inference_v2"
        LOCAL_NAME="mini1"
        REMOTE_NAME="mini2"
    else
        echo -e "${RED}ERROR: Cannot detect machine. Neither 192.168.5.1 nor 192.168.5.2 found on local interfaces.${NC}"
        exit 1
    fi
    echo -e "${GREEN}Detected: ${LOCAL_NAME} (${LOCAL_IP}) — remote: ${REMOTE_NAME} (${REMOTE_IP})${NC}"
}

detect_machine

stop_servers() {
    echo -e "${YELLOW}⏹️  Stopping servers...${NC}"
    
    # Kill main process
    if [ -f .server.pid ]; then
        kill $(cat .server.pid) 2>/dev/null
        rm -f .server.pid
    fi
    
    # Kill local servers
    pkill -f "python.*server\.py" 2>/dev/null
    
    # Kill remote worker server
    ssh $REMOTE_SSH "pkill -f 'python.*server\.py'" 2>/dev/null
    
    # Kill API
    if [ -f .api.pid ]; then
        kill $(cat .api.pid) 2>/dev/null
        rm -f .api.pid
    fi
    pkill -f "python.*api\.py" 2>/dev/null
    
    # Free port
    lsof -ti:8100 | xargs kill -9 2>/dev/null
    
    sleep 1
    echo -e "${GREEN}✓ Stopped${NC}"
}

start_servers() {
    echo -e "${GREEN}🚀 Starting Distributed Inference${NC}"
    echo "=================================="
    
    # Clean up
    rm -f server.log api.log
    
    # Create hosts.json (local = rank 0, remote = rank 1)
    cat > hosts.json << HOSTEOF
[
    {"ssh": "127.0.0.1", "ips": ["${LOCAL_IP}"]},
    {"ssh": "${REMOTE_SSH}", "ips": ["${REMOTE_IP}"]}
]
HOSTEOF

    # Sync server, config, and distributed utils to remote
    echo "Syncing files to ${REMOTE_NAME}..."
    scp server.py pyproject.toml uv.lock ${REMOTE_SSH}:${REMOTE_CWD}/
    scp -r config ${REMOTE_SSH}:${REMOTE_CWD}/
    scp -r distributed ${REMOTE_SSH}:${REMOTE_CWD}/

    # Sync .env if it exists (optional configuration)
    if [ -f .env ]; then
        echo "Syncing .env configuration..."
        scp .env ${REMOTE_SSH}:${REMOTE_CWD}/
    fi
    
    # MLX --cwd applies to ALL nodes, but paths differ per machine.
    # Use a /tmp symlink that resolves to the correct local path on each node.
    SHARED_CWD="/tmp/mlx_ring_cwd"
    ln -sfn "$(pwd)" "${SHARED_CWD}"
    ssh ${REMOTE_SSH} "ln -sfn '${REMOTE_CWD}' '${SHARED_CWD}'"

    # Ensure dependencies are synced on both hosts using uv (prefers .venv)
    echo "Syncing Python deps locally (uv sync)..."
    uv sync --frozen --no-install-project
    echo "Syncing Python deps on ${REMOTE_NAME} (uv sync)..."
    ssh ${REMOTE_SSH} "cd '${REMOTE_CWD}' && uv sync --frozen --no-install-project"

    echo ""
    echo -e "${YELLOW}Starting distributed server...${NC}"

    # Use project venv python from shared symlink so path matches on both hosts
    PYTHON_BIN="${SHARED_CWD}/.venv/bin/python"

    # Build ring hostfile (one port per host starting at 32323)
    RING_HOSTFILE="/tmp/mlx_ring_hostfile.json"
    python3 - <<PY
import json
hosts = [["${LOCAL_IP}:32323"], ["${REMOTE_IP}:32324"]]
with open("${RING_HOSTFILE}", "w") as f:
    json.dump(hosts, f)
PY
    scp ${RING_HOSTFILE} ${REMOTE_SSH}:${RING_HOSTFILE}

    # Start local rank (0) first
    MLX_RANK=0 MLX_WORLD_SIZE=2 MLX_HOSTFILE=${RING_HOSTFILE} nohup ${PYTHON_BIN} ${SHARED_CWD}/server.py >> server.log 2>&1 &
    echo $! > .server.pid

    # Give rank 0 time to start listening
    sleep 5

    # Start remote rank (1)
    ssh ${REMOTE_SSH} "cd '${SHARED_CWD}' && MLX_RANK=1 MLX_WORLD_SIZE=2 MLX_HOSTFILE=${RING_HOSTFILE} nohup ${PYTHON_BIN} ${SHARED_CWD}/server.py >> server.log 2>&1 & echo \\$! > .server.remote.pid"
    
    # Wait for model loading
    echo "Waiting for model loading..."
    sleep 25
    
    if ps -p $(cat .server.pid) > /dev/null 2>&1 && ssh ${REMOTE_SSH} "ps -p \$(cat ${SHARED_CWD}/.server.remote.pid) >/dev/null 2>&1"; then
        echo -e "${GREEN}✓ Distributed server running (local PID: $(cat .server.pid))${NC}"
        
        # Start API
        ${PYTHON_BIN} ${SHARED_CWD}/api.py >> api.log 2>&1 &
        echo $! > .api.pid
        
        sleep 2
        echo -e "${GREEN}✓ API ready at http://localhost:8100${NC}"
        
        echo ""
        echo "Monitor: tail -f server.log"
        echo "Status: ./launch.sh status"
        echo "Test: ./launch.sh test"
    else
        echo -e "${RED}✗ Failed to start${NC}"
        tail -10 server.log
        exit 1
    fi
}

check_status() {
    echo -e "${YELLOW}📊 Status${NC}"
    echo "=========="
    
    echo "Processes:"
    ps aux | grep -E "(server\.py|api\.py)" | grep -v grep || echo "  None"
    
    echo ""
    echo "${REMOTE_NAME} (worker):"
    ssh $REMOTE_SSH "ps aux | grep 'server\.py' | grep -v grep" || echo "  None"
    
    echo ""
    if curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API accessible${NC}"
    else
        echo -e "${RED}✗ API not responding${NC}"
    fi
    
    echo ""
    echo "Recent logs:"
    tail -5 server.log 2>/dev/null || echo "  No logs"
}

test_inference() {
    echo -e "${YELLOW}🧪 Testing Both Ranks${NC}"
    echo "==================="
    
    curl -X POST "http://localhost:8100/v1/chat/completions" \
         -H "Content-Type: application/json" \
         -d '{"messages": [{"role": "user", "content": "Hello, test both ranks"}], "max_tokens": 20}' &
    
    CURL_PID=$!
    
    # Monitor CPU during request
    sleep 1
    echo "CPU usage:"
    ps aux | grep server | grep -v grep | awk '{print "Mini1: " $3 "%"}'
    ssh $REMOTE_SSH "ps aux | grep server | grep -v grep | awk '{print \"${REMOTE_NAME}: \" \$3 \"%\"}'"
    
    wait $CURL_PID
}

case "$ACTION" in
    start)
        stop_servers
        start_servers
        ;;
    stop)
        stop_servers
        ;;
    restart)
        stop_servers
        start_servers
        ;;
    status)
        check_status
        ;;
    test)
        test_inference
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|test}"
        exit 1
        ;;
esac
