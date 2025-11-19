#!/bin/bash

# Simple distributed inference launcher for mini1 + mini2

RED='\033[0;31m'
GREEN='\033[0;32m' 
YELLOW='\033[1;33m'
NC='\033[0m'

ACTION=${1:-start}

stop_servers() {
    echo -e "${YELLOW}⏹️  Stopping servers...${NC}"
    
    # Kill main process
    if [ -f .server.pid ]; then
        kill $(cat .server.pid) 2>/dev/null
        rm -f .server.pid
    fi
    
    # Kill local servers
    pkill -f "python.*server\.py" 2>/dev/null
    
    # Kill mini2 server
    ssh mini2@192.168.5.2 "pkill -f 'python.*server\.py'" 2>/dev/null
    
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
    
    # Create simple hosts.json
    cat > hosts.json << 'EOF'
[
    {"ssh": "localhost", "ips": ["192.168.5.1"]},
    {"ssh": "mini2@192.168.5.2", "ips": ["192.168.5.2"]}
]
EOF
    
    # Sync server, config, and distributed utils to mini2
    echo "Syncing files to mini2..."
    scp server.py mini2@192.168.5.2:/Users/Shared/mlx_distributed_ring_inference_v2/
    scp -r config mini2@192.168.5.2:/Users/Shared/mlx_distributed_ring_inference_v2/
    scp -r distributed mini2@192.168.5.2:/Users/Shared/mlx_distributed_ring_inference_v2/

    # Sync .env if it exists (optional configuration)
    if [ -f .env ]; then
        echo "Syncing .env configuration..."
        scp .env mini2@192.168.5.2:/Users/Shared/mlx_distributed_ring_inference_v2/
    fi
    
    echo ""
    echo -e "${YELLOW}Starting distributed server...${NC}"
    
    # Launch with MLX
    mlx.launch --hostfile hosts.json --backend ring --verbose python3 server.py >> server.log 2>&1 &
    
    echo $! > .server.pid
    
    # Wait for model loading
    echo "Waiting for model loading..."
    sleep 15
    
    if ps -p $(cat .server.pid) > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Distributed server running (PID: $(cat .server.pid))${NC}"
        
        # Start API
        python3 api.py >> api.log 2>&1 &
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
    echo "Mini2:"
    ssh mini2@192.168.5.2 "ps aux | grep 'server\.py' | grep -v grep" || echo "  None"
    
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
    ssh mini2@192.168.5.2 "ps aux | grep server | grep -v grep | awk '{print \"Mini2: \" \$3 \"%\"}'"
    
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