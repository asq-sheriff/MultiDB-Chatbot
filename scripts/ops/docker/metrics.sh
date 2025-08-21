#!/bin/bash
# Metrics monitoring script for RAG Chatbot Platform
# Location: ./scripts/ops/docker/metrics.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Navigate to project root
cd "$(dirname "$0")/../../.." || exit 1

# Get project name dynamically
PROJECT_NAME=$(basename $(pwd) | tr '[:upper:]' '[:lower:]')

# Parse arguments
MODE=${1:-once}
INTERVAL=${2:-5}

echo -e "${GREEN}RAG Chatbot Platform Metrics Monitor${NC}"
echo "====================================="
echo "Project: $PROJECT_NAME"

# Function to display metrics
show_metrics() {
    clear
    echo -e "${GREEN}RAG Chatbot Platform Metrics - $(date)${NC}"
    echo "Project: $PROJECT_NAME"
    echo "================================================================"

    # Container stats
    echo -e "\n${YELLOW}Container Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" \
        | grep -E "(CONTAINER|chatbot-postgres|my-redis|mongodb-atlas-local|scylla-node)" || echo "No containers running"

    # Disk usage
    echo -e "\n${YELLOW}Docker Disk Usage:${NC}"
    docker system df

    # Volume sizes with dynamic project name
    echo -e "\n${YELLOW}Volume Sizes:${NC}"
    for volume in postgres_data redis_data scylla_data1 scylla_data2 scylla_data3 mongo-data mongo-config mongo-mongot; do
        FULL_VOLUME_NAME="${PROJECT_NAME}_${volume}"
        if docker volume inspect $FULL_VOLUME_NAME > /dev/null 2>&1; then
            SIZE=$(docker run --rm -v ${FULL_VOLUME_NAME}:/data alpine du -sh /data 2>/dev/null | cut -f1)
            echo "$volume: $SIZE"
        fi
    done

    # Network stats
    echo -e "\n${YELLOW}Network Information:${NC}"
    docker network ls | grep -E "(NETWORK|${PROJECT_NAME}_default)" || echo "No project network found"

    # Database-specific metrics
    echo -e "\n${YELLOW}Database Metrics:${NC}"

    # PostgreSQL connections
    if docker-compose exec -T postgres pg_isready -U chatbot_user > /dev/null 2>&1; then
        CONN_COUNT=$(docker-compose exec -T postgres psql -U chatbot_user -d chatbot_app -t -c "SELECT count(*) FROM pg_stat_activity;" 2>/dev/null | tr -d ' ')
        echo "PostgreSQL active connections: $CONN_COUNT"

        # Database size
        DB_SIZE=$(docker-compose exec -T postgres psql -U chatbot_user -d chatbot_app -t -c "SELECT pg_size_pretty(pg_database_size('chatbot_app'));" 2>/dev/null | tr -d ' ')
        echo "PostgreSQL database size: $DB_SIZE"
    fi

    # Redis memory
    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        REDIS_MEM=$(docker-compose exec -T redis redis-cli info memory | grep used_memory_human | cut -d: -f2 | tr -d '\r')
        echo "Redis memory usage: $REDIS_MEM"

        # Key count
        KEY_COUNT=$(docker-compose exec -T redis redis-cli dbsize | awk '{print $2}')
        echo "Redis key count: $KEY_COUNT"
    fi

    # MongoDB connections
    if docker-compose exec -T mongodb mongosh --eval "db.adminCommand({ping: 1})" --quiet > /dev/null 2>&1; then
        MONGO_CONN=$(docker-compose exec -T mongodb mongosh --eval "db.serverStatus().connections.current" --quiet 2>/dev/null)
        echo "MongoDB current connections: $MONGO_CONN"

        # Database count
        DB_COUNT=$(docker-compose exec -T mongodb mongosh --eval "db.adminCommand('listDatabases').databases.length" --quiet 2>/dev/null)
        echo "MongoDB database count: $DB_COUNT"
    fi

    # ScyllaDB cluster status
    if docker-compose exec -T scylla-node1 nodetool status > /dev/null 2>&1; then
        SCYLLA_NODES=$(docker-compose exec -T scylla-node1 nodetool status | grep -c "^UN" 2>/dev/null || echo "0")
        echo "ScyllaDB healthy nodes: $SCYLLA_NODES/3"

        # Get load average from first node
        LOAD=$(docker-compose exec -T scylla-node1 nodetool info | grep "Load" | awk '{print $3, $4}' 2>/dev/null || echo "N/A")
        echo "ScyllaDB Node 1 load: $LOAD"
    fi
}

# Function for continuous monitoring
continuous_monitor() {
    echo "Starting continuous monitoring (Press Ctrl+C to stop)..."
    echo "Refresh interval: ${INTERVAL} seconds"
    sleep 2

    while true; do
        show_metrics
        sleep $INTERVAL
    done
}

# Main logic
case $MODE in
    once)
        show_metrics
        ;;
    continuous|watch)
        continuous_monitor
        ;;
    *)
        echo "Usage: $0 [once|continuous] [interval_seconds]"
        echo "  once       - Show metrics once (default)"
        echo "  continuous - Continuously monitor metrics"
        echo "  interval   - Refresh interval in seconds (default: 5)"
        exit 1
        ;;
esac