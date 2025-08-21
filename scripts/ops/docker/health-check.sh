#!/bin/bash
# Health check script for RAG Chatbot Platform
# Location: ./scripts/ops/docker/health-check.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}RAG Chatbot Platform Health Check${NC}"
echo "=================================="

# Navigate to project root
cd "$(dirname "$0")/../../.." || exit 1

# Initialize counters
TOTAL_SERVICES=0
HEALTHY_SERVICES=0
UNHEALTHY_SERVICES=0

# Function to check service health
check_service() {
    local service=$1
    local container=$2
    local check_command=$3

    TOTAL_SERVICES=$((TOTAL_SERVICES + 1))

    echo -n "Checking $service... "

    # Check if container is running
    if ! docker ps | grep -q "$container"; then
        echo -e "${RED}✗ Container not running${NC}"
        UNHEALTHY_SERVICES=$((UNHEALTHY_SERVICES + 1))
        return 1
    fi

    # Execute health check command
    if eval "$check_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Healthy${NC}"
        HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
        return 0
    else
        echo -e "${RED}✗ Unhealthy${NC}"
        UNHEALTHY_SERVICES=$((UNHEALTHY_SERVICES + 1))
        return 1
    fi
}

# Check each service
echo -e "\n${YELLOW}Service Health Status:${NC}"
echo "----------------------"

check_service "PostgreSQL" "chatbot-postgres" \
    "docker-compose exec -T postgres pg_isready -U chatbot_user"

check_service "Redis" "my-redis" \
    "docker-compose exec -T redis redis-cli ping"

check_service "MongoDB" "mongodb-atlas-local" \
    "docker-compose exec -T mongodb mongosh --eval 'db.adminCommand({ping: 1})' --quiet"

# Fixed ScyllaDB health checks - look for UN status (Up Normal) without checking hostname
check_service "ScyllaDB Node 1" "scylla-node1" \
    "docker-compose exec -T scylla-node1 nodetool status | grep -q '^UN'"

check_service "ScyllaDB Node 2" "scylla-node2" \
    "docker-compose exec -T scylla-node2 nodetool status | grep -q '^UN'"

check_service "ScyllaDB Node 3" "scylla-node3" \
    "docker-compose exec -T scylla-node3 nodetool status | grep -q '^UN'"

# Check disk usage
echo -e "\n${YELLOW}Docker Disk Usage:${NC}"
echo "-------------------"
docker system df | head -n 4

# Check memory usage
echo -e "\n${YELLOW}Container Resource Usage:${NC}"
echo "-------------------------"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" | grep -E "(NAME|chatbot-postgres|my-redis|mongodb-atlas-local|scylla-node)"

# Get project name dynamically
PROJECT_NAME=$(basename $(pwd) | tr '[:upper:]' '[:lower:]')

# Summary
echo -e "\n${YELLOW}=== Health Check Summary ===${NC}"
echo "Total Services: $TOTAL_SERVICES"
echo -e "Healthy: ${GREEN}$HEALTHY_SERVICES${NC}"
echo -e "Unhealthy: ${RED}$UNHEALTHY_SERVICES${NC}"

# Additional ScyllaDB cluster verification
if docker-compose exec -T scylla-node1 nodetool status > /dev/null 2>&1; then
    echo -e "\n${YELLOW}ScyllaDB Cluster Status:${NC}"
    echo "------------------------"
    NODES_UP=$(docker-compose exec -T scylla-node1 nodetool status | grep -c "^UN" || echo 0)
    echo "Nodes Up: $NODES_UP/3"

    # Show actual nodetool status for debugging
    if [ $NODES_UP -lt 3 ]; then
        echo -e "\n${YELLOW}Detailed ScyllaDB Status:${NC}"
        docker-compose exec -T scylla-node1 nodetool status
    fi
fi

if [ $UNHEALTHY_SERVICES -eq 0 ]; then
    echo -e "\n${GREEN}✓ All services are healthy!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Some services are unhealthy. Check logs with:${NC}"
    echo "docker-compose logs [service-name]"
    echo ""
    echo "For ScyllaDB issues, try:"
    echo "  docker-compose exec scylla-node1 nodetool status"
    echo "  docker-compose logs scylla-node1 --tail=50"
    exit 1
fi