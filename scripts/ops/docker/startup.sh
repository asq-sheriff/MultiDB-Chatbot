#!/bin/bash
# Startup script for RAG Chatbot Platform
# Location: ./scripts/ops/docker/startup.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting RAG Chatbot Platform...${NC}"
echo "=================================="

# Navigate to project root (assuming script is in ./scripts/ops/docker/)
cd "$(dirname "$0")/../../.." || exit 1

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found in project root${NC}"
    exit 1
fi

# Function to wait for service
wait_for_service() {
    local service=$1
    local max_attempts=$2
    local attempt=0

    echo -e "${YELLOW}Waiting for $service to be healthy...${NC}"

    while [ $attempt -lt $max_attempts ]; do
        if docker-compose ps | grep -q "$service.*healthy"; then
            echo -e "${GREEN}✓ $service is healthy${NC}"
            return 0
        fi

        attempt=$((attempt + 1))
        echo -n "."
        sleep 5
    done

    echo -e "\n${RED}✗ $service failed to become healthy${NC}"
    return 1
}

# Start core databases first
echo -e "\n${YELLOW}Starting core databases...${NC}"
docker-compose up -d postgres redis mongodb

# Wait for databases to be ready
wait_for_service "postgres" 12
wait_for_service "redis" 12
wait_for_service "mongodb" 24

# Start ScyllaDB cluster
echo -e "\n${YELLOW}Starting ScyllaDB cluster...${NC}"
echo "Note: ScyllaDB cluster initialization may take 2-4 minutes"

# Start seed node first
docker-compose up -d scylla-node1
wait_for_service "scylla-node1" 48  # Up to 4 minutes

# Start remaining ScyllaDB nodes
docker-compose up -d scylla-node2
wait_for_service "scylla-node2" 24  # Up to 2 minutes

docker-compose up -d scylla-node3
wait_for_service "scylla-node3" 24  # Up to 2 minutes

# Verify cluster formation
echo -e "\n${YELLOW}Verifying ScyllaDB cluster formation...${NC}"
sleep 10
if docker-compose exec scylla-node1 nodetool status | grep -c "UN" | grep -q "3"; then
    echo -e "${GREEN}✓ ScyllaDB cluster formed successfully with 3 nodes${NC}"
else
    echo -e "${YELLOW}⚠ ScyllaDB cluster may still be forming${NC}"
fi

# Final status check
echo -e "\n${GREEN}=== Service Status ===${NC}"
docker-compose ps

# Display connection information
echo -e "\n${GREEN}=== Connection Information ===${NC}"
echo "PostgreSQL: localhost:5432 (user: chatbot_user, db: chatbot_app)"
echo "Redis: localhost:6379"
echo "MongoDB: localhost:27017 (user: root, password: example)"
echo "ScyllaDB: localhost:9042"

echo -e "\n${GREEN}✓ RAG Chatbot Platform started successfully!${NC}"
echo -e "Run ${YELLOW}./scripts/ops/docker/health-check.sh${NC} to verify all services"