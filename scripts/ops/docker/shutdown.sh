#!/bin/bash
# Graceful shutdown script for RAG Chatbot Platform
# Location: ./scripts/ops/docker/shutdown.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Shutting down RAG Chatbot Platform...${NC}"
echo "======================================"

# Navigate to project root
cd "$(dirname "$0")/../../.." || exit 1

# Ask for backup
read -p "Do you want to backup databases before shutdown? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Running backup...${NC}"
    if [ -f "./scripts/ops/docker/backup.sh" ]; then
        ./scripts/ops/docker/backup.sh
    else
        echo -e "${RED}Backup script not found, skipping backup${NC}"
    fi
fi

# Gracefully stop services in reverse order
echo -e "\n${YELLOW}Stopping ScyllaDB cluster...${NC}"
docker-compose stop scylla-node3 scylla-node2 scylla-node1

echo -e "${YELLOW}Stopping databases...${NC}"
docker-compose stop mongodb redis postgres

# Remove containers but keep volumes
echo -e "${YELLOW}Removing containers (volumes preserved)...${NC}"
docker-compose down

echo -e "\n${GREEN}✓ RAG Chatbot Platform shut down successfully${NC}"
echo -e "Data volumes are preserved. Run ${YELLOW}docker-compose up -d${NC} to restart."