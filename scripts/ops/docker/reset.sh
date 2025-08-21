#!/bin/bash
# Complete reset script for RAG Chatbot Platform
# Location: ./scripts/ops/docker/reset.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to project root
cd "$(dirname "$0")/../../.." || exit 1

# Get project name dynamically
PROJECT_NAME=$(basename $(pwd) | tr '[:upper:]' '[:lower:]')

echo -e "${RED}╔════════════════════════════════════════╗${NC}"
echo -e "${RED}║    COMPLETE ENVIRONMENT RESET TOOL     ║${NC}"
echo -e "${RED}║  This will DELETE ALL data and start   ║${NC}"
echo -e "${RED}║         from scratch!                  ║${NC}"
echo -e "${RED}╚════════════════════════════════════════╝${NC}"
echo ""
echo "Project: $PROJECT_NAME"
echo ""

# Confirm action
echo -e "${YELLOW}This action will:${NC}"
echo "  • Stop all containers"
echo "  • Remove all containers"
echo "  • Delete all volumes (permanent data loss)"
echo "  • Remove all networks"
echo "  • Remove project images"
echo "  • Clear all Docker system cache"
echo ""

read -p "$(echo -e ${RED}Are you ABSOLUTELY sure? Type \'RESET ALL\' to continue: ${NC})" confirm

if [ "$confirm" != "RESET ALL" ]; then
    echo -e "${GREEN}Reset cancelled. No changes made.${NC}"
    exit 0
fi

# Final confirmation
echo ""
read -p "$(echo -e ${RED}Final confirmation - this cannot be undone! Type \'YES\' to proceed: ${NC})" final_confirm

if [ "$final_confirm" != "YES" ]; then
    echo -e "${GREEN}Reset cancelled. No changes made.${NC}"
    exit 0
fi

echo ""
echo -e "${YELLOW}Starting complete reset...${NC}"

# Stop all project containers
echo "Stopping all containers..."
docker-compose down 2>/dev/null || true

# Remove all project containers (including orphaned ones)
echo "Removing containers..."
docker ps -a | grep -E "(chatbot-postgres|my-redis|mongodb-atlas-local|scylla-node)" | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true

# Remove all project volumes with dynamic project name
echo "Removing all data volumes..."
docker volume rm -f \
    ${PROJECT_NAME}_postgres_data \
    ${PROJECT_NAME}_redis_data \
    ${PROJECT_NAME}_scylla_data1 \
    ${PROJECT_NAME}_scylla_data2 \
    ${PROJECT_NAME}_scylla_data3 \
    ${PROJECT_NAME}_mongo-data \
    ${PROJECT_NAME}_mongo-config \
    ${PROJECT_NAME}_mongo-mongot 2>/dev/null || true

# Alternative: remove all volumes with project prefix
docker volume ls | grep "${PROJECT_NAME}_" | awk '{print $2}' | xargs -r docker volume rm -f 2>/dev/null || true

# Remove project network
echo "Removing project network..."
docker network rm ${PROJECT_NAME}_default 2>/dev/null || true

# Clean Docker system
echo "Cleaning Docker system..."
docker system prune -f

# Pull fresh images
echo -e "\n${YELLOW}Pulling fresh images...${NC}"
docker-compose pull

# Start fresh environment
echo -e "\n${YELLOW}Starting fresh environment...${NC}"
./scripts/ops/docker/startup.sh

echo ""
echo -e "${GREEN}╔════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║    RESET COMPLETED SUCCESSFULLY!       ║${NC}"
echo -e "${GREEN}║  Your environment is now fresh and     ║${NC}"
echo -e "${GREEN}║         ready for use.                 ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Initialize databases: python scripts/init_databases.py"
echo "  2. Load sample data: python scripts/seed_data.py"
echo "  3. Start your application"