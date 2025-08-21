#!/bin/bash
# Docker cleanup script for RAG Chatbot Platform
# Location: ./scripts/ops/docker/cleanup.sh

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
echo "Detected project name: $PROJECT_NAME"

# Parse cleanup mode
MODE=${1:-menu}

echo -e "${YELLOW}Docker Cleanup Utility${NC}"
echo "======================"

# Function to show disk usage
show_usage() {
    echo -e "\n${GREEN}Current Docker disk usage:${NC}"
    docker system df
    echo ""

    # Show project-specific containers
    echo -e "${GREEN}Project-specific containers:${NC}"
    docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Size}}" | grep -E "(chatbot-postgres|my-redis|mongodb-atlas-local|scylla-node)" || echo "No project containers found"
    echo ""

    # Show project-specific volumes
    echo -e "${GREEN}Project-specific volumes:${NC}"
    docker volume ls | grep -E "${PROJECT_NAME}_" || echo "No project volumes found"
}

# Function to perform safe cleanup
safe_cleanup() {
    echo -e "\n${YELLOW}Performing safe cleanup...${NC}"
    echo "This will remove stopped containers, unused networks, and dangling images"

    # Remove stopped containers
    echo "Removing stopped containers..."
    docker container prune -f

    # Remove unused networks
    echo "Removing unused networks..."
    docker network prune -f

    # Remove dangling images
    echo "Removing dangling images..."
    docker image prune -f

    # Clean build cache
    echo "Cleaning build cache..."
    docker builder prune -f

    echo -e "${GREEN}✓ Safe cleanup completed${NC}"
}

# Function to cleanup project only
project_cleanup() {
    echo -e "\n${YELLOW}Cleaning up RAG Chatbot Platform resources...${NC}"
    echo -e "${RED}WARNING: This will remove all project data!${NC}"

    read -p "Are you sure? Type 'yes' to continue: " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Cleanup cancelled"
        return
    fi

    # Stop project services if running
    echo "Stopping project services..."
    docker-compose down 2>/dev/null || true

    # Remove project containers
    echo "Removing project containers..."
    docker ps -a | grep -E "(chatbot-postgres|my-redis|mongodb-atlas-local|scylla-node)" | awk '{print $1}' | xargs -r docker rm -f 2>/dev/null || true

    # Remove project volumes with dynamic project name
    echo "Removing project volumes..."
    docker volume ls | grep "${PROJECT_NAME}_" | awk '{print $2}' | xargs -r docker volume rm -f 2>/dev/null || true

    # Alternative: remove specific named volumes
    docker volume rm -f \
        ${PROJECT_NAME}_postgres_data \
        ${PROJECT_NAME}_redis_data \
        ${PROJECT_NAME}_scylla_data1 \
        ${PROJECT_NAME}_scylla_data2 \
        ${PROJECT_NAME}_scylla_data3 \
        ${PROJECT_NAME}_mongo-data \
        ${PROJECT_NAME}_mongo-config \
        ${PROJECT_NAME}_mongo-mongot 2>/dev/null || true

    # Remove project network
    echo "Removing project network..."
    docker network rm ${PROJECT_NAME}_default 2>/dev/null || true

    echo -e "${GREEN}✓ Project cleanup completed${NC}"
}

# Function to perform aggressive cleanup
aggressive_cleanup() {
    echo -e "\n${RED}WARNING: Aggressive cleanup will remove ALL Docker resources!${NC}"
    echo "This includes:"
    echo "  - All stopped containers"
    echo "  - All unused volumes (DATA LOSS!)"
    echo "  - All unused networks"
    echo "  - All unused images"
    echo "  - All build cache"
    echo ""

    read -p "Are you absolutely sure? Type 'DELETE ALL' to continue: " confirm

    if [ "$confirm" = "DELETE ALL" ]; then
        echo -e "\n${YELLOW}Performing aggressive cleanup...${NC}"

        # Complete system cleanup
        docker system prune -a --volumes -f

        # Clean builder cache
        docker builder prune -a -f

        echo -e "${GREEN}✓ Aggressive cleanup completed${NC}"
    else
        echo "Aggressive cleanup cancelled"
    fi
}

# Function for interactive cleanup
interactive_cleanup() {
    echo -e "\n${YELLOW}Select items to clean:${NC}"

    # Containers
    read -p "Remove stopped containers? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker container prune -f
    fi

    # Images
    read -p "Remove dangling images? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker image prune -f
    fi

    read -p "Remove ALL unused images? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker image prune -a -f
    fi

    # Networks
    read -p "Remove unused networks? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker network prune -f
    fi

    # Volumes
    echo -e "${RED}WARNING: Removing volumes will cause data loss!${NC}"
    read -p "Remove unused volumes? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi

    # Build cache
    read -p "Clean build cache? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker builder prune -f
    fi
}

# Main logic
case $MODE in
    safe)
        show_usage
        safe_cleanup
        show_usage
        ;;
    project)
        show_usage
        project_cleanup
        show_usage
        ;;
    aggressive)
        show_usage
        aggressive_cleanup
        show_usage
        ;;
    interactive)
        show_usage
        interactive_cleanup
        show_usage
        ;;
    menu|*)
        # Show menu
        echo "
Select cleanup option:
1) Show current disk usage
2) Safe cleanup (keeps all data)
3) Interactive cleanup (choose what to remove)
4) Project cleanup (remove $PROJECT_NAME only)
5) Aggressive cleanup (removes everything)
6) Exit
"
        read -p "Enter option (1-6): " option

        case $option in
            1)
                show_usage
                ;;
            2)
                show_usage
                safe_cleanup
                show_usage
                ;;
            3)
                show_usage
                interactive_cleanup
                show_usage
                ;;
            4)
                show_usage
                project_cleanup
                show_usage
                ;;
            5)
                show_usage
                aggressive_cleanup
                show_usage
                ;;
            6)
                echo "Exiting..."
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                exit 1
                ;;
        esac
        ;;
esac

echo -e "\n${GREEN}Cleanup operations completed!${NC}"