#!/bin/bash
# Log viewer script for RAG Chatbot Platform
# Location: ./scripts/ops/docker/logs.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Navigate to project root
cd "$(dirname "$0")/../../.." || exit 1

# Parse arguments
SERVICE=${1:-all}
LINES=${2:-50}
FOLLOW=${3:-}

echo -e "${GREEN}RAG Chatbot Platform Log Viewer${NC}"
echo "================================"

# Function to show logs for a specific service
show_service_logs() {
    local service=$1
    local lines=$2
    local follow=$3

    echo -e "\n${YELLOW}Logs for $service:${NC}"
    echo "-------------------"

    if [ "$follow" = "follow" ]; then
        docker-compose logs -f --tail=$lines $service
    else
        docker-compose logs --tail=$lines $service
    fi
}

# Function to show all logs
show_all_logs() {
    local lines=$1
    local follow=$2

    if [ "$follow" = "follow" ]; then
        echo -e "${YELLOW}Following all service logs (Ctrl+C to stop)...${NC}"
        docker-compose logs -f --tail=$lines
    else
        echo -e "${YELLOW}Showing last $lines lines from all services${NC}"
        docker-compose logs --tail=$lines
    fi
}

# Function to search logs
search_logs() {
    local pattern=$1
    local service=${2:-}

    echo -e "${YELLOW}Searching for '$pattern' in logs...${NC}"

    if [ -n "$service" ]; then
        docker-compose logs $service | grep -i "$pattern" | tail -50
    else
        docker-compose logs | grep -i "$pattern" | tail -50
    fi
}

# Function to show error logs
show_errors() {
    local service=${1:-}

    echo -e "${RED}Error logs:${NC}"
    echo "------------"

    if [ -n "$service" ]; then
        docker-compose logs $service | grep -iE "(error|failed|exception|critical)" | tail -50
    else
        docker-compose logs | grep -iE "(error|failed|exception|critical)" | tail -50
    fi
}

# Interactive menu
if [ "$SERVICE" = "menu" ] || [ -z "$SERVICE" ]; then
    echo "
Select log viewing option:
1) View all logs
2) View PostgreSQL logs
3) View Redis logs
4) View MongoDB logs
5) View ScyllaDB logs
6) Follow all logs (real-time)
7) Search logs
8) Show errors only
9) Exit
"
    read -p "Enter option (1-9): " option

    case $option in
        1)
            show_all_logs 100
            ;;
        2)
            show_service_logs postgres 100
            ;;
        3)
            show_service_logs redis 100
            ;;
        4)
            show_service_logs mongodb 100
            ;;
        5)
            echo "Select ScyllaDB node:"
            echo "1) Node 1"
            echo "2) Node 2"
            echo "3) Node 3"
            echo "4) All nodes"
            read -p "Enter option (1-4): " node_option

            case $node_option in
                1) show_service_logs scylla-node1 100 ;;
                2) show_service_logs scylla-node2 100 ;;
                3) show_service_logs scylla-node3 100 ;;
                4)
                    show_service_logs scylla-node1 50
                    show_service_logs scylla-node2 50
                    show_service_logs scylla-node3 50
                    ;;
                *) echo "Invalid option" ;;
            esac
            ;;
        6)
            show_all_logs 50 follow
            ;;
        7)
            read -p "Enter search pattern: " pattern
            read -p "Enter service name (or press enter for all): " service
            search_logs "$pattern" "$service"
            ;;
        8)
            read -p "Enter service name (or press enter for all): " service
            show_errors "$service"
            ;;
        9)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            exit 1
            ;;
    esac
else
    # Command line mode
    case $SERVICE in
        all)
            show_all_logs $LINES $FOLLOW
            ;;
        postgres|redis|mongodb|scylla-node1|scylla-node2|scylla-node3)
            show_service_logs $SERVICE $LINES $FOLLOW
            ;;
        errors)
            show_errors
            ;;
        search)
            if [ -z "$LINES" ]; then
                echo "Usage: $0 search <pattern> [service]"
                exit 1
            fi
            search_logs "$LINES" "$FOLLOW"
            ;;
        follow)
            show_all_logs 50 follow
            ;;
        *)
            echo "Usage: $0 [service|all|errors|search|follow] [lines] [follow]"
            echo ""
            echo "Services: all, postgres, redis, mongodb, scylla-node1, scylla-node2, scylla-node3"
            echo "Special: errors, search, follow"
            echo ""
            echo "Examples:"
            echo "  $0                    # Interactive menu"
            echo "  $0 all                # Show all logs"
            echo "  $0 postgres 100       # Show last 100 lines of PostgreSQL logs"
            echo "  $0 all 50 follow      # Follow all logs"
            echo "  $0 errors             # Show all errors"
            echo "  $0 search 'connection' postgres  # Search for 'connection' in PostgreSQL logs"
            exit 1
            ;;
    esac
fi