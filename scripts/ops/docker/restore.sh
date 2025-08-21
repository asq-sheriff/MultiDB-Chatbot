#!/bin/bash
# Restore script for RAG Chatbot Platform databases
# Location: ./scripts/ops/docker/restore.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Navigate to project root
cd "$(dirname "$0")/../../.." || exit 1

# Configuration
BACKUP_DIR="./backups"

echo -e "${GREEN}RAG Chatbot Platform Restore Utility${NC}"
echo "====================================="

# Parse arguments
SERVICE=${1:-}
BACKUP_PATH=${2:-}

# Function to list available backups
list_backups() {
    echo -e "\n${YELLOW}Available backups:${NC}"
    if [ -d "$BACKUP_DIR" ]; then
        ls -la "$BACKUP_DIR" | grep "^d" | awk '{print $NF}' | grep -v "^\.$" | grep -v "^\.\.$"
    else
        echo "No backups found"
        exit 1
    fi
}

# Function to restore PostgreSQL
restore_postgres() {
    local backup_file=$1

    echo -e "${YELLOW}Restoring PostgreSQL from $backup_file...${NC}"

    # Check if service is running
    if ! docker-compose exec -T postgres pg_isready -U chatbot_user > /dev/null 2>&1; then
        echo -e "${RED}PostgreSQL is not running. Starting service...${NC}"
        docker-compose up -d postgres
        sleep 10
    fi

    # Drop existing connections
    docker-compose exec -T postgres psql -U chatbot_user -d postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'chatbot_app' AND pid <> pg_backend_pid();" 2>/dev/null || true

    # Drop and recreate database
    docker-compose exec -T postgres psql -U chatbot_user -d postgres -c "DROP DATABASE IF EXISTS chatbot_app;" 2>/dev/null || true
    docker-compose exec -T postgres psql -U chatbot_user -d postgres -c "CREATE DATABASE chatbot_app;"

    # Restore backup
    if [[ $backup_file == *.gz ]]; then
        gunzip -c "$backup_file" | docker-compose exec -T postgres psql -U chatbot_user -d chatbot_app
    else
        cat "$backup_file" | docker-compose exec -T postgres psql -U chatbot_user -d chatbot_app
    fi

    echo -e "${GREEN}✓ PostgreSQL restored successfully${NC}"
}

# Function to restore MongoDB
restore_mongodb() {
    local backup_path=$1

    echo -e "${YELLOW}Restoring MongoDB from $backup_path...${NC}"

    # Check if service is running
    if ! docker-compose exec -T mongodb mongosh --eval "db.adminCommand({ping: 1})" --quiet > /dev/null 2>&1; then
        echo -e "${RED}MongoDB is not running. Starting service...${NC}"
        docker-compose up -d mongodb
        sleep 10
    fi

    # Extract backup if compressed
    if [[ $backup_path == *.tar.gz ]]; then
        TEMP_DIR=$(mktemp -d)
        tar -xzf "$backup_path" -C "$TEMP_DIR"
        backup_path="$TEMP_DIR/mongodb_backup"
    fi

    # Copy backup to container
    docker cp "$backup_path" mongodb-atlas-local:/tmp/restore

    # Restore
    docker-compose exec -T mongodb mongorestore \
        --uri="mongodb://root:example@localhost:27017" \
        --drop \
        /tmp/restore

    # Cleanup
    docker-compose exec -T mongodb rm -rf /tmp/restore
    [ -n "${TEMP_DIR:-}" ] && rm -rf "$TEMP_DIR"

    echo -e "${GREEN}✓ MongoDB restored successfully${NC}"
}

# Function to restore Redis
restore_redis() {
    local backup_file=$1

    echo -e "${YELLOW}Restoring Redis from $backup_file...${NC}"

    # Check if service is running
    if ! docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${RED}Redis is not running. Starting service...${NC}"
        docker-compose up -d redis
        sleep 5
    fi

    # Stop Redis to replace dump file
    docker-compose stop redis

    # Extract and copy dump file
    if [[ $backup_file == *.gz ]]; then
        gunzip -c "$backup_file" > /tmp/redis_dump.rdb
        docker cp /tmp/redis_dump.rdb my-redis:/data/dump.rdb
        rm /tmp/redis_dump.rdb
    else
        docker cp "$backup_file" my-redis:/data/dump.rdb
    fi

    # Start Redis
    docker-compose start redis
    sleep 5

    echo -e "${GREEN}✓ Redis restored successfully${NC}"
}

# Function to restore ScyllaDB schema
restore_scylla() {
    local backup_file=$1

    echo -e "${YELLOW}Restoring ScyllaDB schema from $backup_file...${NC}"

    # Check if cluster is running
    if ! docker-compose exec -T scylla-node1 nodetool status | grep -q "UN"; then
        echo -e "${RED}ScyllaDB cluster is not healthy. Please start it first.${NC}"
        exit 1
    fi

    # Restore schema
    if [[ $backup_file == *.gz ]]; then
        gunzip -c "$backup_file" | docker-compose exec -T scylla-node1 cqlsh
    else
        cat "$backup_file" | docker-compose exec -T scylla-node1 cqlsh
    fi

    echo -e "${GREEN}✓ ScyllaDB schema restored successfully${NC}"
    echo -e "${YELLOW}Note: This restores schema only. Data restore requires nodetool snapshot restore${NC}"
}

# Interactive restore
if [ -z "$SERVICE" ]; then
    echo "
Select restore option:
1) List available backups
2) Restore PostgreSQL
3) Restore MongoDB
4) Restore Redis
5) Restore ScyllaDB schema
6) Restore all from a backup directory
7) Exit
"
    read -p "Enter option (1-7): " option

    case $option in
        1)
            list_backups
            ;;
        2)
            list_backups
            echo ""
            read -p "Enter backup directory name: " backup_dir
            read -p "Enter PostgreSQL backup file name: " backup_file
            restore_postgres "$BACKUP_DIR/$backup_dir/$backup_file"
            ;;
        3)
            list_backups
            echo ""
            read -p "Enter backup directory name: " backup_dir
            read -p "Enter MongoDB backup file/directory name: " backup_file
            restore_mongodb "$BACKUP_DIR/$backup_dir/$backup_file"
            ;;
        4)
            list_backups
            echo ""
            read -p "Enter backup directory name: " backup_dir
            read -p "Enter Redis backup file name: " backup_file
            restore_redis "$BACKUP_DIR/$backup_dir/$backup_file"
            ;;
        5)
            list_backups
            echo ""
            read -p "Enter backup directory name: " backup_dir
            read -p "Enter ScyllaDB schema file name: " backup_file
            restore_scylla "$BACKUP_DIR/$backup_dir/$backup_file"
            ;;
        6)
            list_backups
            echo ""
            read -p "Enter backup directory name: " backup_dir
            BACKUP_PATH="$BACKUP_DIR/$backup_dir"

            if [ ! -d "$BACKUP_PATH" ]; then
                echo -e "${RED}Backup directory not found${NC}"
                exit 1
            fi

            echo -e "\n${YELLOW}Restoring all services from $BACKUP_PATH${NC}"

            # Restore each service if backup exists
            [ -f "$BACKUP_PATH/postgres_backup.sql.gz" ] && restore_postgres "$BACKUP_PATH/postgres_backup.sql.gz"
            [ -f "$BACKUP_PATH/mongodb_backup.tar.gz" ] && restore_mongodb "$BACKUP_PATH/mongodb_backup.tar.gz"
            [ -f "$BACKUP_PATH/redis_dump.rdb.gz" ] && restore_redis "$BACKUP_PATH/redis_dump.rdb.gz"
            [ -f "$BACKUP_PATH/scylla_schema.cql.gz" ] && restore_scylla "$BACKUP_PATH/scylla_schema.cql.gz"

            echo -e "\n${GREEN}✓ All available backups restored${NC}"
            ;;
        7)
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
    if [ -z "$BACKUP_PATH" ]; then
        echo "Usage: $0 <service> <backup-file>"
        echo "Services: postgres, mongodb, redis, scylla"
        exit 1
    fi

    case $SERVICE in
        postgres)
            restore_postgres "$BACKUP_PATH"
            ;;
        mongodb)
            restore_mongodb "$BACKUP_PATH"
            ;;
        redis)
            restore_redis "$BACKUP_PATH"
            ;;
        scylla)
            restore_scylla "$BACKUP_PATH"
            ;;
        *)
            echo -e "${RED}Unknown service: $SERVICE${NC}"
            exit 1
            ;;
    esac
fi

echo -e "\n${GREEN}Restore operation completed!${NC}"