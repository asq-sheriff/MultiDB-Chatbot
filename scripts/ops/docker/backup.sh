#!/bin/bash
# Backup script for RAG Chatbot Platform databases
# Location: ./scripts/ops/docker/backup.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Navigate to project root
cd "$(dirname "$0")/../../.." || exit 1

# Configuration
BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_SUBDIR="$BACKUP_DIR/$TIMESTAMP"

# Parse arguments
SERVICE=${1:-all}

# Create backup directory
mkdir -p "$BACKUP_SUBDIR"

echo -e "${GREEN}RAG Chatbot Platform Backup${NC}"
echo "============================"
echo "Backup directory: $BACKUP_SUBDIR"
echo ""

# Function to backup PostgreSQL
backup_postgres() {
    echo -e "${YELLOW}Backing up PostgreSQL...${NC}"

    if docker-compose exec -T postgres pg_isready -U chatbot_user > /dev/null 2>&1; then
        docker-compose exec -T postgres pg_dump -U chatbot_user chatbot_app > "$BACKUP_SUBDIR/postgres_backup.sql"

        # Also backup globals (roles, tablespaces)
        docker-compose exec -T postgres pg_dumpall -U chatbot_user --globals-only > "$BACKUP_SUBDIR/postgres_globals.sql"

        # Compress the backup
        gzip "$BACKUP_SUBDIR/postgres_backup.sql"
        gzip "$BACKUP_SUBDIR/postgres_globals.sql"

        echo -e "${GREEN}✓ PostgreSQL backup completed${NC}"
    else
        echo -e "${RED}✗ PostgreSQL is not running or not healthy${NC}"
        return 1
    fi
}

# Function to backup MongoDB
backup_mongodb() {
    echo -e "${YELLOW}Backing up MongoDB...${NC}"

    if docker-compose exec -T mongodb mongosh --eval "db.adminCommand({ping: 1})" --quiet > /dev/null 2>&1; then
        # Create backup inside container
        docker-compose exec -T mongodb mongodump \
            --uri="mongodb://root:example@localhost:27017" \
            --out=/tmp/mongo_backup > /dev/null 2>&1

        # Copy backup to host
        docker cp mongodb-atlas-local:/tmp/mongo_backup "$BACKUP_SUBDIR/mongodb_backup"

        # Clean up container backup
        docker-compose exec -T mongodb rm -rf /tmp/mongo_backup

        # Compress the backup
        tar -czf "$BACKUP_SUBDIR/mongodb_backup.tar.gz" -C "$BACKUP_SUBDIR" mongodb_backup
        rm -rf "$BACKUP_SUBDIR/mongodb_backup"

        echo -e "${GREEN}✓ MongoDB backup completed${NC}"
    else
        echo -e "${RED}✗ MongoDB is not running or not healthy${NC}"
        return 1
    fi
}

# Function to backup Redis
backup_redis() {
    echo -e "${YELLOW}Backing up Redis...${NC}"

    if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        # Force Redis to save current state
        docker-compose exec -T redis redis-cli BGSAVE

        # Wait for save to complete
        echo -n "Waiting for Redis save to complete"
        while [ "$(docker-compose exec -T redis redis-cli LASTSAVE)" = "$(docker-compose exec -T redis redis-cli LASTSAVE)" ]; do
            echo -n "."
            sleep 1
        done
        echo ""

        # Copy dump file
        docker cp my-redis:/data/dump.rdb "$BACKUP_SUBDIR/redis_dump.rdb"

        # Also backup AOF file if it exists
        if docker-compose exec -T redis test -f /data/appendonly.aof; then
            docker cp my-redis:/data/appendonly.aof "$BACKUP_SUBDIR/redis_appendonly.aof"
        fi

        # Compress Redis backups
        gzip "$BACKUP_SUBDIR/redis_dump.rdb"
        [ -f "$BACKUP_SUBDIR/redis_appendonly.aof" ] && gzip "$BACKUP_SUBDIR/redis_appendonly.aof"

        echo -e "${GREEN}✓ Redis backup completed${NC}"
    else
        echo -e "${RED}✗ Redis is not running or not healthy${NC}"
        return 1
    fi
}

# Function to backup ScyllaDB
backup_scylla() {
    echo -e "${YELLOW}Backing up ScyllaDB schema...${NC}"

    if docker-compose exec -T scylla-node1 nodetool status | grep -q "UN"; then
        # Export schema
        docker-compose exec -T scylla-node1 cqlsh -e "DESC SCHEMA" > "$BACKUP_SUBDIR/scylla_schema.cql" 2>/dev/null || true

        # Note: Full data backup would require nodetool snapshot
        # This is a simplified backup for development
        echo -e "${YELLOW}Note: ScyllaDB schema backed up. For full data backup, use nodetool snapshot${NC}"

        # Compress schema
        [ -f "$BACKUP_SUBDIR/scylla_schema.cql" ] && gzip "$BACKUP_SUBDIR/scylla_schema.cql"

        echo -e "${GREEN}✓ ScyllaDB schema backup completed${NC}"
    else
        echo -e "${RED}✗ ScyllaDB cluster is not healthy${NC}"
        return 1
    fi
}

# Function to backup Docker volumes
backup_volumes() {
    echo -e "${YELLOW}Backing up Docker volumes...${NC}"

    # PostgreSQL volume
    docker run --rm -v postgres_data:/data -v "$(pwd)/$BACKUP_SUBDIR":/backup alpine \
        tar czf /backup/postgres_volume.tar.gz /data 2>/dev/null

    # Redis volume
    docker run --rm -v redis_data:/data -v "$(pwd)/$BACKUP_SUBDIR":/backup alpine \
        tar czf /backup/redis_volume.tar.gz /data 2>/dev/null

    echo -e "${GREEN}✓ Volume backups completed${NC}"
}

# Main backup logic
case $SERVICE in
    all)
        echo -e "${YELLOW}Backing up all services...${NC}\n"
        backup_postgres
        backup_mongodb
        backup_redis
        backup_scylla
        # Optionally backup volumes
        # backup_volumes
        ;;
    postgres)
        backup_postgres
        ;;
    mongodb)
        backup_mongodb
        ;;
    redis)
        backup_redis
        ;;
    scylla)
        backup_scylla
        ;;
    volumes)
        backup_volumes
        ;;
    *)
        echo -e "${RED}Unknown service: $SERVICE${NC}"
        echo "Usage: $0 [all|postgres|mongodb|redis|scylla|volumes]"
        exit 1
        ;;
esac

# Create backup metadata
echo "{
  \"timestamp\": \"$TIMESTAMP\",
  \"date\": \"$(date)\",
  \"services\": \"$SERVICE\",
  \"host\": \"$(hostname)\"
}" > "$BACKUP_SUBDIR/backup_metadata.json"

# Display backup summary
echo -e "\n${GREEN}=== Backup Summary ===${NC}"
echo "Location: $BACKUP_SUBDIR"
echo "Contents:"
ls -lh "$BACKUP_SUBDIR"

# Clean old backups (keep last 7 days)
echo -e "\n${YELLOW}Cleaning old backups...${NC}"
find "$BACKUP_DIR" -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true

echo -e "\n${GREEN}✓ Backup completed successfully!${NC}"