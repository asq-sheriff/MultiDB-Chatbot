# Docker Operations Guide - RAG Chatbot Platform

This comprehensive guide provides all necessary commands and procedures for managing the Docker containers and services for the RAG Chatbot Platform.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Service Architecture](#service-architecture)
3. [Daily Operations](#daily-operations)
4. [Service Management](#service-management)
5. [Database Operations](#database-operations)
6. [Cleanup and Maintenance](#cleanup-and-maintenance)
7. [Troubleshooting](#troubleshooting)
8. [Backup and Recovery](#backup-and-recovery)
9. [Scripts Reference](#scripts-reference)

## Quick Start

### Starting the Platform
```bash
# Start all services
docker-compose up -d

# Or use the startup script for controlled initialization
./scripts/ops/docker/startup.sh

# Verify all services are running
./scripts/ops/docker/health-check.sh

# View logs interactively
./scripts/ops/docker/logs.sh

# Monitor resource usage
./scripts/ops/docker/metrics.sh continuous
```

### Stopping the Platform
```bash
# Graceful shutdown
docker-compose down

# Or use the shutdown script
./scripts/ops/docker/shutdown.sh
```

### Other Useful Commands
```bash
# View all services
docker-compose ps
# Backup all databases
./scripts/ops/docker/backup.sh

# Clean up Docker resources safely
./scripts/ops/docker/cleanup.sh

# Gracefully stop everything
./scripts/ops/docker/shutdown.sh
```
## Service Architecture
Our platform consists of the following services:

| Service | Container Name | Port | Purpose |
|---------|---------------|------|---------|
| PostgreSQL | chatbot-postgres | 5432 | Primary relational database |
| Redis | my-redis | 6379 | Caching and session storage |
| MongoDB | mongodb-atlas-local | 27017 | Document storage |
| ScyllaDB Node 1 | scylla-node1 | 9042 | NoSQL cluster (seed) |
| ScyllaDB Node 2 | scylla-node2 | 9043 | NoSQL cluster node |
| ScyllaDB Node 3 | scylla-node3 | 9044 | NoSQL cluster node |

## Daily Operations

### Morning Startup Routine
```bash
# Use the automated startup script
./scripts/ops/docker/startup.sh

# Or manually start in order
docker-compose up -d postgres redis mongodb
sleep 30  # Wait for databases
docker-compose up -d scylla-node1
sleep 60  # Wait for seed node
docker-compose up -d scylla-node2 scylla-node3
```

### Evening Shutdown Routine
```bash
# Backup before shutdown
./scripts/ops/docker/backup.sh

# Graceful shutdown
./scripts/ops/docker/shutdown.sh
```

### Health Monitoring
```bash
# Quick health check
./scripts/ops/docker/health-check.sh

# Detailed status
docker-compose ps
docker stats --no-stream

# Service logs
docker-compose logs --tail=50 [service-name]
```

## Service Management

### PostgreSQL Operations
```bash
# Connect to database
docker-compose exec postgres psql -U chatbot_user -d chatbot_app

# Quick backup
docker-compose exec postgres pg_dump -U chatbot_user chatbot_app > backup.sql

# Check health
docker-compose exec postgres pg_isready -U chatbot_user

# View active connections
docker-compose exec postgres psql -U chatbot_user -d chatbot_app -c "SELECT * FROM pg_stat_activity;"
```

### Redis Operations
```bash
# Connect to Redis CLI
docker-compose exec redis redis-cli

# Check status
docker-compose exec redis redis-cli ping

# Monitor operations
docker-compose exec redis redis-cli monitor

# Get memory info
docker-compose exec redis redis-cli info memory
```

### MongoDB Operations
```bash
# Connect to MongoDB
docker-compose exec mongodb mongosh -u root -p example

# Check status
docker-compose exec mongodb mongosh --eval "db.adminCommand({ping: 1})" --quiet

# List databases
docker-compose exec mongodb mongosh -u root -p example --eval "show dbs"

# Get server status
docker-compose exec mongodb mongosh -u root -p example --eval "db.serverStatus()"
```

### ScyllaDB Cluster Operations
```bash
# Check cluster status
docker-compose exec scylla-node1 nodetool status

# Connect to CQL shell
docker-compose exec scylla-node1 cqlsh

# Check node info
docker-compose exec scylla-node1 nodetool info

# Describe cluster
docker-compose exec scylla-node1 nodetool describecluster

# Repair cluster (if needed)
docker-compose exec scylla-node1 nodetool repair
```

## Database Operations

### Creating Connections

#### PostgreSQL Connection
```python
# Python example
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="chatbot_app",
    user="chatbot_user",
    password="secure_password"
)
```

#### Redis Connection
```python
# Python example
import redis
r = redis.Redis(host='localhost', port=6379, decode_responses=True)
```

#### MongoDB Connection
```python
# Python example
from pymongo import MongoClient
client = MongoClient('mongodb://root:example@localhost:27017/')
```

#### ScyllaDB Connection
```python
# Python example using cassandra-driver
from cassandra.cluster import Cluster
cluster = Cluster(['localhost'], port=9042)
session = cluster.connect()
```

## Cleanup and Maintenance

### Disk Space Management
```bash
# Check Docker disk usage
docker system df

# Safe cleanup (preserves data)
./scripts/ops/docker/cleanup.sh safe

# Project-specific cleanup
./scripts/ops/docker/cleanup.sh project

# Aggressive cleanup (WARNING: data loss)
./scripts/ops/docker/cleanup.sh aggressive
```

### Manual Cleanup Commands
```bash
# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Remove unused networks
docker network prune -f

# Remove unused volumes (CAUTION: data loss)
docker volume prune -f

# Complete system cleanup
docker system prune -a --volumes -f
```

### Scheduled Maintenance
Add to crontab for automated cleanup:
```bash
# Daily container cleanup
0 2 * * * /path/to/scripts/ops/docker/cleanup.sh safe >> /var/log/docker-cleanup.log 2>&1

# Weekly backup
0 3 * * 0 /path/to/scripts/ops/docker/backup.sh >> /var/log/docker-backup.log 2>&1
```

## Troubleshooting

### Common Issues and Solutions

#### Port Already in Use
```bash
# Find process using port (e.g., 5432)
sudo lsof -i :5432

# Kill process or change port in docker-compose.yml
ports:
  - "5433:5432"  # Use different host port
```

#### Container Won't Start
```bash
# Check logs
docker-compose logs [service-name]

# Force recreate
docker-compose up -d --force-recreate [service-name]

# Reset service completely
docker-compose rm -fsv [service-name]
docker-compose up -d [service-name]
```

#### ScyllaDB Cluster Issues
```bash
# Check node health
docker-compose exec scylla-node1 nodetool status

# If nodes show as DN (down), restart in order
docker-compose restart scylla-node1
sleep 60
docker-compose restart scylla-node2
sleep 30
docker-compose restart scylla-node3
```

#### Connection Refused Between Services
```bash
# Verify network
docker network inspect $(docker-compose ps -q | head -1)

# Test connectivity
docker-compose exec [source-service] ping [target-service]

# Ensure using service names, not localhost
# Correct: postgres, redis, mongodb, scylla-node1
# Wrong: localhost, 127.0.0.1
```

### Performance Issues
```bash
# Check resource usage
docker stats

# Check container limits
docker inspect [container-name] | grep -A 5 "Memory"

# View detailed metrics
./scripts/ops/docker/metrics.sh
```

## Backup and Recovery

### Automated Backup
```bash
# Run full backup
./scripts/ops/docker/backup.sh

# Backup specific service
./scripts/ops/docker/backup.sh postgres
./scripts/ops/docker/backup.sh mongodb
./scripts/ops/docker/backup.sh redis
```

### Manual Backup Commands

#### PostgreSQL Backup
```bash
# Database dump
docker-compose exec postgres pg_dump -U chatbot_user chatbot_app > backups/postgres_$(date +%Y%m%d).sql

# Volume backup
docker run --rm -v postgres_data:/data -v $(pwd)/backups:/backup alpine \
  tar czf /backup/postgres_volume_$(date +%Y%m%d).tar.gz /data
```

#### MongoDB Backup
```bash
# Database dump
docker-compose exec mongodb mongodump --uri="mongodb://root:example@localhost:27017" --out=/tmp/backup
docker cp mongodb-atlas-local:/tmp/backup ./backups/mongo_$(date +%Y%m%d)
```

#### Redis Backup
```bash
# Save snapshot
docker-compose exec redis redis-cli save
docker cp my-redis:/data/dump.rdb ./backups/redis_$(date +%Y%m%d).rdb
```

### Recovery Procedures

#### PostgreSQL Recovery
```bash
# From SQL dump
cat backups/postgres_backup.sql | docker-compose exec -T postgres psql -U chatbot_user -d chatbot_app

# From volume backup
docker run --rm -v postgres_data:/data -v $(pwd)/backups:/backup alpine \
  tar xzf /backup/postgres_volume_backup.tar.gz -C /
```

#### MongoDB Recovery
```bash
# Copy backup to container
docker cp backups/mongo_backup mongodb-atlas-local:/tmp/restore

# Restore
docker-compose exec mongodb mongorestore --uri="mongodb://root:example@localhost:27017" /tmp/restore
```

## Scripts Reference

All operational scripts are located in `./scripts/ops/docker/`:

| Script | Purpose | Usage |
|--------|---------|-------|
| `startup.sh` | Start all services in correct order | `./scripts/ops/docker/startup.sh` |
| `shutdown.sh` | Gracefully stop all services | `./scripts/ops/docker/shutdown.sh` |
| `health-check.sh` | Verify all services are healthy | `./scripts/ops/docker/health-check.sh` |
| `backup.sh` | Backup all databases | `./scripts/ops/docker/backup.sh [service]` |
| `restore.sh` | Restore from backup | `./scripts/ops/docker/restore.sh [service] [backup-file]` |
| `cleanup.sh` | Clean Docker resources | `./scripts/ops/docker/cleanup.sh [safe|project|aggressive]` |
| `reset.sh` | Reset entire environment | `./scripts/ops/docker/reset.sh` |
| `metrics.sh` | Display resource metrics | `./scripts/ops/docker/metrics.sh` |
| `logs.sh` | Tail logs for all services | `./scripts/ops/docker/logs.sh [service]` |

## Environment Variables

Create a `.env` file in the project root:
```bash
# PostgreSQL
POSTGRES_DB=chatbot_app
POSTGRES_USER=chatbot_user
POSTGRES_PASSWORD=secure_password

# MongoDB
MONGODB_INITDB_ROOT_USERNAME=root
MONGODB_INITDB_ROOT_PASSWORD=example
```

## Best Practices

1. **Always backup before major operations**
2. **Use health checks before depending on services**
3. **Monitor disk usage regularly**
4. **Keep logs rotated and cleaned**
5. **Use the provided scripts for consistency**
6. **Document any custom modifications**
7. **Test recovery procedures periodically**

## Support and Resources

- Docker Documentation: https://docs.docker.com/
- Docker Compose Reference: https://docs.docker.com/compose/
- Project Issues: [Create an issue in the project repository]

## Quick Reference Card

```bash
# Most used commands
docker-compose up -d                    # Start all services
docker-compose down                      # Stop all services
docker-compose ps                        # List services
docker-compose logs -f [service]         # View logs
docker-compose exec [service] [command]  # Execute command
docker stats                             # Monitor resources
./scripts/ops/docker/health-check.sh    # Health check
./scripts/ops/docker/backup.sh          # Backup all
./scripts/ops/docker/cleanup.sh safe    # Safe cleanup
```