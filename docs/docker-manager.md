# Docker Container and Services Management Guide

Here's a comprehensive guide to manage your RAG Chatbot Platform containers and services.
This guide provides comprehensive commands for managing your multi-database RAG chatbot platform. The key containers to monitor closely are the ScyllaDB cluster nodes, as they have specific startup dependencies and health requirements.

## Quick Command Reference

```bash
# Start all services in background
docker-compose up -d

# Stop all services
docker-compose down

# View running services
docker-compose ps

# View logs for all services
docker-compose logs

# View logs for specific service
docker-compose logs postgres

# Restart a specific service
docker-compose restart redis

# Rebuild and restart a service
docker-compose up -d --build postgres

# Execute command in running container
docker-compose exec postgres psql -U chatbot_user -d chatbot_app
```

## Service-Specific Management

### PostgreSQL Database

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U chatbot_user -d chatbot_app

# Backup database
docker-compose exec postgres pg_dump -U chatbot_user chatbot_app > backup.sql

# Restore database
cat backup.sql | docker-compose exec -T postgres psql -U chatbot_user -d chatbot_app

# Check database health
docker-compose exec postgres pg_isready -U chatbot_user
```

### Redis

```bash
# Connect to Redis CLI
docker-compose exec redis redis-cli

# Monitor Redis operations
docker-compose exec redis redis-cli monitor

# Check Redis info
docker-compose exec redis redis-cli info
```

### MongoDB

```bash
# Connect to MongoDB shell
docker-compose exec mongodb mongosh -u root -p example

# Check MongoDB status
docker-compose exec mongodb mongosh --eval "db.adminCommand({ping: 1})"

# List databases
docker-compose exec mongodb mongosh --eval "show dbs" -u root -p example
```

### ScyllaDB Cluster

```bash
# Check node status
docker-compose exec scylla-node1 nodetool status

# Check cluster info
docker-compose exec scylla-node1 nodetool describecluster

# Check node health
docker-compose exec scylla-node1 nodetool info

# Connect to CQL shell
docker-compose exec scylla-node1 cqlsh

# Repair nodes (if needed)
docker-compose exec scylla-node1 nodetool repair
```

## Maintenance Operations

### Starting Services

```bash
# Start all services
docker-compose up -d

# Start specific services only
docker-compose up -d postgres redis mongodb

# Start services with rebuild
docker-compose up -d --build

# Start services and follow logs
docker-compose up
```

### Stopping Services

```bash
# Stop all services (remove containers)
docker-compose down

# Stop services but keep volumes
docker-compose down --volumes

# Stop specific services
docker-compose stop postgres redis

# Stop and remove volumes (WARNING: data loss)
docker-compose down -v
```

### Monitoring and Logs

```bash
# View logs for all services
docker-compose logs -f

# View logs for specific service
docker-compose logs -f scylla-node1

# View resource usage
docker stats

# Check service health
docker-compose ps --filter "status=running"

# View detailed service information
docker-compose images
docker-compose top
```

### Data Management

```bash
# Backup PostgreSQL data
docker-compose exec postgres pg_dumpall -U chatbot_user > full_backup.sql

# Backup specific volume data
docker run --rm --volumes-from chatbot-postgres -v $(pwd):/backup alpine tar cvf /backup/postgres_backup.tar /var/lib/postgresql/data

# List volumes
docker volume ls

# Inspect volume
docker volume inspect rag-chatbot-platform_postgres_data

# Remove specific volume (WARNING: data loss)
docker volume rm rag-chatbot-platform_postgres_data
```

### Reset Operations

```bash
# Reset specific service (delete data)
docker-compose down -v postgres
docker-compose up -d postgres

# Reset all services (COMPLETE DATA LOSS)
docker-compose down -v
docker-compose up -d

# Reinitialize databases (after reset)
docker-compose exec api python scripts/init_all_databases.py
docker-compose exec api python scripts/seed_sample_data.py
```

### Troubleshooting Commands

```bash
# Check service health status
docker-compose ps --services | xargs -I {} sh -c 'echo {}: $(docker inspect -f "{{.State.Status}}" {})'

# View resource limits
docker-compose ps --services | xargs -I {} sh -c 'echo {}:; docker inspect -f "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}" {}'

# Check service dependencies
docker-compose ps --services | xargs -I {} sh -c 'echo {}:; docker inspect -f "{{.HostConfig.Links}}" {}'

# Get service IP addresses
docker-compose ps --services | xargs -I {} sh -c 'echo {}:; docker inspect -f "{{.NetworkSettings.Networks.rag-chatbot-platform_default.IPAddress}}" {}'
```

## Common Maintenance Scenarios

### Scenario 1: Daily Startup
```bash
# Start essential services first
docker-compose up -d postgres redis mongodb

# Wait for databases to be ready, then start others
sleep 30
docker-compose up -d scylla-node1 scylla-node2 scylla-node3

# Wait for ScyllaDB cluster formation
sleep 60
docker-compose up -d  # Start remaining services
```

### Scenario 2: Database Backup
```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Backup PostgreSQL
docker-compose exec postgres pg_dump -U chatbot_user chatbot_app > backups/$(date +%Y%m%d)/postgres_backup.sql

# Backup MongoDB (using mongodump inside container)
docker-compose exec mongodb mongodump --uri="mongodb://root:example@localhost:27017" --out=/tmp/backup
docker cp mongodb-atlas-local:/tmp/backup ./backups/$(date +%Y%m%d)/mongodb_backup

# Backup Redis (if persistence is enabled)
docker-compose exec redis redis-cli save
docker cp my-redis:/data/dump.rdb ./backups/$(date +%Y%m%d)/redis_dump.rdb
```

### Scenario 3: Service Update
```bash
# Pull latest images
docker-compose pull

# Recreate services with new images
docker-compose up -d

# Or update specific service
docker-compose pull postgres
docker-compose up -d --force-recreate postgres
```

### Scenario 4: Debugging Issues
```bash
# Check all service logs for errors
docker-compose logs --tail=100 | grep -i error

# Check resource usage
docker-compose top

# Check network connectivity between services
docker-compose exec api ping postgres
docker-compose exec api curl http://redis:6379

# Access container shell for debugging
docker-compose exec api /bin/bash
docker-compose exec postgres /bin/bash
```

## Health Check Commands

```bash
# Verify all services are healthy
./scripts/check_requirements.sh

# Or manually check each service:
# PostgreSQL
docker-compose exec postgres pg_isready -U chatbot_user

# Redis
docker-compose exec redis redis-cli ping

# MongoDB
docker-compose exec mongodb mongosh --eval "db.adminCommand({ping: 1})" --quiet

# ScyllaDB
docker-compose exec scylla-node1 nodetool status | grep UN
```

