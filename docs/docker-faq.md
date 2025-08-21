# Docker Container and Services Management - Comprehensive FAQ

## Table of Contents
1. [General Docker Compose Questions](#general-docker-compose-questions)
2. [Database Management FAQs](#database-management-faqs)
3. [Troubleshooting Common Issues](#troubleshooting-common-issues)
4. [Backup and Recovery](#backup-and-recovery)
5. [Performance and Optimization](#performance-and-optimization)
6. [Security Considerations](#security-considerations)
7. [Advanced Scenarios](#advanced-scenarios)

## General Docker Compose Questions

### Q: What's the difference between `docker-compose up` and `docker-compose up -d`?
**A:** 
- `docker-compose up` starts services in the foreground and attaches your terminal to the logs of all services
- `docker-compose up -d` starts services in the background (detached mode), allowing you to continue using your terminal

### Q: How do I see which services are currently running?
**A:** Use `docker-compose ps` to see the status of all services defined in your docker-compose.yml file. For more detailed information, use `docker-compose ps --services` to list just service names.

### Q: What's the proper way to stop and restart my development environment?
**A:** 
```bash
# For graceful shutdown that preserves data
docker-compose down

# To restart
docker-compose up -d

# For a complete fresh start (WARNING: deletes all data)
docker-compose down -v
docker-compose up -d
```

### Q: How can I check the resource usage of my containers?
**A:** Use `docker stats` to see real-time CPU, memory, and network usage for all containers. For a snapshot, use `docker-compose top` to see processes running inside each container.

## Database Management FAQs

### Q: Why does my ScyllaDB cluster take so long to start?
**A:** ScyllaDB nodes have a complex startup process:
1. Node 1 starts as the seed node (can take 2-4 minutes)
2. Node 2 waits for Node 1 to be healthy before starting
3. Node 3 waits for Node 2 to be healthy
4. Cluster formation and gossip protocol synchronization

**Solution:** Be patient! The health checks are configured to allow up to 4 minutes for nodes to become healthy.

### Q: How do I know if my databases are properly connected?
**A:** Use these verification commands:
```bash
# PostgreSQL
docker-compose exec postgres pg_isready -U chatbot_user

# MongoDB
docker-compose exec mongodb mongosh --eval "db.adminCommand({ping: 1})" --quiet

# Redis
docker-compose exec redis redis-cli ping

# ScyllaDB
docker-compose exec scylla-node1 nodetool status
```

### Q: What should I do if a database service fails to start?
**A:** 
1. Check logs: `docker-compose logs [service-name]`
2. Look for common issues: 
   - Port conflicts (check if ports 5432, 27017, 6379, 9042 are already in use)
   - Disk space issues
   - Corrupted volumes
3. Try restarting just that service: `docker-compose up -d --force-recreate [service-name]`

## Troubleshooting Common Issues

### Q: I'm getting "port already in use" errors. How do I fix this?
**A:** This happens when another process is using the same ports. Solutions:
1. Identify the conflicting process: `sudo lsof -i :5432` (replace with your port)
2. Stop the conflicting service
3. Or change the port mapping in docker-compose.yml:
   ```yaml
   ports:
     - "5433:5432"  # Map host port 5433 to container port 5432
   ```

### Q: My containers are running but the application isn't working. How do I debug?
**A:** Follow this debugging checklist:
1. **Check service health**: `docker-compose ps`
2. **Examine logs**: `docker-compose logs --tail=100`
3. **Test connectivity between services**:
   ```bash
   docker-compose exec api ping postgres
   docker-compose exec api nc -zv redis 6379
   ```
4. **Check resource limits**: `docker stats`
5. **Verify environment variables**: `docker-compose exec api env`

### Q: How do I handle "connection refused" errors between services?
**A:** This usually indicates:
1. The target service isn't running: `docker-compose ps`
2. Network issues: `docker network inspect rag-chatbot-platform_default`
3. DNS resolution problems within the Docker network

**Solution:** Ensure all services are using the Docker Compose service names (e.g., "postgres", not "localhost") for connections.

## Backup and Recovery

### Q: What's the best strategy for backing up my development environment?
**A:** Implement a 3-2-1 backup strategy:
1. **Regular database exports** (daily):
   ```bash
   # PostgreSQL
   docker-compose exec postgres pg_dump -U chatbot_user chatbot_app > backup_$(date +%Y%m%d).sql
   
   # MongoDB
   docker-compose exec mongodb mongodump --uri="mongodb://root:example@localhost:27017" --out=/tmp/backup
   docker cp mongodb-atlas-local:/tmp/backup ./mongo_backup_$(date +%Y%m%d)
   ```

2. **Volume backups** (weekly):
   ```bash
   docker run --rm --volumes-from chatbot-postgres -v $(pwd):/backup alpine \
     tar cvf /backup/postgres_volume_$(date +%Y%m%d).tar /var/lib/postgresql/data
   ```

3. **Docker Compose configuration backup**: Keep your docker-compose.yml and .env files in version control

### Q: How do I restore from a backup?
**A:** 
**PostgreSQL:**
```bash
# Copy backup file to container
docker cp backup.sql chatbot-postgres:/tmp/backup.sql

# Restore
docker-compose exec postgres psql -U chatbot_user -d chatbot_app -f /tmp/backup.sql
```

**MongoDB:**
```bash
# Copy backup to container
docker cp mongo_backup mongodb-atlas-local:/tmp/restore

# Restore
docker-compose exec mongodb mongorestore --uri="mongodb://root:example@localhost:27017" /tmp/restore
```

## Performance and Optimization

### Q: My containers are using too much memory. How can I optimize?
**A:** 
1. **Set memory limits** in docker-compose.yml:
   ```yaml
   services:
     postgres:
       deploy:
         resources:
           limits:
             memory: 1G
   ```

2. **Monitor memory usage**: `docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"`

3. **Optimize database configurations**:
   - Reduce PostgreSQL's shared_buffers if limited memory
   - Configure Redis maxmemory policy
   - Adjust ScyllaDB's memory settings (--memory parameter)

### Q: How can I improve container startup time?
**A:** 
1. **Use named volumes** instead of host-mounted volumes for better performance
2. **Pre-pull images**: `docker-compose pull` before starting
3. **Optimize Docker layer caching** in your Dockerfiles
4. **Use healthchecks** to manage service dependencies effectively

## Security Considerations

### Q: Are my database passwords secure in docker-compose.yml?
**A:** No! Never commit passwords to version control. Instead:

1. **Use environment variables**:
   ```yaml
   environment:
     POSTGRES_PASSWORD: ${DB_PASSWORD}
   ```

2. **Create a .env file** (add to .gitignore):
   ```
   DB_PASSWORD=secure_password_123
   ```

3. **Use Docker secrets** for production deployments

### Q: How do I secure communications between containers?
**A:** 
1. **Use Docker's internal network** which is isolated by default
2. **Implement TLS** for database connections
3. **Use Docker secrets** for sensitive information
4. **Regularly update** your Docker images to patch security vulnerabilities

## Advanced Scenarios

### Q: How do I handle schema migrations in this multi-database environment?
**A:** Implement a migration strategy:

1. **Use database-specific migration tools**:
   - PostgreSQL: Flyway or PostgreSQL native migrations
   - MongoDB: Mongoose migrations or custom scripts
   - ScyllaDB: CQL scripts with versioning

2. **Create migration scripts** that run on application startup:
   ```bash
   # In your API Dockerfile
   COPY migrations/ /app/migrations/
   CMD ["sh", "-c", "python run_migrations.py && python app.py"]
   ```

### Q: How can I simulate production environments in development?
**A:** 
1. **Use Docker Compose profiles** to enable/disable services:
   ```yaml
   services:
     monitoring:
       profiles: ["production"]
       image: prometheus
   ```

2. **Create multiple compose files**:
   ```bash
   # Base configuration
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

3. **Use resource limits** to simulate production constraints

### Q: What's the best way to update to newer versions of database images?
**A:** 
1. **Test upgrades in a staging environment** first
2. **Check version compatibility** between your application and new database versions
3. **Follow this upgrade process**:
   ```bash
   # Pull new images
   docker-compose pull
   
   # Backup current data
   ./scripts/backup.sh
   
   # Stop services
   docker-compose down
   
   # Start with new images
   docker-compose up -d
   
   # Run any necessary migration scripts
   docker-compose exec api python migrate_database.py
   ```

### Q: How do I monitor the health of my multi-database system?
**A:** Implement comprehensive monitoring:

1. **Database-specific health checks**:
   ```bash
   # PostgreSQL
   docker-compose exec postgres pg_isready -U chatbot_user
   
   # ScyllaDB cluster health
   docker-compose exec scylla-node1 nodetool status
   ```

2. **Use monitoring tools**:
   ```yaml
   services:
     prometheus:
       image: prom/prometheus
       ports:
         - "9090:9090"
       
     grafana:
       image: grafana/grafana
       ports:
         - "3000:3000"
   ```

3. **Set up alerts** for critical issues like disk space, memory usage, or service downtime

This FAQ covers the most common scenarios you'll encounter while managing your multi-database RAG chatbot platform. Remember that the key to successful container management is understanding the dependencies between your services and implementing robust monitoring and backup procedures.