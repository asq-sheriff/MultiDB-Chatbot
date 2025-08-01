# 🏗️ System Architecture Documentation

## Overview

The Unified Three-Database Chatbot implements a sophisticated multi-database architecture where each database is optimized for its specific use case, creating a scalable, performant, and maintainable system.

## 🎯 Database Responsibilities

### ScyllaDB - High-Performance Persistent Storage
**Purpose**: Long-term data storage requiring high write throughput and massive scale

**Data Models**:
- **Conversation History**: All user-bot interactions with full context
- **Knowledge Base**: FAQ entries with keyword-based search capabilities  
- **User Feedback**: Feedback collection with sentiment analysis
- **Background Task Results**: Persistent storage of analysis and research results

**Benefits**:
- Handles millions of conversations with linear scalability
- Sub-millisecond read latency for conversation retrieval
- Automatic data distribution and replication
- Compatible with Cassandra ecosystem tools

**Configuration**:
```python
# ScyllaDB Config
hosts: ["127.0.0.1"]
port: 9042
keyspace: "unified_chatbot_ks"
datacenter: "datacenter1"
```

### Redis - High-Speed Operations
**Purpose**: Ultra-fast operations requiring sub-millisecond response times

**Data Structures**:
- **Response Cache**: Hashed message → response mapping (Strings)
- **User Sessions**: Session data with automatic expiration (Hashes)
- **Analytics Counters**: Real-time metrics and counters (Strings/Counters)
- **Notification Queues**: FIFO task completion notifications (Lists)
- **Popularity Tracking**: Trending questions (Sorted Sets)

**Benefits**:
- Cache hit rates > 70% reduce database load significantly
- Session data available instantly across requests
- Real-time analytics without database queries
- Automatic data expiration prevents memory bloat

**Configuration**:
```python
# Redis Config
host: "localhost"
port: 6379
db: 0
default_cache_ttl: 3600      # 1 hour
session_ttl: 86400           # 24 hours
analytics_ttl: 604800        # 7 days
```

### PostgreSQL - Business Logic & Relationships
**Purpose**: Complex relational data requiring ACID compliance and advanced queries

**Core Models**:
- **Users**: Authentication, profiles, preferences
- **Subscriptions**: Billing plans and feature access
- **Usage Records**: Quota tracking and billing data
- **Audit Logs**: Compliance and security tracking
- **Organizations**: Multi-tenant support (future)

**Benefits**:
- ACID compliance ensures data consistency
- Complex joins for analytics and reporting
- Built-in authentication and authorization
- Comprehensive audit trails for compliance

**Configuration**:
```python
# PostgreSQL Config  
host: "localhost"
port: 5432
database: "unified_chatbot_db"
pool_size: 10
max_overflow: 20
url: "postgresql+asyncpg://user:pass@host:port/db"
```

## 🔄 Data Flow Architecture

### 1. User Message Processing Flow

```mermaid
sequenceDiagram
    participant U as User (CLI/API)
    participant C as ChatbotService
    participant R as Redis
    participant P as PostgreSQL  
    participant S as ScyllaDB

    U->>C: Send message
    C->>P: Validate user & quota
    C->>R: Check cache
    alt Cache Hit
        R-->>C: Return cached response
    else Cache Miss
        C->>S: Query knowledge base
        S-->>C: Return knowledge match
        C->>R: Cache response
    end
    C->>S: Store conversation
    C->>P: Record usage
    C-->>U: Return response
```

### 2. Background Task Processing Flow

```mermaid
sequenceDiagram
    participant U as User
    participant C as ChatbotService  
    participant BT as BackgroundTasks
    participant R as Redis
    participant P as PostgreSQL
    participant S as ScyllaDB

    U->>C: Submit /analyze task
    C->>P: Check user quota
    C->>BT: Queue task
    BT->>BT: Process task (async)
    BT->>S: Store results  
    BT->>R: Send notification
    BT->>P: Record usage
    U->>R: Check notifications
    R-->>U: Return task completion
```

### 3. Authentication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant M as Main (CLI)
    participant A as AuthService
    participant P as PostgreSQL
    participant R as Redis

    U->>M: Login request
    M->>A: Authenticate user
    A->>P: Verify credentials
    P-->>A: Return user data
    A->>A: Generate JWT token
    A-->>M: Return token + user
    M->>R: Create session
    M-->>U: Welcome message
```

## 🔧 Service Architecture

### Core Services

#### ChatbotService
**Purpose**: Primary orchestrator for all chatbot functionality

**Responsibilities**:
- Message routing and processing
- Session management
- Cache coordination
- Background task delegation
- Response generation

**Integration Points**:
- Uses `KnowledgeService` for ScyllaDB queries
- Uses `CacheModel` for Redis operations
- Uses `BackgroundTaskService` for async processing
- Uses `AnalyticsModel` for metrics

#### AuthService  
**Purpose**: User authentication and authorization

**Responsibilities**:
- User registration and login
- JWT token generation and validation
- Password hashing and verification
- User session management

**Security Features**:
- bcrypt password hashing
- JWT tokens with configurable expiration
- Rate limiting protection
- Account lockout prevention

#### BackgroundTaskService
**Purpose**: Asynchronous task processing

**Responsibilities**:
- Task queue management
- Parallel task execution
- Result storage and retrieval
- Progress tracking and notifications

**Task Types**:
- Data analysis with statistical processing
- Research with source aggregation
- Report generation
- Custom processing workflows

#### MultiDatabaseService
**Purpose**: Cross-database coordination

**Responsibilities**:
- Transaction coordination across databases
- Data consistency maintenance
- Quota enforcement
- Usage tracking

## 📊 Performance Characteristics

### Response Time Targets

| Operation Type | Target Time | Database | Optimization |
|----------------|-------------|----------|--------------|
| Cached Response | < 10ms | Redis | Hash lookup |
| Knowledge Query | < 100ms | ScyllaDB | Indexed search |
| User Authentication | < 200ms | PostgreSQL | Connection pooling |
| Background Task Submit | < 500ms | All | Async processing |

### Scalability Metrics

| Component | Current Capacity | Scale Limit | Scaling Method |
|-----------|------------------|-------------|----------------|
| Redis Cache | 10GB memory | 500GB | Vertical scaling |
| ScyllaDB | 1TB storage | Petabytes | Horizontal scaling |
| PostgreSQL | 1000 concurrent users | 10,000+ | Read replicas |
| Background Tasks | 10 concurrent | 100+ | Worker scaling |

### Cache Efficiency

**Target Metrics**:
- Cache hit rate: > 70%
- Cache response time: < 5ms
- Cache memory usage: < 80% of allocated
- Cache eviction rate: < 10% daily

**Optimization Strategies**:
- Intelligent cache keys based on message similarity
- TTL optimization based on content type
- Cache warming for popular queries
- Smart eviction policies (LRU + frequency)

## 🔐 Security Architecture

### Authentication & Authorization

**Authentication Flow**:
1. User credentials validated against PostgreSQL
2. JWT token generated with user claims
3. Token stored in session for subsequent requests
4. Token validated on each protected operation

**Authorization Levels**:
- **Guest**: Read-only access, no quota tracking
- **Free User**: Basic features with quotas
- **Pro User**: Enhanced features and higher quotas  
- **Enterprise**: Full features with custom limits

### Data Security

**Encryption**:
- Passwords: bcrypt hashing with salt
- JWT tokens: HMAC-SHA256 signing
- Database connections: TLS in production
- Inter-service communication: TLS certificates

**Access Control**:
- Database user permissions minimized by service
- API rate limiting by user and IP
- Session timeout enforcement
- Audit logging for all sensitive operations

### Compliance Features

**GDPR Compliance**:
- User data deletion workflows
- Data export capabilities
- Consent management
- Processing purpose tracking

**SOC 2 Compliance**:
- Comprehensive audit trails
- Access control monitoring
- Security event logging
- Regular security assessments

## 🚀 Deployment Architecture

### Development Environment

```yaml
# Local development setup
CLI App ←→ Local Databases
├── ScyllaDB (Docker)
├── Redis (Docker)  
└── PostgreSQL (Docker)
```

### Production Environment

```yaml
# Production deployment
Load Balancer
├── App Server 1 ←→ Database Cluster
├── App Server 2 ←→ ├── ScyllaDB Cluster (3+ nodes)
└── App Server N   ├── Redis Cluster (3+ nodes)
                   └── PostgreSQL Primary + Replicas
```

### Container Architecture

```dockerfile
# Multi-stage build
FROM python:3.11-slim as base
# Install dependencies

FROM base as production  
# Copy application code
# Set production configurations
# Health check endpoints
```

### Infrastructure as Code

**Terraform Configuration**:
- VPC and networking setup
- Database cluster provisioning
- Security group configurations
- Load balancer setup
- Auto-scaling groups

**Kubernetes Deployment**:
- Application pods with horizontal scaling
- Database stateful sets
- ConfigMaps for environment variables
- Secrets for sensitive data
- Ingress controllers for routing

## 📈 Monitoring & Observability

### Application Metrics

**Business Metrics**:
- User registration and login rates
- Message processing volume
- Background task completion rates
- Subscription plan distribution

**Technical Metrics**:
- Database connection pool utilization
- Cache hit rates and response times
- Error rates by service and endpoint
- Resource utilization (CPU, memory, disk)

### Logging Strategy

**Structured Logging**:
```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "level": "INFO",
  "service": "chatbot_service",
  "user_id": "123e4567-e89b-12d3-a456-426614174000",
  "action": "message_processed",
  "duration_ms": 45,
  "databases_used": ["redis", "scylladb"],
  "cache_hit": true
}
```

**Log Aggregation**:
- Centralized logging with ELK stack
- Log retention policies by importance
- Real-time alerting on error patterns
- Log analysis for performance optimization

### Health Checks

**Application Health**:
- `/health` endpoint for basic status
- `/health/deep` for database connectivity
- `/metrics` for Prometheus integration
- `/ready` for Kubernetes readiness probes

**Database Health**:
- Connection pool status monitoring
- Query performance tracking
- Replication lag monitoring
- Storage utilization alerts

## 🔄 Data Lifecycle Management

### Data Retention Policies

| Data Type | Retention Period | Archive Strategy | Deletion Method |
|-----------|------------------|------------------|-----------------|
| Conversations | 7 years | Cold storage after 1 year | Soft delete + archival |
| User Sessions | 30 days | No archival | Automatic expiration |
| Analytics Data | 2 years | Aggregated summaries | Rolling deletion |
| Audit Logs | 10 years | Compressed storage | Legal hold support |

### Backup Strategy

**PostgreSQL**:
- Daily full backups
- Hourly incremental backups
- Point-in-time recovery capability
- Cross-region backup replication

**ScyllaDB**:
- Automated snapshot backups
- Incremental backup support
- Cross-datacenter replication
- Table-level restore capability

**Redis**:
- RDB snapshots every 6 hours
- AOF (Append Only File) for persistence
- Backup before major operations
- Quick restore from snapshots

## 🔮 Future Architecture Considerations

### Scaling Enhancements

**Microservices Migration**:
- Service decomposition by domain
- API gateway implementation
- Service mesh for communication
- Independent scaling and deployment

**Event-Driven Architecture**:
- Message queues for async processing
- Event sourcing for audit trails
- CQRS for read/write separation
- Saga patterns for distributed transactions

### Technology Upgrades

**Database Optimization**:
- ScyllaDB materialized views
- PostgreSQL partitioning
- Redis modules for advanced features
- Graph database for relationship queries

**AI/ML Integration**:
- Vector databases for semantic search
- Machine learning model serving
- Real-time recommendation engines
- Natural language processing improvements

This architecture provides a solid foundation for scaling from thousands to millions of users while maintaining performance, security, and reliability standards.