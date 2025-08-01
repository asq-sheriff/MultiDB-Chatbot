# 🤖 Enhanced FAQ Chatbot - Multi-Database Architecture

A production-ready chatbot system built with **FastAPI** and **Python**, featuring a unified three-database architecture for optimal performance, scalability, and reliability.

## 🏗️ **Architecture Overview**

**Three-Database System:**
- **🔴 Redis** - Ultra-fast caching, sessions, real-time analytics, notifications
- **🟢 PostgreSQL** - User authentication, billing, quotas, audit trails  
- **🔵 ScyllaDB** - High-performance conversation history, knowledge base, feedback

**Dual Interface:**
- **🖥️ CLI Interface** - Interactive terminal application (`main.py`)
- **🌐 REST API** - FastAPI web service with full OpenAPI documentation

## ✨ **Key Features**

### 🚀 **Core Capabilities**
- **Intelligent Q&A System** with comprehensive knowledge base
- **Real-time Response Caching** for sub-10ms response times
- **Session Management** with conversation persistence
- **Background Task Processing** for analysis and research
- **Multi-tenant Authentication** with subscription plans (Free/Pro/Enterprise)
- **Real-time Notifications** for task completion

### 📊 **Advanced Features**
- **Usage Quotas & Billing** with plan-based limits
- **Comprehensive Analytics** and usage tracking
- **Audit Logging** for compliance and security
- **Health Monitoring** with detailed status endpoints
- **Graceful Degradation** - continues working even if some databases are unavailable

### 🔒 **Enterprise Ready**
- **JWT Authentication** with secure token management
- **Role-based Access Control** with user permissions
- **Data Persistence** across application restarts
- **Horizontal Scalability** with clustered database support
- **Production Deployment** ready with Docker Compose

## 🚀 **Quick Start**

### **Prerequisites**
- **Python 3.11+** (3.12+ recommended)
- **Docker & Docker Compose** 
- **Git**

### **1. Clone & Setup**
```bash
# Clone the repository
git clone <your-repo-url>
cd MultiDB

# Create virtual environment
python -m venv .venv_multidb
source .venv_multidb/bin/activate  # On Windows: .venv_multidb\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Start Databases**
```bash
# Start all databases with Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps
```

### **3. Run the Application**

**CLI Interface:**
```bash
python main.py
```

**API Server:**
```bash
uvicorn app.api.main:app --reload

# API will be available at:
# • Main API: http://localhost:8000
# • Interactive Docs: http://localhost:8000/docs
# • Health Check: http://localhost:8000/health
```

## 💻 **Usage Examples**

### **CLI Interface**

```bash
# Start interactive session
python main.py

# Available commands:
👤 You: What is Redis?
🤖 Bot: Redis is an in-memory data structure store...

👤 You: /analyze customer purchase patterns from Q4 2024
🤖 Bot: Analysis task submitted! Check /notifications for results.

👤 You: /research latest trends in AI
🤖 Bot: Research task started in background...

👤 You: /notifications
🤖 Bot: 📬 You have 2 new notifications...

👤 You: /stats
🤖 Bot: 📊 System Statistics: Cache hit rate: 85.2%...

👤 You: /dashboard
🤖 Bot: 📋 Your Usage: Messages: 45/1000 (Free Plan)...
```

### **API Usage**

**Authentication:**
```bash
# Register new user
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!", "subscription_plan": "free"}'

# Login and get token
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "SecurePass123!"}'

# Response: {"access_token": "eyJhbGciOiJIUzI1NiIs...", "token_type": "bearer"}
```

**Chat API:**
```bash
# Send message
curl -X POST "http://localhost:8000/chat/message" \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Redis?", "session_id": "optional-session-id"}'

# Get conversation history
curl -H "Authorization: Bearer <your-token>" \
  "http://localhost:8000/chat/history/<session-id>"
```

**Background Tasks:**
```bash
**Background Tasks:**
```bash
# Submit analysis task
curl -X POST "http://localhost:8000/tasks/analyze" \
  -H "Authorization: Bearer <your-token>" \
  -H "Content-Type: application/json" \
  -d '{"description": "Analyze sales trends Q4 2024"}'

# Check task status
curl -H "Authorization: Bearer <your-token>" \
  "http://localhost:8000/tasks/<task-id>/status"

# Get notifications
curl -H "Authorization: Bearer <your-token>" \
  "http://localhost:8000/notifications"
```

## 🗃️ **Database Architecture**

### **Redis (Port 6379)**
```
Purpose: Ultra-fast operations & caching
- Response cache (cache:faq:*)
- User sessions (session:user:*)
- Real-time analytics (analytics:*)
- Notification queues (notifications:user:*)
- Popular question tracking
```

### **PostgreSQL (Port 5432)**
```
Purpose: Business logic & relationships
- users: Authentication & user management
- subscriptions: Billing plans & limits
- usage_records: Quota tracking & analytics
- audit_logs: Security & compliance
- organizations: Multi-tenant support
```

### **ScyllaDB Cluster (Ports 9042, 9043, 9044)**
```
Purpose: High-performance persistent storage
- conversation_history: Chat persistence
- knowledge_base: Q&A knowledge store
- user_feedback: Ratings & feedback
- Supports 3-node cluster for high availability
```

## 🛠️ **Development**

### **Project Structure**
```
MultiDB/
├── main.py                    # CLI interface entry point
├── docker-compose.yml         # Database orchestration
├── requirements.txt           # Python dependencies
├── .env                      # Environment configuration
├── app/
│   ├── api/                  # FastAPI application
│   │   ├── main.py          # API entry point
│   │   ├── endpoints/       # API route handlers
│   │   └── dependencies.py  # Dependency injection
│   ├── database/            # Database connections & models
│   │   ├── postgres_*.py   # PostgreSQL integration
│   │   ├── redis_*.py      # Redis integration
│   │   └── scylla_*.py     # ScyllaDB integration
│   ├── services/            # Business logic layer
│   │   ├── chatbot_service.py      # Core chat logic
│   │   ├── auth_service.py         # Authentication
│   │   ├── background_tasks.py     # Task processing
│   │   └── multi_db_service.py     # Database coordination
│   └── utils/               # Shared utilities
├── docs/                     # Documentation
└── scripts/                  # Utility scripts
```

### **Environment Configuration**
Copy `.env.example` to `.env` and configure:

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chatbot_app
POSTGRES_USER=chatbot_user
POSTGRES_PASSWORD=secure_password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ScyllaDB
SCYLLA_HOST=localhost
SCYLLA_PORT=9042

# Application
SECRET_KEY=your-secret-key-here
LOG_LEVEL=INFO
ENABLE_POSTGRESQL=true
```

### **Running Tests**
```bash
# Test API endpoints
python test_api.py

# Test database connections
python test_redis_connection.py

# Test notification system
python test_notifications.py

# Health check
curl http://localhost:8000/health
```

## 🐳 **Docker Management**

### **Basic Commands**
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Restart specific service
docker-compose restart redis
```

### **Database Operations**
```bash
# PostgreSQL shell
docker-compose exec postgres psql -U chatbot_user -d chatbot_app

# Redis CLI
docker-compose exec redis redis-cli

# ScyllaDB CQL shell
docker-compose exec scylla-node1 cqlsh
```

### **Backup & Recovery**
```bash
# PostgreSQL backup
docker-compose exec postgres pg_dump -U chatbot_user chatbot_app > backup.sql

# Redis backup
docker-compose exec redis redis-cli BGSAVE

# Restore PostgreSQL
docker-compose exec -T postgres psql -U chatbot_user -d chatbot_app < backup.sql
```

## 📈 **Performance & Scaling**

### **Performance Targets**
- **Cached Responses**: <10ms
- **Knowledge Queries**: <100ms  
- **Authentication**: <200ms
- **Background Tasks**: <500ms submission
- **Cache Hit Rate**: >70%

### **Scaling Options**
- **Redis**: Cluster mode with read replicas
- **PostgreSQL**: Read replicas and connection pooling
- **ScyllaDB**: Horizontal scaling with additional nodes
- **Application**: Load balancer with multiple FastAPI instances

## 🚢 **Deployment**

### **Production Checklist**
- [ ] Change default passwords in `.env`
- [ ] Set `SECRET_KEY` to secure random value
- [ ] Configure SSL/TLS certificates
- [ ] Set up monitoring and alerting
- [ ] Configure backup strategies
- [ ] Review security settings
- [ ] Load test with expected traffic

### **Cloud Deployment**
The application supports deployment on:
- **AWS**: RDS (PostgreSQL), ElastiCache (Redis), Keyspaces (ScyllaDB-compatible)
- **GCP**: Cloud SQL, Memorystore, Bigtable
- **Azure**: Database services with managed Redis and Cosmos DB
- **Kubernetes**: Full containerized deployment with Helm charts

## 🔧 **Subscription Plans**

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Messages/month | 1,000 | 10,000 | Unlimited |
| Background tasks | 10 | 100 | 1,000+ |
| API Access | Limited | Full | Full + Custom |
| Support | Community | Email | Priority |
| Analytics | Basic | Advanced | Custom |

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black app/
isort app/
```

## 📚 **Documentation**

- **API Documentation**: http://localhost:8000/docs (when running)
- **User Guide**: [docs/User_Guide_v1.0.md](docs/User_Guide_v1.0.md)
- **Design Document**: [docs/Design_v1.0.md](docs/Design_v1.0.md)
- **Architecture**: [docs/architecture.md](docs/architecture.md)

## 🐛 **Troubleshooting**

### **Common Issues**

**Database Connection Issues:**
```bash
# Check Docker services
docker-compose ps

# Check logs
docker-compose logs postgres
docker-compose logs redis
docker-compose logs scylla-node1
```

**API Not Starting:**
```bash
# Check Python environment
python --version
pip list

# Check port availability
lsof -i :8000

# Start with debug mode
uvicorn app.api.main:app --reload --log-level debug
```

**ScyllaDB Cluster Issues:**
```bash
# Wait for cluster formation (can take 2-3 minutes)
sleep 120

# Check cluster status
docker-compose exec scylla-node1 nodetool status

# Restart if needed
docker-compose restart scylla-node1 scylla-node2 scylla-node3
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **FastAPI** for the excellent async web framework
- **Redis** for blazing-fast caching and pub/sub
- **PostgreSQL** for robust relational data management
- **ScyllaDB** for high-performance NoSQL capabilities
- **Docker** for containerization and deployment simplicity

---

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/your-username/MultiDB/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/MultiDB/discussions)
- **Email**: support@yourcompany.com

**Built with ❤️ using Python, FastAPI, and modern database technologies.**