# 🚀 Unified Three-Database Chatbot

A sophisticated chatbot application with **ScyllaDB + Redis + PostgreSQL** integration, featuring both CLI and API interfaces with comprehensive user authentication, background task processing, and enterprise-grade features.

## 🎯 Overview

This project demonstrates a **unified three-database architecture** where each database serves its optimal purpose:

- **ScyllaDB**: High-performance persistent storage for conversations, knowledge base, and feedback
- **Redis**: Lightning-fast caching, session management, and real-time analytics  
- **PostgreSQL**: User authentication, subscription management, billing, and audit trails

## ✨ Features

### 🔐 Authentication & User Management
- User registration and login with JWT tokens
- Subscription plans (Free, Pro, Enterprise) with quota management
- Guest mode for anonymous usage
- Password hashing with bcrypt
- Session persistence across CLI restarts

### 💬 Intelligent Chatbot
- Knowledge base with semantic search
- Redis-powered response caching for sub-millisecond responses
- Conversation history stored permanently in ScyllaDB
- Context-aware responses with confidence scoring
- Feedback collection and analysis

### 🔄 Background Task Processing
- Asynchronous data analysis and research tasks
- Real-time notifications via Redis queues
- Task result persistence and retrieval
- Quota-based task limits per subscription plan
- Intelligent request routing with automatic background processing

### 📊 Analytics & Monitoring
- Real-time usage analytics across all databases
- Personal user dashboards with subscription insights
- Cache hit rate monitoring and optimization
- Comprehensive audit trails for compliance
- Session and performance analytics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   CLI Interface │    │   API Interface │
│                 │    │    (FastAPI)    │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
        ┌────────────▼────────────┐
        │   Enhanced Chatbot      │
        │      Service            │
        └─┬──────────┬──────────┬─┘
          │          │          │
  ┌───────▼─┐   ┌────▼───┐   ┌──▼─────────┐
  │ScyllaDB │   │ Redis  │   │PostgreSQL │
  │         │   │        │   │           │
  │• Convos │   │• Cache │   │• Users    │
  │• Knowledge│ │• Sessions│ │• Billing  │
  │• Feedback│  │• Analytics│ │• Quotas   │
  │• Tasks  │   │• Notifications│• Audit │
  └─────────┘   └────────┘   └───────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for databases)
- Git

### 1. Clone and Setup

```bash
git clone <your-repo>
cd MultiDB
python -m venv .venv_multidb
source .venv_multidb/bin/activate  # On Windows: .venv_multidb\Scripts\activate
pip install -r requirements.txt
```

### 2. Start Databases

```bash
# Start all three databases with Docker Compose
docker-compose up -d

# Verify databases are running
docker ps
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Update .env with your settings (default values work for local development)
```

### 4. Run the Application

```bash
# Start the unified CLI
python main.py
```

## 📱 Usage

### Authentication Flow

```bash
🔐 ENHANCED CHATBOT - Authentication Options
1. Login with existing account
2. Create new account 
3. Continue as guest (limited features)
```

### Available Commands

#### Basic Commands (All Users)
```bash
help                    # Show available commands
/stats                  # System and session statistics
/feedback <message>     # Submit feedback
/my-feedback           # View your feedback history
quit                   # Exit application
```

#### Background Tasks (All Users)
```bash
/analyze <description>  # Start background data analysis
/research <topic>      # Start background research task
/notifications         # Check task completion notifications
/notifications peek    # Preview notifications without marking read
/notifications clear   # Clear all notifications
```

#### Authenticated User Commands
```bash
/dashboard             # Personal dashboard with usage stats
/profile              # User profile and subscription info
```

### Example Usage

```bash
👤 You: What is Redis?
🤖 Bot: Redis is an in-memory data structure store used as a database, cache, and message broker...

👤 You: /analyze customer purchase patterns from Q4 2024
🤖 Bot: 📊 Data Analysis Started!
🎯 Task: customer purchase patterns from Q4 2024
🆔 Task ID: a1b2c3d4...
⏱️ Status: Processing in background

👤 You: /notifications
🤖 Bot: 📬 1 Notification (marked as read):
1. ✅ Data Analysis Complete! Your analysis of 'customer purchase patterns from Q4 2024' completed successfully...
```

## 🎯 Subscription Plans

| Feature | Free | Pro | Enterprise |
|---------|------|-----|------------|
| Messages/month | 1,000 | 10,000 | Unlimited |
| Background tasks/month | 10 | 100 | 1,000 |
| API calls/month | 100 | 1,000 | 10,000 |
| Priority processing | ❌ | ✅ | ✅ |
| Advanced analytics | ❌ | ✅ | ✅ |
| Custom integrations | ❌ | ❌ | ✅ |

## 🔧 Configuration

### Environment Variables

```bash
# PostgreSQL (Required)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=unified_chatbot_db
POSTGRES_USER=chatbot_user
POSTGRES_PASSWORD=secure_password_123

# Redis (Required)  
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Security (Required)
SECRET_KEY=your_generated_secret_key_here

# Optional
LOG_LEVEL=INFO
ENABLE_POSTGRESQL=true
```

### Database Configuration

The application automatically configures:
- **ScyllaDB**: `localhost:9042`, keyspace: `unified_chatbot_ks`
- **Redis**: Configured via environment variables
- **PostgreSQL**: Full async connection pool with optimized settings

## 🐛 Troubleshooting

### Common Issues

#### bcrypt Warning
```bash
# Fix compatibility warning
pip install bcrypt==4.2.0 passlib[bcrypt]==1.7.4
```

#### Database Connection Issues
```bash
# Check database status
docker ps
docker logs chatbot-postgres
docker logs chatbot-redis  
docker logs chatbot-scylla

# Restart databases
docker-compose restart
```

#### Authentication Issues
```bash
# Verify PostgreSQL is running and accessible
psql -h localhost -U chatbot_user -d unified_chatbot_db

# Check environment variables
python -c "from app.config import config; print(config.postgresql.url)"
```

## 📚 API Documentation

The application includes a FastAPI interface alongside the CLI:

```bash
# Start API server
python app/api/main.py

# Access interactive docs
open http://localhost:8000/docs
```

API endpoints mirror CLI functionality with RESTful interfaces for web integration.

## 🏢 Production Deployment

### Docker Production Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis
      - scylla
    
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: unified_chatbot_db
      POSTGRES_USER: chatbot_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
  redis:
    image: redis:6.2
    volumes:
      - redis_data:/data
      
  scylla:
    image: scylladb/scylla:5.2
    volumes:
      - scylla_data:/var/lib/scylla
```

### Security Considerations

- Change all default passwords
- Generate strong SECRET_KEY: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- Enable SSL/TLS for database connections
- Implement proper network security
- Regular security updates
- Use environment-specific configurations

## 🔍 Monitoring

### Built-in Analytics

- Real-time cache hit rates
- User activity tracking  
- Background task performance
- Database connection health
- Response time monitoring

### External Monitoring

Consider integrating:
- **Prometheus + Grafana** for metrics
- **ELK Stack** for log analysis
- **DataDog** for comprehensive monitoring
- **Sentry** for error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-asyncio black flake8

# Run tests
pytest

# Format code
black .

# Check code quality
flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review [GitHub Issues](https://github.com/your-repo/issues)
3. Join our [Discord Community](https://discord.gg/your-invite)
4. Email support: support@your-domain.com

## 🎉 Acknowledgments

- **ScyllaDB** for high-performance NoSQL storage
- **Redis** for blazing-fast caching and session management
- **PostgreSQL** for robust relational data management
- **FastAPI** for modern Python web framework
- **SQLAlchemy** for powerful ORM capabilities

---

**Built with ❤️ using Python, ScyllaDB, Redis, and PostgreSQL**