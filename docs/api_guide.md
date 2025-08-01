# 🔌 API Integration Documentation

## Overview

The Unified Three-Database Chatbot provides both CLI and API interfaces that share the same backend infrastructure. This guide covers the FastAPI web interface, how it integrates with the CLI, and how to use both together.

## 🏗️ API Architecture

### Shared Infrastructure

Both CLI and API use the same:
- **Database layer**: ScyllaDB + Redis + PostgreSQL
- **Authentication system**: JWT tokens and user management
- **Business logic**: Services for billing, user management, and background tasks
- **Data models**: Consistent data structures across interfaces

```
┌─────────────┐    ┌─────────────┐
│ CLI Interface│    │ API Interface│
│  (main.py)  │    │ (FastAPI)   │
└─────┬───────┘    └─────┬───────┘
      │                  │
      └─────────┬────────┘
                │
    ┌───────────▼───────────┐
    │   Shared Services     │
    │ • auth_service        │
    │ • user_service        │
    │ • billing_service     │
    │ • multi_db_service    │
    └─┬──────────┬────────┬─┘
      │          │        │
  ScyllaDB    Redis   PostgreSQL
```

## 🚀 Starting the API Server

### Development Server

```bash
# Activate virtual environment
source .venv_multidb/bin/activate

# Start FastAPI development server
cd app/api
python main.py

# Or use uvicorn directly
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Server

```bash
# Using Gunicorn with uvicorn workers
gunicorn app.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker run -p 8000:8000 -e PORT=8000 your-app-image
```

### Verify API is Running

```bash
# Check health endpoint
curl http://localhost:8000/health

# Access interactive documentation
open http://localhost:8000/docs

# Access ReDoc documentation  
open http://localhost:8000/redoc
```

## 🔐 Authentication Endpoints

### User Registration

**Endpoint:** `POST /auth/register`

```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123",
    "subscription_plan": "free"
  }'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "email": "user@example.com",
    "subscription_plan": "free",
    "is_active": true,
    "is_verified": true
  }
}
```

### User Login

**Endpoint:** `POST /auth/login`

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123"
  }'
```

### Get Current User Profile

**Endpoint:** `GET /auth/me`

```bash
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "email": "user@example.com",
  "subscription_plan": "free",
  "is_active": true,
  "is_verified": true
}
```

### User Dashboard

**Endpoint:** `GET /auth/dashboard`

```bash
curl -X GET "http://localhost:8000/auth/dashboard" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response:**
```json
{
  "user": {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "email": "user@example.com",
    "subscription_plan": "free",
    "is_active": true,
    "created_at": "2025-01-15T10:30:00Z"
  },
  "usage": {
    "messages_this_month": 45,
    "background_tasks_this_month": 3,
    "quota_remaining": 955,
    "limits": {
      "messages": 1000,
      "background_tasks": 10,
      "api_calls": 100
    }
  },
  "sessions": {
    "active_sessions": 1,
    "last_activity": "2025-01-29T14:30:00Z"
  },
  "conversations": {
    "total_conversations": 0,
    "recent_activity": []
  }
}
```

## 💬 Chat Endpoints

### Send Message

**Endpoint:** `POST /chat/message`

```bash
curl -X POST "http://localhost:8000/chat/message" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "message": "What is Redis?",
    "session_id": "optional-session-id"
  }'
```

**Response:**
```json
{
  "response": "Redis is an in-memory data structure store used as a database, cache, and message broker...",
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "confidence": 0.95,
  "cached": true,
  "response_time_ms": 45,
  "databases_used": ["redis", "scylladb"]
}
```

### Get Conversation History

**Endpoint:** `GET /chat/history/{session_id}`

```bash
curl -X GET "http://localhost:8000/chat/history/123e4567-e89b-12d3-a456-426614174000" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response:**
```json
{
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "messages": [
    {
      "actor": "user",
      "message": "What is Redis?",
      "timestamp": "2025-01-29T14:30:00Z"
    },
    {
      "actor": "bot", 
      "message": "Redis is an in-memory data structure store...",
      "timestamp": "2025-01-29T14:30:01Z",
      "confidence": 0.95
    }
  ],
  "total_messages": 2
}
```

## 🔄 Background Task Endpoints

### Submit Analysis Task

**Endpoint:** `POST /tasks/analyze`

```bash
curl -X POST "http://localhost:8000/tasks/analyze" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "description": "Analyze customer purchase patterns from Q4 2024",
    "data_source": "sales_database",
    "parameters": {
      "time_period": "Q4_2024",
      "metrics": ["conversion_rate", "avg_order_value"]
    }
  }'
```

**Response:**
```json
{
  "task_id": "task_123e4567-e89b-12d3-a456-426614174000",
  "status": "submitted",
  "estimated_duration": "5-10 minutes",
  "message": "Analysis task started successfully"
}
```

### Submit Research Task

**Endpoint:** `POST /tasks/research`

```bash
curl -X POST "http://localhost:8000/tasks/research" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "topic": "Latest trends in artificial intelligence",
    "scope": "comprehensive",
    "sources": ["academic", "industry", "news"]
  }'
```

### Get Task Status

**Endpoint:** `GET /tasks/{task_id}/status`

```bash
curl -X GET "http://localhost:8000/tasks/task_123e4567-e89b-12d3-a456-426614174000/status" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response:**
```json
{
  "task_id": "task_123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "created_at": "2025-01-29T14:30:00Z",
  "completed_at": "2025-01-29T14:35:30Z",
  "duration_seconds": 330,
  "result_available": true
}
```

### Get Task Results

**Endpoint:** `GET /tasks/{task_id}/results`

```bash
curl -X GET "http://localhost:8000/tasks/task_123e4567-e89b-12d3-a456-426614174000/results" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response:**
```json
{
  "task_id": "task_123e4567-e89b-12d3-a456-426614174000",
  "task_type": "analysis",
  "results": {
    "summary": "Analysis of customer purchase patterns completed successfully",
    "insights": [
      "Conversion rate increased 15% in Q4 vs Q3",
      "Mobile purchases grew 40% year-over-year",
      "Average order value peaked during Black Friday week"
    ],
    "quantitative_results": {
      "total_transactions": 15847,
      "conversion_rate": 0.087,
      "avg_order_value": 127.45
    },
    "recommendations": [
      "Optimize mobile checkout flow",
      "Expand Black Friday promotional period",
      "Target high-value customer segments"
    ]
  }
}
```

## 🔔 Notification Endpoints

### Get User Notifications

**Endpoint:** `GET /notifications`

```bash
curl -X GET "http://localhost:8000/notifications?limit=10" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response:**
```json
{
  "notifications": [
    {
      "id": "notif_123e4567-e89b-12d3-a456-426614174000",
      "type": "success",
      "title": "Analysis Complete!",
      "message": "Your analysis of customer purchase patterns is ready",
      "data": {
        "task_id": "task_123e4567-e89b-12d3-a456-426614174000",
        "task_type": "analysis"
      },
      "created_at": "2025-01-29T14:35:30Z",
      "read": false
    }
  ],
  "total_count": 1,
  "unread_count": 1
}
```

### Mark Notifications as Read

**Endpoint:** `POST /notifications/mark-read`

```bash
curl -X POST "http://localhost:8000/notifications/mark-read" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "notification_ids": ["notif_123e4567-e89b-12d3-a456-426614174000"]
  }'
```

## 📊 Analytics Endpoints

### Get System Statistics

**Endpoint:** `GET /analytics/stats`

```bash
curl -X GET "http://localhost:8000/analytics/stats" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response:**
```json
{
  "global_stats": {
    "total_users": 1247,
    "total_sessions": 5632,
    "total_messages": 45891,
    "cache_hit_rate": 0.785
  },
  "user_stats": {
    "messages_sent": 45,
    "sessions_created": 3,
    "background_tasks": 2,
    "cache_hits": 35
  },
  "system_health": {
    "databases": {
      "postgresql": "healthy",
      "redis": "healthy", 
      "scylladb": "healthy"
    },
    "response_time_avg_ms": 125
  }
}
```

### Get Usage Analytics

**Endpoint:** `GET /analytics/usage`

```bash
curl -X GET "http://localhost:8000/analytics/usage?period=7d" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 🔗 CLI and API Integration

### Shared Authentication

**Using API token in CLI:**

1. **Get token from API:**
```bash
# Login via API
TOKEN=$(curl -s -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}' | \
  jq -r '.access_token')
```

2. **Use token in CLI:**
The CLI automatically handles authentication, but you can verify the token works:
```bash
# Verify token is valid
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer $TOKEN"
```

### Cross-Platform Data Continuity

**Scenario: Start in CLI, continue in API**

1. **Start conversation in CLI:**
```bash
python main.py
# Login with account
# Ask questions, submit background tasks
# Note the session ID from /session-stats
```

2. **Continue conversation via API:**
```bash
# Use the session ID from CLI
curl -X POST "http://localhost:8000/chat/message" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"message": "What was my last analysis about?", "session_id": "CLI_SESSION_ID"}'
```

**Scenario: Submit task via API, check status in CLI**

1. **Submit via API:**
```bash
curl -X POST "http://localhost:8000/tasks/analyze" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"description": "Market analysis for 2025"}'
```

2. **Check in CLI:**
```bash
python main.py
# Login with same account
# Use: /notifications
# See the completed analysis
```

## 🐳 Docker API Deployment

### API-Only Container

```dockerfile
# Dockerfile.api
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Full-Stack Docker Compose

```yaml
# docker-compose.fullstack.yml
version: '3.8'
services:
  api:
    build: 
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    environment:
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis
      - scylla

  cli:
    build: .
    environment:
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      - postgres
      - redis  
      - scylla
    stdin_open: true
    tty: true

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: unified_chatbot_db
      POSTGRES_USER: chatbot_user
      POSTGRES_PASSWORD: secure_password_123
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6.2-alpine
    volumes:
      - redis_data:/data

  scylla:
    image: scylladb/scylla:5.2
    volumes:
      - scylla_data:/var/lib/scylla

volumes:
  postgres_data:
  redis_data:
  scylla_data:
```

## 📱 Client Examples

### Python Client

```python
import requests
import json

class ChatbotClient:
    def __init__(self, base_url="http://localhost:8000", token=None):
        self.base_url = base_url
        self.token = token
        self.session = requests.Session()
        if token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})
    
    def login(self, email, password):
        response = self.session.post(f"{self.base_url}/auth/login", json={
            "email": email,
            "password": password
        })
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            return data["user"]
        return None
    
    def send_message(self, message, session_id=None):
        response = self.session.post(f"{self.base_url}/chat/message", json={
            "message": message,
            "session_id": session_id
        })
        return response.json() if response.status_code == 200 else None
    
    def submit_analysis(self, description):
        response = self.session.post(f"{self.base_url}/tasks/analyze", json={
            "description": description
        })
        return response.json() if response.status_code == 200 else None
    
    def get_notifications(self):
        response = self.session.get(f"{self.base_url}/notifications")
        return response.json() if response.status_code == 200 else None

# Usage example
client = ChatbotClient()
user = client.login("user@example.com", "password")
if user:
    response = client.send_message("What is Redis?")
    print(response["response"])
    
    task = client.submit_analysis("Q4 sales analysis")
    print(f"Task submitted: {task['task_id']}")
```

### JavaScript/Node.js Client

```javascript
class ChatbotAPI {
    constructor(baseURL = 'http://localhost:8000', token = null) {
        this.baseURL = baseURL;
        this.token = token;
    }

    async login(email, password) {
        const response = await fetch(`${this.baseURL}/auth/login`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({email, password})
        });
        
        if (response.ok) {
            const data = await response.json();
            this.token = data.access_token;
            return data.user;
        }
        return null;
    }

    async sendMessage(message, sessionId = null) {
        const response = await fetch(`${this.baseURL}/chat/message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.token}`
            },
            body: JSON.stringify({message, session_id: sessionId})
        });
        
        return response.ok ? await response.json() : null;
    }

    async getNotifications() {
        const response = await fetch(`${this.baseURL}/notifications`, {
            headers: {'Authorization': `Bearer ${this.token}`}
        });
        
        return response.ok ? await response.json() : null;
    }
}

// Usage
const client = new ChatbotAPI();
const user = await client.login('user@example.com', 'password');
if (user) {
    const response = await client.sendMessage('Hello!');
    console.log(response.response);
}
```

## 🔒 Security Considerations

### API Security

**Authentication:**
- JWT tokens with configurable expiration
- Secure password hashing with bcrypt
- Rate limiting by user and IP address

**Authorization:**
- Role-based access control
- Subscription plan-based feature gating
- Resource quotas and usage tracking

**Data Security:**
- HTTPS in production (TLS 1.3)
- Request/response validation
- SQL injection prevention
- CORS configuration

### Production Security

```python
# Security middleware configuration
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)
```

## 📈 API Performance

### Response Time Targets

| Endpoint Type | Target Time | Optimization |
|---------------|-------------|--------------|
| Authentication | < 200ms | Connection pooling |
| Chat message | < 100ms | Redis caching |
| Background task submit | < 500ms | Async processing |
| Dashboard data | < 300ms | Database optimization |

### Monitoring

```python
# Built-in metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return {
        "requests_total": metrics.requests_total,
        "response_time_avg": metrics.response_time_avg,
        "active_users": metrics.active_users,
        "database_health": await check_database_health()
    }
```

## 🚀 Future API Enhancements

### Planned Features

- **WebSocket support** for real-time chat
- **Streaming responses** for long-form content
- **File upload** for document analysis
- **Webhook support** for task completion notifications
- **GraphQL endpoint** for flexible data queries
- **OpenAPI 3.1** schema with enhanced documentation

### API Versioning

```python
# v2 endpoints with enhanced features
@app.include_router(v1_router, prefix="/api/v1")
@app.include_router(v2_router, prefix="/api/v2")
```

---

**The API provides a powerful, scalable interface that complements the CLI, enabling web applications, mobile apps, and integrations while sharing the same robust three-database backend! 🚀**