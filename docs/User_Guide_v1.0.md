
# 👥 User Guide – Unified Three-Database Chatbot (CLI & API)

## Table of Contents
1. [Getting Started](#getting-started)
2. [Setup & Deployment](#setup--deployment)
3. [Authentication & Accounts](#authentication--accounts)
4. [CLI Usage](#cli-usage)
    - [Basic Commands](#basic-commands)
    - [Background Tasks](#background-tasks)
    - [Notifications](#notifications)
    - [Statistics & Analytics](#statistics--analytics)
    - [Session Management](#session-management)
    - [Workflow Examples](#cli-workflow-examples)
5. [API Usage](#api-usage)
    - [Running the API Server](#running-the-api-server)
    - [Authentication: Register/Login & Token Management](#authentication-registerlogin--token-management)
    - [API Endpoints & Example Usage](#api-endpoints--example-usage)
    - [Using CLI & API Together](#using-cli--api-together)
6. [Troubleshooting & FAQ](#troubleshooting--faq)
7. [Best Practices & Tips](#best-practices--tips)

---

## Getting Started

Welcome! This guide will help you set up and use the chatbot system, whether you prefer the CLI (command-line) or the web API. You’ll learn registration/authentication, how to submit analysis and research tasks, manage notifications, and retrieve analytics.

---

## Setup & Deployment

### Prerequisites
- **Python** 3.11+ (3.12+ recommended)
- **Docker** & **Docker Compose** (for database services)
- **Git**

### Quick Start (Local)

```bash
# Clone repository
git clone <repo-url>
cd MultiDB

# Create & activate virtual environment
python -m venv .venv_multidb
source .venv_multidb/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start required databases
cp docker-compose.example.yml docker-compose.yml
docker-compose up -d

# Copy environment template and edit if needed
cp .env.example .env

# Start CLI app
python main.py
```

### Cloud/Kubernetes/Production
For cloud and advanced deployments (AWS, GCP, Azure, K8s), see the [Detail Design Document](Detail_Design_Document.md) for high-level strategies.

---

## Authentication & Accounts

When starting the CLI (`python main.py`), you’ll see:

```
🔐 ENHANCED CHATBOT - Authentication Options
===========================================
1. Login with existing account
2. Create new account
3. Continue as guest (limited features)
```

### Registration (CLI)
```bash
# Start app
python main.py

# Select: 2 (Create new account)
Email: you@example.com
Password: ********

# Choose plan: 1 (Free), 2 (Pro), 3 (Enterprise)
```

### Login (CLI)
```bash
# Select: 1 (Login)
Email: you@example.com
Password: ********
```

**Guest access:** Choose option 3 for basic Q&A (no quotas, no persistence).

### Plans

| Feature         | Free   | Pro      | Enterprise  |
|-----------------|--------|----------|-------------|
| Messages/month  | 1,000  | 10,000   | Unlimited   |
| Background tasks| 10     | 100      | 1,000+      |
| API Access      | Limited| Full     | Full+Custom |

---

## CLI Usage

### Basic Commands

```bash
help                 # Show available commands
/stats               # System and session stats
/feedback <message>  # Submit feedback
/my-feedback         # Show your feedback
/quit                # Exit the chatbot
```

### Background Tasks

- `/analyze <desc>`: Data analysis in background
    - `/analyze customer purchase patterns from Q4 2024`
- `/research <topic>`: Start background research
    - `/research latest trends in AI`
- Task status is sent via notifications

### Notifications

- `/notifications`: See completed task notifications
- `/notifications peek`: Preview unread notifications
- `/notifications clear`: Clear all notifications

### Statistics & Analytics

- `/stats`: View global/session metrics, cache hit rate, active users, etc.
- `/dashboard`: See your usage (messages, quota, plan)
- `/profile`: View account details

### Session Management

- Authenticated users’ sessions persist across restarts.
- Guest sessions are ephemeral.

---

### CLI Workflow Examples

**Chatting with the bot**
```bash
👤 You: What is Redis?
🤖 Bot: Redis is an in-memory data structure store...
```

**Requesting analysis**
```bash
/analyze sales trends Q4 2024
# Notification arrives when analysis is ready
/notifications
```

**Viewing stats and dashboard**
```bash
/stats
/dashboard
/profile
```

**Submitting feedback**
```bash
/feedback This bot saved me hours!
```

---

## API Usage

### Running the API Server

#### Development Server
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Production (example)
```bash
gunicorn app.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

#### Health Check
```bash
curl http://localhost:8000/health
```
Docs:  
- Swagger: http://localhost:8000/docs  
- ReDoc:   http://localhost:8000/redoc

---

### Authentication: Register/Login & Token Management

**Register:**
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "StrongPassword!", "subscription_plan": "free"}'
```
Response includes an access token.

**Login:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "StrongPassword!"}'
```
Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Use token for all authenticated requests:**
```bash
curl -H "Authorization: Bearer <access_token>" ...
```

---

### API Endpoints & Example Usage

**Get Current User Profile**
```bash
curl -H "Authorization: Bearer <token>" http://localhost:8000/auth/me
```

**Send Chat Message**
```bash
curl -X POST http://localhost:8000/chat/message \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"message": "What is Redis?", "session_id": "optional-session-id"}'
```

**Get Conversation History**
```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/chat/history/<session_id>
```

**Submit Analysis Task**
```bash
curl -X POST http://localhost:8000/tasks/analyze \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"description": "Analyze sales Q4 2024"}'
```

**Check Task Status**
```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/tasks/<task_id>/status
```

**Get Task Results**
```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/tasks/<task_id>/results
```

**Get Notifications**
```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/notifications
```

**System Statistics**
```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/analytics/stats
```

---

### Using CLI & API Together

- **Start in CLI**: Get session ID for a conversation.
- **Continue in API**: Use the same session ID in `/chat/message`.
- **Background task submitted via API**: Results/notifications visible in CLI with `/notifications`.
- **Tokens are compatible**: Authenticate in API, paste token in CLI (advanced).

---

## Troubleshooting & FAQ

- **Database won’t start?**  
  Check `docker ps`, logs, and that ports are available.

- **Auth or token errors?**  
  Ensure PostgreSQL is running and `.env` is correct.

- **No response from API?**  
  Check `/health` endpoint and log output.

- **Can’t see analysis results?**  
  Use `/notifications` (CLI) or `/notifications` endpoint (API).

- **Upgrade issues?**  
  Remove `__pycache__` folders, restart Docker, re-run setup.

---

## Best Practices & Tips

- **Monitor quotas**: `/dashboard` (CLI) and analytics endpoints (API)
- **Be specific**: Use clear, specific analysis/research descriptions for best results
- **Re-use sessions**: For multi-step workflows, use consistent session IDs
- **Check notifications often**: Results and research are delivered here
- **Use cache wisely**: Re-phrase questions to leverage cache hits and speed

---

## Need More Help?

- **Technical setup issues:** See [Detail Design Document](Detail_Design_Document.md)
- **Bugs, feature requests:** Contact your platform administrator or development team.

---

