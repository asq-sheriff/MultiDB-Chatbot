
# Enhanced Multi-Database RAG-Based Chatbot Application v3.0

[![Python 3.13.3](https://img.shields.io/badge/python-3.13.3-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, AI-powered conversational system implementing **state-of-the-art Retrieval-Augmented Generation (RAG)** with a **multi-database architecture**, **real-time vector search**, and **enterprise-grade scalability**.

---

## 📌 Executive Summary (For AI Strategy Leaders)

This system is designed to **accelerate AI adoption** in enterprise environments where:
- **Scalability & resilience** are non-negotiable — supports multi-node distributed databases.
- **Latency** matters — sub-2s responses with vector search + caching.
- **Accuracy** is critical — advanced RAG pipeline ensures context-aware answers.
- **Hybrid AI integration** allows M-series Mac, CUDA GPU, or CPU fallback.

**Strategic Differentiators:**
- **Multi-DB Architecture**: MongoDB Atlas (vector search), PostgreSQL (business logic), ScyllaDB (high-throughput history), Redis (cache/session).
- **Enterprise Optimized**: Graceful degradation, health checks, background processing.
- **Deploy Anywhere**: Local dev, on-prem, or cloud-native.

---

## 🏗 Architecture Overview

![System Architecture](docs/images/architecture.png)

### Component Layers
1. **Client Layer**: Web, Mobile, API Clients, Admin Dashboard
2. **FastAPI Application Layer**: Endpoints, Dependency Injection
3. **Service Layer**: Chatbot, Embedding, Generation, Auth, Multi-DB, Background Tasks
4. **Database Layer**: MongoDB, PostgreSQL, ScyllaDB, Redis
5. **AI Models**: sentence-transformers, Qwen3, Mistral fallback

**See full technical details:** [📄 System Design Document](docs/multidb_rag_chatbot_v1.0.md)

---

## 🚀 Quick Start

### 1️⃣ Clone & Environment Setup
```bash
git clone https://github.com/your-org/multidb-rag-chatbot.git
cd multidb-rag-chatbot

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
