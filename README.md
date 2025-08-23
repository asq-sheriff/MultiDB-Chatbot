# MultiDB Chatbot 🤖 — Production-Grade RAG with a Composable AI Stack

![CI](https://github.com/asq-sheriff/MultiDB-Chatbot/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

A **production-ready Retrieval-Augmented Generation (RAG) system** showcasing a **Composable AI Stack** for modern enterprise-grade conversational AI.  

Built as part of my **AI/Cloud Engineering portfolio**, this project demonstrates how to design and deliver **scalable, secure, and multi-database conversational systems**.

---

## ✨ Key Highlights

- **Multi-Database Architecture**  
  - **MongoDB Atlas** for vector search  
  - **PostgreSQL** for authentication, billing & transactions  
  - **ScyllaDB** for high-throughput conversation history  
  - **Redis** for session & caching  

- **Composable AI Stack** (inspired by [Blueprint doc](/docs/multidb_rag_chatbot_v3.0.md) and [Composable AI Stack Whitepaper](/docs/Composable_AI_Stack_Blueprint.pdf)):  
  - **Dagster** — Data Plane (document ingestion → chunking → embedding → vector store)  
  - **Ray Serve** — Serving Plane (low-latency, stateful LangGraph agent)  
  - **Prefect** — Control Plane (blue/green deployment, validation flows)  
  - **LangChain / LangGraph** — Agent logic, RAG orchestration  

- **Enterprise Features**  
  - Authentication (JWT, bcrypt)  
  - Billing & subscription with quotas  
  - Role-based access (admin flag `is_superuser`)  
  - Rate limiting + quota enforcement  
  - Dockerized for reproducible deployments  

---

## 🏗 Architecture

![Architecture](docs/images/architecture.png)

**Two-Plane Design** (from [Composable AI Stack Blueprint](/docs/Composable_AI_Stack_Blueprint.pdf)):  
- **Data Plane (Dagster)** — Reliable ingestion, embeddings, vector index versioning.  
- **Serving Plane (Ray Serve)** — Low-latency inference, conversational memory.  
- **Control Plane (Prefect)** — Automated rollout, validation, blue/green deployments.  

---

## 🚀 Quickstart

**Local Dev**
```bash
git clone https://github.com/asq-sheriff/MultiDB-Chatbot.git
cd MultiDB-Chatbot
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python ./app/api/main.py
```

**Docker**
```bash
docker-compose up --build
```

App runs at: [http://localhost:8000/docs](http://localhost:8000/docs) (FastAPI Swagger UI).

---

## 🔍 Techstack used

- **Python 3.13+** — async, modern tooling (ruff, pytest, mypy, pre-commit)  
- **Databases** — MongoDB Atlas Vector Search, PostgreSQL, Redis, ScyllaDB  
- **AI/ML** — LangChain, LangGraph, SentenceTransformers, Qwen3-1.7B  
- **Orchestration** — Dagster, Prefect, Ray Serve  
- **Cloud/DevOps** — Docker, CI/CD with GitHub Actions, branch protection  

---

## 📈 Roadmap

- [ ] AWS Deployment (EC2/VPC, ALB → Ray Serve)  
- [ ] Prefect-driven blue/green with regression suite  
- [ ] A/B testing new embedding models in production  
- [ ] Public demo endpoint  

---

## 📄 Documentation

- [📄 Codebase Overview](docs/Codebase_Overview.md)
- [Unified System Design (v3.0)](/docs/multidb_rag_chatbot_v3.0.md)

---

## 📜 License
MIT
