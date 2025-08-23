# MultiDB Chatbot 🤖 — V1 (Production‑Minded RAG Foundation)

![CI](https://github.com/asq-sheriff/MultiDB-Chatbot/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

A production‑minded **Retrieval‑Augmented Generation (RAG)** chatbot that demonstrates clean retrieval, a pragmatic multi‑database design, and professional dev/ops hygiene. **V1 prioritizes clarity and correctness** and lays the groundwork for a composable two‑plane architecture in upcoming releases.

> For a concise narrative of what V1 ships and where it’s going next, see [🗺️ V1 Summary & Roadmap](docs/V1_Summary_and_Roadmap.md).  
> For a deeper engineering dive, see [📄 Unified Design (v3.0)](docs/multidb_rag_chatbot_v3.0.md).  
> For a newcomer guide to the repo, see [📘 Codebase Overview](docs/Codebase_Overview.md).

---

## ✨ Current Highlights (V1)

- **API & App**: FastAPI service with clean request/response models.
- **Retrieval**: **MongoDB Atlas Vector Search** for similarity search with metadata filters.
- **Data Layer**:
  - **Redis** (optional) for session cache/rate limiting.
  - **PostgreSQL** scaffolding for auth/billing/RBAC (feature‑flagged; minimal usage in V1).
- **Ops**: Docker Compose for local dev; GitHub Actions CI (ruff + pytest).
- **Docs**: Purpose‑built documentation for reviewers and new contributors (see links above).

> Note: Agent frameworks and orchestration layers are intentionally **not part of V1** and are tracked in the Roadmap.

---

## 🚀 Quickstart

**Local (venv)**
```bash
git clone https://github.com/asq-sheriff/MultiDB-Chatbot.git
cd MultiDB-Chatbot
python -m venv venv && source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python ./app/api/main.py
# Open http://localhost:8000/docs for the FastAPI Swagger UI
```

**Docker**
```bash
docker-compose up --build
# Open http://localhost:8000/docs
```

**Common issues**
- Port collisions → stop old containers (`docker ps` / `docker stop <id>`).  
- DB connection issues → verify `.env` and container health (`docker compose ps`).  
- Atlas TLS/SRV configuration → ensure correct URI and CA if using strict TLS.

---

## 🔍 Tech Stack (V1 — Implemented)

- **Language**: Python 3.13 (CI validated on 3.11/3.12)
- **Web**: FastAPI
- **Vector Retrieval**: MongoDB Atlas Vector Search
- **State/Cache**: Redis (optional)
- **Relational**: PostgreSQL (auth/billing scaffolding; limited usage in V1)
- **Tooling**: Docker, Ruff, Pytest, Makefile utilities

> The long‑term two‑plane architecture (Dagster, Ray Serve, Prefect, LangChain/LangGraph) is on the Roadmap below.

---

## 🗂 Documentation

- [📘 Codebase Overview](docs/Codebase_Overview.md) — structure, responsibilities, flows, “how to get productive in <60 min.”  
- [🗺️ V1 Summary & Roadmap](docs/V1_Summary_and_Roadmap.md) — what V1 ships, trade‑offs, and the Optimal Path forward.  
- [📄 Unified Design (v3.0)](docs/multidb_rag_chatbot_v3.0.md) — engineering‑focused system design (multi‑DB, retrieval, architecture).

**Architecture Diagram**: If `docs/images/architecture.png` is not present, use the diagrams embedded in the [Unified Design (v3.0)](docs/multidb_rag_chatbot_v3.0.md).

---

## 📈 Roadmap (What’s Next)

### Modeling & Agents
- Integrate **SentenceTransformers** (e.g., `all-mpnet-base-v2`) for embeddings.
- Add **LangChain / LangGraph** for a stateful, tool‑using agent.
- Introduce a primary generation model (**Qwen3‑1.7B**) with a clean provider abstraction.
- Golden‑set evals and hallucination checks.

### Orchestration (Two‑Plane Architecture)
- **Dagster — Data Plane**: ingestion → chunking → embedding → vector index versioning (+ FreshnessPolicy/sensors).
- **Ray Serve — Serving Plane**: low‑latency, stateful agent deployments (actor model + request batching).
- **Prefect — Control Plane**: blue/green rollout, approval gates, automated rollback.

### Platform & Infra
- Shift **system‑of‑record to PostgreSQL** (users, quotas, feature flags, event store).
- **Terraform on AWS**: VPC, ECS/EKS, ALB, S3, ECR, IAM least‑privilege, SSM/Secrets Manager.
- Observability: OTel traces, structured logs, Langfuse.

### Productization
- Public demo endpoint.
- PRD‑style safety guardrails (PII scrubbing, jailbreak filters).
- A/B experiments across embeddings & LLMs with cost/latency budgets.

---

## 🤝 Contributing (TL;DR)

1. Branch from `main`: `feat/<slug>` or `fix/<slug>`  
2. Run local checks: `ruff check . && pytest -q`  
3. Open a PR (template provided); CI must pass  
4. For data/model changes, include eval results or golden‑set diffs

---

## 📜 License

MIT
