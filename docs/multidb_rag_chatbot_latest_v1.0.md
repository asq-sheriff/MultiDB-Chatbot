# **Multi-Database RAG Chatbot Platform - Ultimate System Design Document**
## **Enterprise AI Platform with Vector Search, Billing & Production Operations**

<div align="center">

**Version 5.0 | January 2025**

*The Definitive Reference for AI Engineers, System Architects, Data Architects & Data Scientists*

[![System Status](https://img.shields.io/badge/Status-Production_Ready-green)]()
[![Architecture](https://img.shields.io/badge/Architecture-Microservices-blue)]()
[![AI Models](https://img.shields.io/badge/AI-RAG_Enabled-purple)]()
[![Databases](https://img.shields.io/badge/Databases-Multi_DB-orange)]()

</div>

---

## **🎯 Welcome to Your RAG Platform Journey**

Think of this document as your comprehensive guide to understanding and building a production-grade RAG (Retrieval-Augmented Generation) chatbot system. We'll start with the fundamental concepts and gradually build up to the complex architectural decisions and operational practices that make this system enterprise-ready.

### **What Makes This System Special**

Before diving into the technical details, let's understand what we're building and why it matters. Imagine you're constructing a highly intelligent assistant that can:

- Answer questions by finding relevant information from millions of documents (like having a photographic memory)
- Generate natural, contextual responses (like having a eloquent speaker)
- Handle thousands of simultaneous users (like having an army of assistants)
- Track usage and enforce limits fairly (like having a smart billing department)
- Protect sensitive information (like having a security team)

The system achieves this through a carefully orchestrated dance of multiple specialized components, each optimized for its specific role. Let's explore how these pieces work together.

### **Your Learning Path**

| Your Role | Start Here | Core Sections | Advanced Topics |
|-----------|------------|---------------|-----------------|
| **AI Engineer** | Section 3: Understanding RAG | 4: AI/ML Pipeline, 5: Service Implementation | 11: Performance Optimization |
| **System Architect** | Section 2: System Architecture | 6: Data Design, 8: Security | 12: Operations & Reliability |
| **Data Architect** | Section 6: Data Design | 3: RAG Pipeline, 7: API Design | 13: Scaling Strategies |
| **Data Scientist** | Section 4: ML Models | 3: RAG Pipeline, 11: Performance | 14: Evaluation & Testing |
| **New Developer** | Section 1: Quickstart | 9: Getting Started, 10: First Deployment | 15: Troubleshooting |

---

## **📚 Table of Contents**

1. **[Quickstart: Your First Hour](#1-quickstart-your-first-hour)** - Get the system running
2. **[System Architecture: The Big Picture](#2-system-architecture-the-big-picture)** - Understanding the design
3. **[Understanding RAG: The Core Innovation](#3-understanding-rag-the-core-innovation)** - How RAG works
4. **[AI/ML Pipeline: The Intelligence Layer](#4-aiml-pipeline-the-intelligence-layer)** - Models and processing
5. **[Service Implementation: Building Blocks](#5-service-implementation-building-blocks)** - Core services
6. **[Data Design: The Foundation](#6-data-design-the-foundation)** - Database architecture
7. **[API Design: The Interface](#7-api-design-the-interface)** - Endpoints and contracts
8. **[Security & Governance: Protection Layers](#8-security--governance-protection-layers)** - Auth, RBAC, compliance
9. **[Getting Started: Setup Guide](#9-getting-started-setup-guide)** - Detailed installation
10. **[Production Deployment: Going Live](#10-production-deployment-going-live)** - Kubernetes and scaling
11. **[Performance Optimization: Speed & Efficiency](#11-performance-optimization-speed--efficiency)** - Tuning guide
12. **[Operations & Reliability: Keeping It Running](#12-operations--reliability-keeping-it-running)** - SRE practices
13. **[Scaling Strategies: Growing the System](#13-scaling-strategies-growing-the-system)** - Horizontal and vertical
14. **[Testing & Evaluation: Quality Assurance](#14-testing--evaluation-quality-assurance)** - Test strategies
15. **[Troubleshooting: When Things Go Wrong](#15-troubleshooting-when-things-go-wrong)** - Common issues
16. **[Cost Management: Efficiency & Budgeting](#16-cost-management-efficiency--budgeting)** - Cost optimization
17. **[Roadmap & Evolution: What's Next](#17-roadmap--evolution-whats-next)** - Future enhancements

---

## **1. Quickstart: Your First Hour**

Let's get you up and running quickly. This section will have you chatting with the system in under an hour, giving you a tangible feel for what we're building before we dive into the details.

### **Prerequisites Check**

First, ensure you have these tools installed. Think of them as your workshop tools:

```bash
# Check if you have the required tools
docker --version          # Should be 20.10+
docker-compose --version   # Should be 2.0+
python --version          # Should be 3.11+
git --version            # Any recent version

# Check available resources
# You need at least 16GB RAM and 50GB disk space
df -h                    # Check disk space
free -h                  # Check memory (Linux/Mac)
```

### **Quick Setup (15 minutes)**

Now let's build your development environment. This is like setting up your workspace:

```bash
# 1. Clone the repository - getting the blueprints
git clone https://github.com/your-org/rag-chatbot-platform.git
cd rag-chatbot-platform

# 2. Create your configuration - personalizing your setup
cp .env.example .env
# Edit .env with your favorite editor
# For now, the defaults will work

# 3. Start the infrastructure - powering up the engines
docker-compose up -d

# This starts:
# - PostgreSQL (your filing cabinet for users and billing)
# - MongoDB (your library for documents and vectors)
# - Redis (your short-term memory for caching)
# - ScyllaDB (your conversation history ledger)

# 4. Wait for services to be healthy
docker-compose ps
# All services should show "healthy" status

# 5. Install Python dependencies - getting your tools ready
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 6. Initialize the databases - setting up the storage
python scripts/init_all_databases.py

# 7. Seed sample data - adding some initial knowledge
python scripts/seed_sample_data.py

# 8. Start the application - launching the system
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### **Your First Conversation (5 minutes)**

Now let's have your first conversation with the system:

```bash
# 1. Register yourself as a user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "you@example.com",
    "password": "YourSecurePassword123!"
  }'

# 2. Login to get your access token
TOKEN=$(curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "you@example.com",
    "password": "YourSecurePassword123!"
  }' | jq -r '.access_token')

# 3. Ask your first question
curl -X POST http://localhost:8000/chat/message \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is RAG and how does it work?",
    "enable_rag": true
  }'

# You should receive a detailed answer with citations!
```

### **Understanding What Just Happened**

When you sent that chat message, here's the journey it took through the system:

1. **Authentication**: Your token was verified, confirming you're a valid user
2. **Rate Limiting**: The system checked you haven't exceeded request limits
3. **Quota Check**: Your subscription plan's message allowance was verified
4. **Query Processing**: Your question was analyzed and converted to a vector
5. **Retrieval**: Relevant documents were found using vector similarity
6. **Generation**: An LLM created an answer using the retrieved context
7. **Response**: You received an answer with citations to source documents

This entire process happened in under 3 seconds! Now let's understand how each piece works.

---

## **2. System Architecture: The Big Picture**

### **2.1 Guiding Principles**

Before we examine the architecture, let's understand the core principles that guide every design decision. These principles are like the constitution of our system:

#### **Reliability First**
The system prioritizes reliability over features. This means:
- **Fail closed on security**: When in doubt about authentication or authorization, deny access
- **Fail open with guardrails on features**: If retrieval fails, provide a degraded but safe response
- **Graceful degradation**: When a component fails, the system continues with reduced functionality

#### **Separation of Concerns**
Each service has a single, well-defined responsibility. Think of it like a restaurant:
- The host (API Gateway) greets and seats customers
- The waiter (Chat Service) takes orders and coordinates
- The chef (Generation Service) prepares the meal
- The sommelier (Knowledge Service) recommends pairings

#### **Traceable Decisions**
Every response includes metadata about how it was created:
- Which documents were retrieved
- What ranking algorithm was used
- Which model generated the response
- How many tokens were consumed

#### **Runtime Configurability**
Critical behaviors can be adjusted without redeploying:
- Feature flags control experimental features
- Model routing can switch between providers
- Rate limits and quotas can be tuned
- Cache strategies can be modified

### **2.2 High-Level Architecture**

Let's visualize the complete system architecture. Think of this as the blueprint of a complex building:

```mermaid
flowchart TB
    subgraph "Client Layer - The Front Door"
        WEB[Web Application<br/>React/Next.js]
        MOBILE[Mobile Apps<br/>iOS/Android]
        API_CLIENT[API Clients<br/>Python/JS SDKs]
        PARTNERS[Partner Integrations]
    end
    
    subgraph "Edge Layer - The Reception"
        CDN[CDN<br/>Static Assets]
        WAF[WAF<br/>DDoS Protection]
        LB[Load Balancer<br/>Traffic Distribution]
        GATEWAY[API Gateway<br/>Request Routing]
    end
    
    subgraph "Middleware Layer - The Security Checkpoint"
        AUTH_MW[Authentication<br/>JWT Validation]
        RATE_MW[Rate Limiter<br/>Request Throttling]
        QUOTA_MW[Quota Checker<br/>Usage Limits]
        ADMIN_MW[Admin Validator<br/>Privilege Check]
        FEATURE_MW[Feature Flags<br/>A/B Testing]
    end
    
    subgraph "Service Layer - The Workers"
        subgraph "Core Services"
            AUTH_SVC[Auth Service<br/>Identity Management]
            USER_SVC[User Service<br/>Profile Management]
            BILLING_SVC[Billing Service<br/>Subscriptions & Usage]
            AUDIT_SVC[Audit Service<br/>Compliance Logging]
        end
        
        subgraph "AI Services"
            CHAT_SVC[Chat Orchestrator<br/>Conversation Management]
            KNOWLEDGE_SVC[Knowledge Service<br/>Information Retrieval]
            EMBED_SVC[Embedding Service<br/>Vector Generation]
            GEN_SVC[Generation Service<br/>Text Creation]
            RANK_SVC[Ranking Service<br/>Result Ordering]
        end
        
        subgraph "Data Services"
            INGEST_SVC[Ingestion Service<br/>Document Processing]
            INDEX_SVC[Indexing Service<br/>Search Optimization]
            EXPORT_SVC[Export Service<br/>Data Extraction]
        end
    end
    
    subgraph "Data Layer - The Storage"
        subgraph "Primary Stores"
            PG[(PostgreSQL<br/>Users, Billing, Audit)]
            MONGO[(MongoDB Atlas<br/>Documents, Vectors)]
        end
        
        subgraph "Performance Stores"
            REDIS[(Redis<br/>Cache, Sessions, Queues)]
            SCYLLA[(ScyllaDB<br/>Conversations, Analytics)]
        end
        
        subgraph "Object Storage"
            S3[S3/Blob<br/>Raw Documents, Models]
        end
    end
    
    subgraph "Infrastructure - The Foundation"
        K8S[Kubernetes<br/>Container Orchestration]
        ISTIO[Service Mesh<br/>Traffic Management]
        MONITOR[Monitoring<br/>Prometheus/Grafana]
        LOG[Logging<br/>ELK Stack]
    end
    
    WEB --> CDN --> WAF --> LB
    MOBILE --> WAF --> LB
    API_CLIENT --> LB
    PARTNERS --> LB
    
    LB --> GATEWAY
    GATEWAY --> AUTH_MW
    AUTH_MW --> RATE_MW
    RATE_MW --> QUOTA_MW
    QUOTA_MW --> ADMIN_MW
    ADMIN_MW --> FEATURE_MW
    
    FEATURE_MW --> CHAT_SVC
    
    CHAT_SVC --> KNOWLEDGE_SVC
    KNOWLEDGE_SVC --> EMBED_SVC
    KNOWLEDGE_SVC --> RANK_SVC
    CHAT_SVC --> GEN_SVC
    
    AUTH_MW --> AUTH_SVC
    QUOTA_MW --> BILLING_SVC
    
    AUTH_SVC --> PG
    USER_SVC --> PG
    BILLING_SVC --> PG
    AUDIT_SVC --> PG
    
    KNOWLEDGE_SVC --> MONGO
    EMBED_SVC --> REDIS
    CHAT_SVC --> SCYLLA
    
    INGEST_SVC --> S3
    INGEST_SVC --> MONGO
    
    K8S --> MONITOR
    K8S --> LOG
```

### **2.3 Request Lifecycle - Following a Query**

Let's trace a single request through the system to understand how all components work together. This is like following a package through a delivery system:

```mermaid
sequenceDiagram
    participant U as User
    participant G as Gateway
    participant A as Auth
    participant R as Rate Limiter
    participant Q as Quota
    participant C as Chat Service
    participant K as Knowledge
    participant E as Embedding
    participant V as Vector DB
    participant L as LLM
    participant H as History
    
    Note over U,H: User asks "What is machine learning?"
    
    U->>G: POST /chat {"message": "What is ML?"}
    G->>A: Validate JWT token
    A-->>G: ✓ User verified
    
    G->>R: Check rate limit
    R-->>G: ✓ Within limits (5/60 remaining)
    
    G->>Q: Check message quota
    Q-->>G: ✓ Has quota (95/100 used)
    
    G->>C: Process message
    
    Note over C,L: RAG Pipeline Begins
    
    C->>E: Generate embedding
    E-->>C: [0.23, -0.15, ...] (768D)
    
    C->>K: Retrieve relevant docs
    K->>V: Vector search
    V-->>K: Top 5 documents
    K-->>C: Ranked context
    
    C->>L: Generate response
    Note right of L: Prompt includes:<br/>- System instructions<br/>- Retrieved context<br/>- User question
    L-->>C: Generated answer
    
    C->>H: Store conversation
    H-->>C: ✓ Saved
    
    C-->>G: Response + metadata
    G-->>U: {"answer": "...", "sources": [...]}
    
    Note over U,H: Total time: ~2.5 seconds
```

### **2.4 Service Boundaries and Responsibilities**

Understanding service boundaries is crucial. Each service is like a specialized department in a company:

#### **Core Services**

**Auth Service** - The Security Department
- Issues and validates JWT tokens
- Manages user sessions
- Handles password resets
- Enforces organization boundaries
- Tracks login attempts and security events

**Billing Service** - The Finance Department
- Tracks usage across all resources
- Enforces subscription limits
- Processes plan upgrades/downgrades
- Generates usage reports
- Manages payment integration

**User Service** - The HR Department
- Manages user profiles
- Handles preferences
- Manages organization membership
- Tracks user activity

#### **AI Services**

**Chat Orchestrator** - The Conductor
- Coordinates the entire conversation flow
- Enforces business rules
- Manages conversation state
- Handles streaming responses
- Records usage metrics

**Knowledge Service** - The Librarian
- Classifies queries
- Executes search strategies
- Combines multiple retrieval methods
- Ranks and filters results

**Embedding Service** - The Translator
- Converts text to vectors
- Maintains embedding consistency
- Handles batch processing
- Manages model versioning

**Generation Service** - The Writer
- Manages LLM interactions
- Constructs prompts
- Handles token budgeting
- Implements safety filters

---

## **3. Understanding RAG: The Core Innovation**

### **3.1 What is RAG?**

RAG (Retrieval-Augmented Generation) is like giving an AI assistant access to a vast library before answering questions. Instead of relying solely on what the model learned during training, RAG:

1. **Retrieves** relevant information from your documents
2. **Augments** the AI's prompt with this context
3. **Generates** an informed, accurate response

Think of it this way: Without RAG, asking an AI model a question is like asking a very knowledgeable person to answer from memory alone. With RAG, it's like allowing them to consult relevant books, documents, and notes before responding.

### **3.2 The RAG Pipeline**

Let's explore each stage of the RAG pipeline in detail:

```mermaid
flowchart LR
    subgraph "1. Query Understanding"
        Q[User Query] --> QA[Query Analysis]
        QA --> QC[Query Classification]
        QC --> QE[Query Enhancement]
    end
    
    subgraph "2. Retrieval"
        QE --> RT{Routing Decision}
        RT -->|Simple| LEX[Lexical Search]
        RT -->|Semantic| VEC[Vector Search]
        RT -->|Complex| HYB[Hybrid Search]
        
        LEX --> RET[Retrieved Docs]
        VEC --> RET
        HYB --> RET
    end
    
    subgraph "3. Ranking & Fusion"
        RET --> RRF[RRF Fusion]
        RRF --> RANK[Re-ranking]
        RANK --> FILT[Filtering]
    end
    
    subgraph "4. Context Building"
        FILT --> CTX[Context Assembly]
        CTX --> TRUNC[Token Budgeting]
        TRUNC --> PROMPT[Prompt Construction]
    end
    
    subgraph "5. Generation"
        PROMPT --> LLM[LLM Processing]
        LLM --> POST[Post-processing]
        POST --> RESP[Final Response]
    end
    
    RESP --> U[User Gets Answer]
```

#### **Stage 1: Query Understanding**

First, we need to understand what the user is really asking:

```python
class QueryAnalyzer:
    """
    Analyzes and classifies user queries to determine the best retrieval strategy.
    Think of this as understanding the user's intent.
    """
    
    def analyze(self, query: str) -> QueryAnalysis:
        # Extract key entities and concepts
        entities = self.extract_entities(query)  # "machine learning" -> ML, AI
        
        # Determine query type
        if len(query.split()) <= 3 and "?" not in query:
            query_type = QueryType.KEYWORD  # "Python tutorial"
        elif self.is_factual(query):
            query_type = QueryType.FACTUAL  # "When was Python created?"
        elif self.requires_reasoning(query):
            query_type = QueryType.COMPLEX  # "Compare Python vs JavaScript for ML"
        else:
            query_type = QueryType.SEMANTIC  # Most general questions
        
        # Enhance query with synonyms and related terms
        enhanced_terms = self.expand_query(query)  # "ML" -> ["machine learning", "AI", "deep learning"]
        
        return QueryAnalysis(
            original=query,
            type=query_type,
            entities=entities,
            enhanced_terms=enhanced_terms,
            suggested_strategy=self.suggest_strategy(query_type)
        )
```

#### **Stage 2: Retrieval Strategies**

Based on the query type, we use different retrieval strategies:

**Lexical Search** - Finding exact matches
- Best for: Known terms, product names, specific phrases
- Method: BM25, TF-IDF
- Example: "Error code 404" finds documents with that exact phrase

**Vector Search** - Finding semantic similarity
- Best for: Conceptual questions, paraphrased queries
- Method: Cosine similarity in embedding space
- Example: "How do I teach a computer to recognize images?" finds ML/CV documents

**Hybrid Search** - Best of both worlds
- Best for: Complex queries requiring both precision and recall
- Method: Parallel lexical + vector, then fusion
- Example: "Debug Python memory leak in TensorFlow" needs both exact terms and concepts

```python
class RetrievalRouter:
    """
    Routes queries to appropriate retrieval strategies.
    Like choosing the right tool for the job.
    """
    
    async def retrieve(self, query: str, analysis: QueryAnalysis) -> List[Document]:
        if analysis.type == QueryType.KEYWORD:
            # Use lexical search for exact matches
            return await self.lexical_search(query)
            
        elif analysis.type == QueryType.SEMANTIC:
            # Use vector search for conceptual queries
            embedding = await self.embed(query)
            return await self.vector_search(embedding)
            
        else:  # COMPLEX or FACTUAL
            # Use hybrid approach for best coverage
            lexical_task = asyncio.create_task(self.lexical_search(query))
            vector_task = asyncio.create_task(self.vector_search(await self.embed(query)))
            
            lexical_results, vector_results = await asyncio.gather(
                lexical_task, vector_task
            )
            
            # Combine using Reciprocal Rank Fusion
            return self.reciprocal_rank_fusion(
                [lexical_results, vector_results],
                k=60  # RRF constant - higher values favor high-ranked docs
            )
```

#### **Stage 3: Ranking and Fusion**

When we have results from multiple sources, we need to combine and rank them intelligently:

```python
def reciprocal_rank_fusion(self, result_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) combines multiple ranked lists.
    
    The idea: If a document appears high in multiple lists, it's likely relevant.
    Formula: score(d) = Σ(1 / (k + rank(d)))
    
    Think of it like a voting system where each retrieval method votes for documents,
    with higher-ranked documents getting more voting power.
    """
    
    fusion_scores = {}
    
    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = doc.id
            
            if doc_id not in fusion_scores:
                fusion_scores[doc_id] = {
                    "score": 0,
                    "document": doc,
                    "appearances": []
                }
            
            # Add reciprocal rank score
            # The constant k controls how much we favor top-ranked documents
            # Higher k = more democratic (less difference between ranks)
            # Lower k = more elitist (huge advantage to top ranks)
            fusion_scores[doc_id]["score"] += 1 / (k + rank)
            fusion_scores[doc_id]["appearances"].append({
                "source": result_list.source,
                "rank": rank
            })
    
    # Sort by fusion score
    ranked_results = sorted(
        fusion_scores.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    return [r["document"] for r in ranked_results]
```

#### **Stage 4: Context Building**

Now we need to build the context that will be provided to the LLM:

```python
class ContextBuilder:
    """
    Builds optimal context for the LLM within token constraints.
    Like preparing a briefing for someone who needs to answer a question.
    """
    
    def build_context(
        self,
        query: str,
        documents: List[Document],
        max_tokens: int = 2000
    ) -> Context:
        # Start with most relevant documents
        selected_docs = []
        total_tokens = 0
        
        for doc in documents:
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            doc_tokens = len(doc.content) // 4
            
            if total_tokens + doc_tokens <= max_tokens:
                selected_docs.append(doc)
                total_tokens += doc_tokens
            else:
                # Try to fit a truncated version
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Minimum useful chunk
                    truncated_content = self.truncate_to_tokens(
                        doc.content,
                        remaining_tokens
                    )
                    doc.content = truncated_content
                    selected_docs.append(doc)
                    break
        
        return Context(
            documents=selected_docs,
            total_tokens=total_tokens,
            truncation_applied=len(documents) > len(selected_docs)
        )
```

#### **Stage 5: Generation with Context**

Finally, we construct the prompt and generate the response:

```python
class PromptConstructor:
    """
    Constructs effective prompts for the LLM.
    This is like giving clear instructions to an expert.
    """
    
    def construct_rag_prompt(
        self,
        query: str,
        context: Context,
        system_instructions: str = None
    ) -> str:
        prompt_parts = []
        
        # System instructions set the tone and rules
        if system_instructions:
            prompt_parts.append(f"System: {system_instructions}")
        else:
            prompt_parts.append(
                "System: You are a helpful AI assistant. "
                "Answer questions based on the provided context. "
                "If the context doesn't contain relevant information, say so. "
                "Always cite your sources."
            )
        
        # Add context with clear formatting
        prompt_parts.append("\n## Relevant Context:")
        
        for i, doc in enumerate(context.documents, 1):
            prompt_parts.append(
                f"\n### Source {i} (Relevance: {doc.score:.2f})"
                f"\nFrom: {doc.metadata.get('source', 'Unknown')}"
                f"\n{doc.content}"
            )
        
        # Add the user's question
        prompt_parts.append(f"\n## User Question:\n{query}")
        
        # Request format
        prompt_parts.append(
            "\n## Your Response:\n"
            "Please provide a comprehensive answer based on the context above. "
            "Include source numbers [1], [2], etc. for your citations."
        )
        
        return "\n".join(prompt_parts)
```

### **3.3 Why RAG Works So Well**

RAG solves several fundamental problems with pure LLM approaches:

1. **Hallucination Reduction**: By grounding responses in retrieved documents, we reduce the model's tendency to make things up

2. **Knowledge Currency**: RAG can access information that didn't exist when the model was trained

3. **Domain Specificity**: You can build expertise in your specific domain without retraining models

4. **Transparency**: Every answer can cite its sources, building trust

5. **Cost Efficiency**: Cheaper than fine-tuning large models for every use case

---

## **4. AI/ML Pipeline: The Intelligence Layer**

### **4.1 Model Architecture Overview**

Our AI pipeline uses specialized models for different tasks, like having different experts for different jobs:

```mermaid
graph TB
    subgraph "Embedding Models"
        E1[all-mpnet-base-v2<br/>768D, General Purpose]
        E2[all-MiniLM-L6-v2<br/>384D, Fast]
        E3[Custom Domain Model<br/>768D, Specialized]
    end
    
    subgraph "Generation Models"
        G1[Qwen3-1.7B<br/>Primary, Balanced]
        G2[Llama-3-8B<br/>High Quality]
        G3[Mistral-7B<br/>Fast Response]
    end
    
    subgraph "Ranking Models"
        R1[cross-encoder/ms-marco<br/>Re-ranking]
        R2[ColBERT<br/>Efficient Ranking]
    end
    
    subgraph "Specialized Models"
        S1[BERT Classifier<br/>Query Intent]
        S2[NER Model<br/>Entity Extraction]
        S3[Sentiment Model<br/>Tone Analysis]
    end
```

### **4.2 Embedding Service Deep Dive**

The embedding service converts text into numerical vectors that capture semantic meaning:

```python
class EmbeddingService:
    """
    Production-grade embedding service with multiple optimizations.
    Think of this as a universal translator that converts any text
    into a mathematical representation that computers can compare.
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = self._load_model()
        
        # Optimization: Cache frequently used embeddings
        self.cache = LRUCache(maxsize=10000)
        
        # Optimization: Batch queue for efficiency
        self.batch_queue = asyncio.Queue()
        self.batch_processor = None
        
    def _load_model(self):
        """
        Load and optimize the model for production use.
        """
        model = SentenceTransformer(self.config.model_name)
        
        # Set optimal sequence length
        model.max_seq_length = 512
        
        # Enable mixed precision for faster inference
        if self.config.device != "cpu":
            model = model.half()  # FP16 for GPU
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        return model
    
    async def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        The process:
        1. Check cache for existing embedding
        2. Normalize and clean text
        3. Generate embedding
        4. Post-process (normalize to unit length)
        5. Cache result
        """
        
        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Clean text (remove excess whitespace, special chars)
        cleaned_text = self._clean_text(text)
        
        # Generate embedding
        embedding = await self._generate_embedding(cleaned_text)
        
        # Normalize to unit length for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        # Cache result
        if use_cache:
            self.cache[cache_key] = embedding
        
        return embedding
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Efficiently embed multiple texts.
        
        Optimizations:
        - Process in optimal batch sizes
        - Use GPU parallelism
        - Minimize memory transfers
        """
        
        embeddings = []
        
        # Process in batches for memory efficiency
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            
            # Clean all texts in batch
            cleaned_batch = [self._clean_text(t) for t in batch]
            
            # Generate embeddings in parallel on GPU
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    cleaned_batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _clean_text(self, text: str) -> str:
        """
        Prepare text for embedding.
        
        Cleaning steps:
        1. Remove extra whitespace
        2. Handle special characters
        3. Truncate to max length
        4. Ensure minimum length
        """
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove problematic special characters
        text = text.replace('\u0000', '')  # Null characters
        
        # Ensure text isn't too long (truncate at word boundary)
        if len(text) > self.config.max_length:
            text = text[:self.config.max_length].rsplit(' ', 1)[0]
        
        # Ensure text isn't too short
        if len(text) < 3:
            text = text + " [empty]"
        
        return text
```

### **4.3 Generation Service Architecture**

The generation service manages interactions with Large Language Models:

```python
class GenerationService:
    """
    Manages text generation with multiple models and strategies.
    Like a writing team that can adapt its style and approach.
    """
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.models = self._load_models()
        self.token_counter = self._init_tokenizer()
        
    async def generate(
        self,
        prompt: str,
        model_name: str = None,
        stream: bool = False,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text with comprehensive controls.
        
        Parameters explained:
        - prompt: The complete prompt including context
        - model_name: Override default model selection
        - stream: Enable token-by-token streaming
        - kwargs: Model-specific parameters (temperature, top_p, etc.)
        """
        
        # Select model based on request characteristics
        model = self._select_model(model_name, prompt)
        
        # Calculate token budget
        prompt_tokens = self.count_tokens(prompt)
        max_new_tokens = min(
            kwargs.get('max_tokens', 512),
            self.config.total_token_limit - prompt_tokens
        )
        
        # Apply safety constraints
        generation_params = self._apply_safety_constraints({
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'top_k': kwargs.get('top_k', 40),
            'max_new_tokens': max_new_tokens,
            'repetition_penalty': kwargs.get('repetition_penalty', 1.05),
            'stop_sequences': kwargs.get('stop_sequences', [])
        })
        
        # Generate response
        if stream:
            return self._stream_generation(model, prompt, generation_params)
        else:
            return await self._generate_complete(model, prompt, generation_params)
    
    def _select_model(self, requested_model: str, prompt: str) -> Model:
        """
        Intelligent model selection based on requirements.
        
        Selection criteria:
        - User preference (if specified)
        - Prompt length (longer prompts → models with larger context)
        - Required quality (production → higher quality models)
        - Latency requirements (real-time → faster models)
        - Cost constraints (budget-limited → efficient models)
        """
        
        if requested_model and requested_model in self.models:
            return self.models[requested_model]
        
        prompt_length = len(prompt)
        
        if prompt_length > 8000:
            # Need model with large context window
            return self.models['qwen-32k']
        elif prompt_length < 1000 and self.config.optimize_for_speed:
            # Can use faster, smaller model
            return self.models['mistral-7b']
        else:
            # Default balanced choice
            return self.models['qwen3-1.7b']
    
    def _apply_safety_constraints(self, params: dict) -> dict:
        """
        Apply safety limits to generation parameters.
        
        Constraints explained:
        - Temperature: Cap at 1.0 to prevent too random outputs
        - Top_p: Minimum 0.1 to avoid deterministic loops
        - Max tokens: Enforce budget limits
        - Stop sequences: Add safety stops
        """
        
        # Temperature: Controls randomness (0=deterministic, 1=creative)
        params['temperature'] = min(params['temperature'], 1.0)
        params['temperature'] = max(params['temperature'], 0.0)
        
        # Top_p: Nucleus sampling (smaller = more focused)
        params['top_p'] = min(params['top_p'], 1.0)
        params['top_p'] = max(params['top_p'], 0.1)
        
        # Add safety stop sequences
        safety_stops = [
            "\nUser:",
            "\nHuman:",
            "<|endoftext|>",
            "```"  # Prevent code injection attempts
        ]
        params['stop_sequences'].extend(safety_stops)
        
        return params
    
    async def _generate_complete(
        self,
        model: Model,
        prompt: str,
        params: dict
    ) -> GenerationResult:
        """
        Generate complete response at once.
        """
        
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = model.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model.config.max_position_embeddings
            )
            
            # Move to correct device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate with controlled decoding
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    **params,
                    pad_token_id=model.tokenizer.eos_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    use_cache=True  # Speed up generation
                )
            
            # Decode response
            response = model.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):],
                skip_special_tokens=True
            )
            
            # Calculate metrics
            generation_time = time.time() - start_time
            output_tokens = len(outputs[0]) - len(inputs['input_ids'][0])
            
            return GenerationResult(
                text=response.strip(),
                model_used=model.name,
                prompt_tokens=len(inputs['input_ids'][0]),
                completion_tokens=output_tokens,
                total_tokens=len(outputs[0]),
                generation_time_ms=generation_time * 1000,
                tokens_per_second=output_tokens / generation_time if generation_time > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # Return safe fallback response
            return GenerationResult(
                text="I apologize, but I'm unable to generate a response at this moment. Please try again.",
                error=str(e),
                model_used=model.name
            )
```

### **4.4 Model Optimization Strategies**

Let's explore how we optimize model performance for production:

```python
class ModelOptimizer:
    """
    Optimizes models for production deployment.
    Like tuning a race car for maximum performance.
    """
    
    @staticmethod
    def optimize_for_inference(model, config: OptimizationConfig):
        """
        Apply various optimization techniques to improve inference speed.
        """
        
        optimized_model = model
        
        # 1. Quantization: Reduce precision for faster computation
        if config.enable_quantization:
            # INT8 quantization can give 2-4x speedup with minimal accuracy loss
            optimized_model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},  # Quantize linear layers
                dtype=torch.qint8
            )
        
        # 2. Graph optimization: Fuse operations
        if config.enable_graph_optimization:
            optimized_model = torch.jit.script(model)
            optimized_model = torch.jit.optimize_for_inference(optimized_model)
        
        # 3. Mixed precision: Use FP16 where possible
        if config.enable_mixed_precision and torch.cuda.is_available():
            optimized_model = optimized_model.half()
        
        # 4. Batching optimization: Process multiple inputs together
        if config.enable_dynamic_batching:
            optimized_model = DynamicBatchingWrapper(
                optimized_model,
                max_batch_size=config.max_batch_size,
                timeout_ms=config.batching_timeout
            )
        
        # 5. Caching: Cache intermediate computations
        if config.enable_kv_cache:
            optimized_model = KVCacheWrapper(optimized_model)
        
        return optimized_model
```

---

## **5. Service Implementation: Building Blocks**

Now let's dive deep into the implementation of our core services. Each service is designed with production requirements in mind.

### **5.1 Chat Orchestrator Service**

The Chat Orchestrator is the conductor of our symphony, coordinating all other services:

```python
class ChatOrchestrator:
    """
    Orchestrates the entire chat interaction flow.
    This is the brain that coordinates all other services to produce a response.
    """
    
    def __init__(
        self,
        knowledge_service: KnowledgeService,
        generation_service: GenerationService,
        billing_service: BillingService,
        cache_service: CacheService,
        history_service: HistoryService
    ):
        self.knowledge = knowledge_service
        self.generation = generation_service
        self.billing = billing_service
        self.cache = cache_service
        self.history = history_service
        
    async def process_message(
        self,
        user_id: str,
        message: str,
        session_id: str = None,
        enable_rag: bool = True,
        stream: bool = False
    ) -> ChatResponse:
        """
        Process a user message through the complete pipeline.
        
        The journey of a message:
        1. Validation and setup
        2. Cache check
        3. RAG retrieval (if enabled)
        4. Generation
        5. Post-processing
        6. Storage and billing
        """
        
        # Create or retrieve session
        session = await self._get_or_create_session(user_id, session_id)
        
        # Check cache for recent similar questions
        cache_key = self._generate_cache_key(message, enable_rag)
        cached_response = await self.cache.get(cache_key)
        
        if cached_response and not stream:
            # Return cached response with metadata
            return ChatResponse(
                answer=cached_response['answer'],
                sources=cached_response.get('sources', []),
                cached=True,
                session_id=session.id
            )
        
        # Perform RAG if enabled
        context = None
        sources = []
        
        if enable_rag:
            # Retrieve relevant context
            retrieval_result = await self.knowledge.retrieve(
                query=message,
                user_id=user_id,
                top_k=8,
                filters={'user_accessible': True}
            )
            
            context = retrieval_result.documents
            sources = self._format_sources(retrieval_result.documents)
        
        # Generate response
        generation_params = {
            'stream': stream,
            'max_tokens': 512,
            'temperature': 0.7
        }
        
        if stream:
            # Return streaming response
            return self._create_streaming_response(
                message, context, generation_params, session
            )
        else:
            # Generate complete response
            prompt = self._build_prompt(message, context)
            result = await self.generation.generate(prompt, **generation_params)
            
            # Post-process response
            processed_answer = self._post_process_answer(
                result.text,
                sources
            )
            
            # Store in history (async, don't wait)
            asyncio.create_task(
                self.history.store_interaction(
                    session_id=session.id,
                    user_message=message,
                    assistant_response=processed_answer,
                    context_used=context,
                    metadata={
                        'generation_time_ms': result.generation_time_ms,
                        'tokens_used': result.total_tokens,
                        'model': result.model_used
                    }
                )
            )
            
            # Record usage for billing (async, don't wait)
            asyncio.create_task(
                self.billing.record_usage(
                    user_id=user_id,
                    resource='messages',
                    quantity=1,
                    metadata={'tokens': result.total_tokens}
                )
            )
            
            # Cache response
            await self.cache.set(
                cache_key,
                {
                    'answer': processed_answer,
                    'sources': sources
                },
                ttl=3600  # 1 hour
            )
            
            return ChatResponse(
                answer=processed_answer,
                sources=sources,
                session_id=session.id,
                tokens_used=result.total_tokens,
                generation_time_ms=result.generation_time_ms,
                model_used=result.model_used,
                cached=False
            )
    
    def _build_prompt(self, message: str, context: List[Document] = None) -> str:
        """
        Build an effective prompt for the LLM.
        
        Prompt engineering principles:
        1. Clear system instructions
        2. Structured context presentation
        3. Explicit output requirements
        4. Safety boundaries
        """
        
        prompt_parts = []
        
        # System instructions
        prompt_parts.append(
            "You are a helpful AI assistant. Your responses should be:\n"
            "- Accurate and based on the provided context\n"
            "- Clear and well-structured\n"
            "- Honest about limitations\n"
            "- Respectful and professional\n"
        )
        
        # Add context if available
        if context:
            prompt_parts.append("\n## Relevant Information:\n")
            
            for i, doc in enumerate(context[:5], 1):  # Limit to top 5
                prompt_parts.append(
                    f"\n[{i}] {doc.title or 'Document'}\n"
                    f"Relevance: {doc.score:.2%}\n"
                    f"Content: {doc.content[:500]}...\n"  # Truncate long content
                )
        
        # Add user question
        prompt_parts.append(f"\n## User Question:\n{message}\n")
        
        # Response instructions
        prompt_parts.append(
            "\n## Your Response:\n"
            "Please provide a helpful answer. "
            "If you use information from the context, cite it with [1], [2], etc.\n"
        )
        
        return "\n".join(prompt_parts)
    
    def _post_process_answer(self, answer: str, sources: List[Source]) -> str:
        """
        Clean and enhance the generated answer.
        
        Post-processing steps:
        1. Remove any inappropriate content
        2. Fix formatting issues
        3. Add source citations if missing
        4. Ensure response completeness
        """
        
        # Remove any potential prompt leakage
        answer = self._remove_prompt_artifacts(answer)
        
        # Ensure citations are properly formatted
        if sources and '[1]' not in answer and any(sources):
            # Add a citation note at the end
            answer += "\n\n*Note: This response is based on the provided sources.*"
        
        # Ensure response isn't cut off
        if not answer.endswith(('.', '!', '?', '"', '\n')):
            # Response might be truncated, add indicator
            answer += "..."
        
        return answer.strip()
```

### **5.2 Knowledge Service Implementation**

The Knowledge Service manages all aspects of information retrieval:

```python
class KnowledgeService:
    """
    Manages document retrieval and ranking.
    This is our intelligent librarian that knows how to find the right information.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        lexical_store: LexicalStore,
        embedding_service: EmbeddingService,
        ranker: Ranker = None
    ):
        self.vector_store = vector_store
        self.lexical_store = lexical_store
        self.embedding = embedding_service
        self.ranker = ranker or DefaultRanker()
        
        # Query classifier for routing
        self.classifier = QueryClassifier()
        
    async def retrieve(
        self,
        query: str,
        user_id: str = None,
        top_k: int = 10,
        filters: dict = None,
        strategy: str = "auto"
    ) -> RetrievalResult:
        """
        Main retrieval method with intelligent routing.
        
        Strategies explained:
        - "auto": System decides based on query analysis
        - "vector": Pure semantic search
        - "lexical": Exact term matching
        - "hybrid": Both methods with fusion
        """
        
        # Analyze query to determine best strategy
        if strategy == "auto":
            strategy = await self._determine_strategy(query)
        
        # Apply user-specific filters
        if user_id:
            filters = self._add_user_filters(user_id, filters)
        
        # Execute retrieval based on strategy
        if strategy == "vector":
            results = await self._vector_retrieval(query, top_k, filters)
        elif strategy == "lexical":
            results = await self._lexical_retrieval(query, top_k, filters)
        else:  # hybrid
            results = await self._hybrid_retrieval(query, top_k, filters)
        
        # Re-rank if we have a ranker model
        if self.ranker and len(results) > 0:
            results = await self._rerank_results(query, results)
        
        # Build retrieval result with metadata
        return RetrievalResult(
            documents=results,
            strategy_used=strategy,
            total_retrieved=len(results),
            filters_applied=filters
        )
    
    async def _vector_retrieval(
        self,
        query: str,
        top_k: int,
        filters: dict = None
    ) -> List[Document]:
        """
        Semantic search using vector similarity.
        
        The process:
        1. Generate query embedding
        2. Search vector index
        3. Apply filters
        4. Return top-k results
        """
        
        # Generate query embedding
        query_embedding = await self.embedding.embed_text(query)
        
        # Search vector store
        search_results = await self.vector_store.search(
            vector=query_embedding,
            top_k=top_k * 2,  # Over-retrieve for filtering
            filters=filters
        )
        
        # Convert to documents with scores
        documents = []
        for result in search_results[:top_k]:
            doc = Document(
                id=result['id'],
                content=result['content'],
                title=result.get('title'),
                score=result['score'],
                metadata=result.get('metadata', {})
            )
            documents.append(doc)
        
        return documents
    
    async def _hybrid_retrieval(
        self,
        query: str,
        top_k: int,
        filters: dict = None
    ) -> List[Document]:
        """
        Combine vector and lexical search for best results.
        
        Why hybrid works:
        - Vector search finds conceptually similar documents
        - Lexical search finds exact term matches
        - Together they cover both precision and recall
        """
        
        # Execute both searches in parallel
        vector_task = asyncio.create_task(
            self._vector_retrieval(query, top_k, filters)
        )
        lexical_task = asyncio.create_task(
            self._lexical_retrieval(query, top_k, filters)
        )
        
        vector_results, lexical_results = await asyncio.gather(
            vector_task, lexical_task
        )
        
        # Combine using RRF
        combined_results = self._reciprocal_rank_fusion(
            [vector_results, lexical_results],
            k=60
        )
        
        return combined_results[:top_k]
    
    async def _rerank_results(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Re-rank documents using a cross-encoder model.
        
        Cross-encoder benefits:
        - More accurate than bi-encoder (embedding similarity)
        - Considers query-document interaction
        - Better at nuanced relevance
        
        Trade-offs:
        - Slower than vector similarity
        - Can't pre-compute
        """
        
        if not documents:
            return documents
        
        # Prepare pairs for ranking
        pairs = [(query, doc.content) for doc in documents]
        
        # Get scores from ranker
        scores = await self.ranker.score_pairs(pairs)
        
        # Combine scores with documents
        for doc, score in zip(documents, scores):
            # Blend original and rerank scores
            doc.rerank_score = score
            doc.final_score = 0.7 * score + 0.3 * doc.score
        
        # Sort by final score
        documents.sort(key=lambda d: d.final_score, reverse=True)
        
        return documents
```

### **5.3 Billing and Quota Service**

The billing service manages subscriptions and enforces usage limits:

```python
class BillingService:
    """
    Manages subscriptions, usage tracking, and quota enforcement.
    This is the financial controller of our system.
    """
    
    def __init__(self, db: Database, cache: Cache):
        self.db = db
        self.cache = cache
        
        # Plan definitions
        self.plans = {
            'free': {
                'name': 'Free Plan',
                'limits': {
                    'messages_per_month': 100,
                    'tokens_per_message': 1000,
                    'api_calls_per_day': 100,
                    'storage_mb': 100
                },
                'price_cents': 0
            },
            'pro': {
                'name': 'Pro Plan',
                'limits': {
                    'messages_per_month': 1000,
                    'tokens_per_message': 2000,
                    'api_calls_per_day': 1000,
                    'storage_mb': 1000
                },
                'price_cents': 2900  # $29.00
            },
            'enterprise': {
                'name': 'Enterprise Plan',
                'limits': {
                    'messages_per_month': -1,  # Unlimited
                    'tokens_per_message': 4000,
                    'api_calls_per_day': -1,  # Unlimited
                    'storage_mb': 10000
                },
                'price_cents': 9900  # $99.00
            }
        }
    
    async def check_quota(
        self,
        user_id: str,
        resource: str,
        quantity: int = 1
    ) -> QuotaCheck:
        """
        Check if user has sufficient quota for a resource.
        
        The check process:
        1. Get user's subscription plan
        2. Check current usage for the period
        3. Compare against limits
        4. Return availability and remaining quota
        """
        
        # Try cache first for performance
        cache_key = f"quota:{user_id}:{resource}"
        cached = await self.cache.get(cache_key)
        
        if cached and cached['timestamp'] > time.time() - 60:
            # Use cached result if fresh (< 1 minute old)
            return QuotaCheck(**cached['data'])
        
        # Get user's subscription
        subscription = await self.db.get_user_subscription(user_id)
        if not subscription:
            # Default to free plan
            subscription = {'plan': 'free'}
        
        # Get plan limits
        plan = self.plans[subscription['plan']]
        limit = plan['limits'].get(resource, 0)
        
        # Unlimited check
        if limit == -1:
            result = QuotaCheck(
                allowed=True,
                limit=None,
                used=0,
                remaining=None,
                plan=subscription['plan']
            )
            await self._cache_quota_check(cache_key, result)
            return result
        
        # Get current usage
        period_start = self._get_period_start(resource)
        usage = await self.db.get_usage(
            user_id=user_id,
            resource=resource,
            since=period_start
        )
        
        # Check if within limits
        remaining = limit - usage
        allowed = remaining >= quantity
        
        result = QuotaCheck(
            allowed=allowed,
            limit=limit,
            used=usage,
            remaining=remaining,
            plan=subscription['plan'],
            period_reset=self._get_period_end(resource)
        )
        
        # Cache result
        await self._cache_quota_check(cache_key, result)
        
        return result
    
    async def record_usage(
        self,
        user_id: str,
        resource: str,
        quantity: int,
        metadata: dict = None
    ):
        """
        Record resource usage for billing.
        
        Important: This should be called AFTER the resource is consumed
        to ensure accurate billing.
        """
        
        # Create usage record
        usage_record = {
            'user_id': user_id,
            'resource': resource,
            'quantity': quantity,
            'timestamp': datetime.utcnow(),
            'metadata': metadata or {}
        }
        
        # Store in database (async, fire-and-forget)
        asyncio.create_task(
            self.db.insert_usage_record(usage_record)
        )
        
        # Invalidate quota cache
        cache_key = f"quota:{user_id}:{resource}"
        await self.cache.delete(cache_key)
        
        # Check for quota warnings
        quota = await self.check_quota(user_id, resource, 0)
        
        if quota.remaining is not None and quota.remaining < quota.limit * 0.1:
            # Less than 10% remaining, send warning
            await self._send_quota_warning(user_id, resource, quota)
    
    def _get_period_start(self, resource: str) -> datetime:
        """
        Get the start of the current billing period for a resource.
        
        Period definitions:
        - messages_per_month: Start of current month
        - api_calls_per_day: Start of current day
        - tokens_per_message: Per-request (no period)
        """
        
        now = datetime.utcnow()
        
        if 'month' in resource:
            # Monthly period
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif 'day' in resource:
            # Daily period
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # No period (per-request)
            return now
```

---

## **6. Data Design: The Foundation**

Understanding our data design is crucial for system performance and reliability. Let's explore each database and its role.

### **6.1 Database Selection Strategy**

We use different databases for different purposes, like using the right tool for each job:

```mermaid
graph LR
    subgraph "Decision Tree"
        Q{Data Type?}
        Q -->|Transactional| PG[PostgreSQL]
        Q -->|Documents + Vectors| MONGO[MongoDB]
        Q -->|Time Series| SCYLLA[ScyllaDB]
        Q -->|Cache/Session| REDIS[Redis]
        Q -->|Large Files| S3[Object Storage]
        
        PG --> ACID[ACID Compliance<br/>Relationships<br/>Strong Consistency]
        MONGO --> FLEX[Flexible Schema<br/>Vector Search<br/>Rich Queries]
        SCYLLA --> SCALE[High Throughput<br/>Time Series<br/>Wide Columns]
        REDIS --> FAST[Sub-ms Latency<br/>Pub/Sub<br/>Expiration]
        S3 --> CHEAP[Cost Effective<br/>Unlimited Scale<br/>CDN Integration]
    end
```

### **6.2 PostgreSQL - The System of Record**

PostgreSQL stores all critical business data requiring ACID guarantees:

```sql
-- Core user and organization structure
CREATE SCHEMA IF NOT EXISTS core;

-- Organizations table
CREATE TABLE core.organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255) UNIQUE,
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Users table with comprehensive fields
CREATE TABLE core.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    
    -- Status flags
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    is_superuser BOOLEAN DEFAULT false,  -- Admin privileges
    
    -- Relationships
    organization_id UUID REFERENCES core.organizations(id),
    
    -- Subscription and settings
    subscription_plan VARCHAR(50) DEFAULT 'free',
    preferences JSONB DEFAULT '{}',
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
);

-- Subscriptions with billing information
CREATE TABLE core.subscriptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES core.users(id) ON DELETE CASCADE,
    
    -- Plan details
    plan_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    billing_cycle VARCHAR(20),  -- 'monthly', 'yearly'
    
    -- Dates
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ends_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ,
    
    -- Billing
    amount_cents INTEGER NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Limits and features
    limits JSONB NOT NULL DEFAULT '{}',
    features JSONB DEFAULT '[]',
    
    -- Settings
    auto_renew BOOLEAN DEFAULT true,
    
    -- Metadata
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure only one active subscription per user
    CONSTRAINT one_active_subscription UNIQUE (user_id, status) 
        WHERE status IN ('active', 'trialing')
);

-- Usage tracking for billing
CREATE TABLE core.usage_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES core.users(id) ON DELETE CASCADE,
    
    -- Resource identification
    resource_type VARCHAR(50) NOT NULL,  -- 'messages', 'tokens', 'api_calls'
    quantity DECIMAL(10, 2) NOT NULL,
    
    -- Billing period
    billing_period_start TIMESTAMPTZ NOT NULL,
    billing_period_end TIMESTAMPTZ NOT NULL,
    
    -- Additional data
    metadata JSONB DEFAULT '{}',
    
    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for efficient queries
    INDEX idx_usage_user_period (user_id, billing_period_start, billing_period_end),
    INDEX idx_usage_resource (resource_type, created_at DESC)
);

-- Audit log for compliance
CREATE TABLE core.audit_logs (
    id BIGSERIAL PRIMARY KEY,
    
    -- Actor information
    user_id UUID REFERENCES core.users(id),
    ip_address INET,
    user_agent TEXT,
    
    -- Action details
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    
    -- Request and response
    request_method VARCHAR(10),
    request_path TEXT,
    request_body JSONB,
    response_status INTEGER,
    response_time_ms INTEGER,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for efficient querying
    INDEX idx_audit_user (user_id, created_at DESC),
    INDEX idx_audit_action (action, created_at DESC),
    INDEX idx_audit_resource (resource_type, resource_id)
) PARTITION BY RANGE (created_at);

-- Create monthly partitions for audit logs
CREATE TABLE core.audit_logs_2025_01 PARTITION OF core.audit_logs
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Add update triggers
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON core.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON core.organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
```

### **6.3 MongoDB - Documents and Vectors**

MongoDB stores our knowledge base and vector embeddings:

```javascript
// Database design for MongoDB
use rag_chatbot;

// Create collections with schemas
db.createCollection("documents", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["title", "content", "source", "created_at"],
      properties: {
        _id: { bsonType: "objectId" },
        title: { bsonType: "string" },
        content: { bsonType: "string" },
        source: { 
          bsonType: "string",
          enum: ["upload", "web", "api", "manual"]
        },
        
        // Metadata
        organization_id: { bsonType: "string" },
        category: { bsonType: "string" },
        tags: { bsonType: "array", items: { bsonType: "string" } },
        
        // Processing information
        processing_status: {
          bsonType: "string",
          enum: ["pending", "processing", "completed", "failed"]
        },
        chunk_count: { bsonType: "int" },
        
        // Timestamps
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" },
        last_accessed_at: { bsonType: "date" },
        
        // Usage statistics
        access_count: { bsonType: "int" },
        relevance_scores: { bsonType: "array", items: { bsonType: "double" } }
      }
    }
  }
});

// Embeddings collection with vector index
db.createCollection("embeddings", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["document_id", "chunk_index", "content", "embedding"],
      properties: {
        _id: { bsonType: "objectId" },
        
        // References
        document_id: { bsonType: "objectId" },
        chunk_index: { bsonType: "int" },
        
        // Content
        content: { bsonType: "string" },
        
        // Vector embedding (768 dimensions)
        embedding: {
          bsonType: "array",
          items: { bsonType: "double" },
          minItems: 768,
          maxItems: 768
        },
        
        // Metadata for filtering
        organization_id: { bsonType: "string" },
        category: { bsonType: "string" },
        source: { bsonType: "string" },
        tags: { bsonType: "array", items: { bsonType: "string" } },
        
        // Chunk metadata
        chunk_metadata: {
          bsonType: "object",
          properties: {
            start_char: { bsonType: "int" },
            end_char: { bsonType: "int" },
            total_chunks: { bsonType: "int" },
            overlap_chars: { bsonType: "int" }
          }
        },
        
        // Timestamps
        created_at: { bsonType: "date" },
        last_accessed_at: { bsonType: "date" }
      }
    }
  }
});

// Create vector search index
db.embeddings.createSearchIndex({
  name: "vector_search_index",
  type: "vectorSearch",
  definition: {
    fields: [{
      type: "vector",
      path: "embedding",
      numDimensions: 768,
      similarity: "cosine"
    }]
  }
});

// Create compound indexes for efficient filtering
db.embeddings.createIndex({ 
  "organization_id": 1, 
  "category": 1,
  "created_at": -1 
});

db.embeddings.createIndex({ 
  "document_id": 1, 
  "chunk_index": 1 
}, { unique: true });

// Text search index on content
db.embeddings.createIndex({ 
  "content": "text" 
});

// Aggregation pipeline for hybrid search
function hybridSearch(query, vector, organizationId, limit = 10) {
  return db.embeddings.aggregate([
    // Stage 1: Vector search
    {
      $vectorSearch: {
        index: "vector_search_index",
        path: "embedding",
        queryVector: vector,
        numCandidates: limit * 10,
        limit: limit * 2
      }
    },
    
    // Stage 2: Add metadata filters
    {
      $match: {
        organization_id: organizationId
      }
    },
    
    // Stage 3: Add text search score
    {
      $addFields: {
        textScore: { $meta: "textScore" },
        vectorScore: { $meta: "vectorSearchScore" }
      }
    },
    
    // Stage 4: Combine scores
    {
      $addFields: {
        combinedScore: {
          $add: [
            { $multiply: ["$vectorScore", 0.7] },
            { $multiply: ["$textScore", 0.3] }
          ]
        }
      }
    },
    
    // Stage 5: Sort by combined score
    {
      $sort: { combinedScore: -1 }
    },
    
    // Stage 6: Limit results
    {
      $limit: limit
    },
    
    // Stage 7: Join with documents
    {
      $lookup: {
        from: "documents",
        localField: "document_id",
        foreignField: "_id",
        as: "document"
      }
    }
  ]);
}
```

### **6.4 ScyllaDB - Conversation History**

ScyllaDB handles high-throughput time-series data:

```sql
-- Create keyspace with replication
CREATE KEYSPACE IF NOT EXISTS chatbot_ks
WITH replication = {
  'class': 'NetworkTopologyStrategy',
  'datacenter1': 3  -- 3 replicas in primary datacenter
}
AND durable_writes = true;

USE chatbot_ks;

-- Conversation history table
CREATE TABLE IF NOT EXISTS conversation_history (
    -- Partition key
    session_id UUID,
    
    -- Clustering keys
    timestamp TIMESTAMP,
    message_id UUID,
    
    -- Data columns
    user_id UUID,
    actor TEXT,  -- 'user' or 'assistant'
    message TEXT,
    
    -- Response metadata
    embedding_used BOOLEAN,
    tokens_used INT,
    response_time_ms INT,
    model_used TEXT,
    confidence DOUBLE,
    
    -- Context information
    sources_used LIST<TEXT>,
    context_chunks LIST<FROZEN<MAP<TEXT, TEXT>>>,
    
    -- User feedback
    feedback_score INT,  -- 1-5 rating
    feedback_text TEXT,
    
    -- Additional metadata
    metadata MAP<TEXT, TEXT>,
    
    PRIMARY KEY (session_id, timestamp, message_id)
) WITH CLUSTERING ORDER BY (timestamp DESC, message_id ASC)
  AND compression = {'class': 'LZ4Compressor'}
  AND compaction = {
    'class': 'TimeWindowCompactionStrategy',
    'compaction_window_unit': 'HOURS',
    'compaction_window_size': 24
  }
  AND default_time_to_live = 15552000  -- 180 days
  AND gc_grace_seconds = 86400;

-- Index for user queries
CREATE INDEX IF NOT EXISTS idx_user_conversations 
ON conversation_history (user_id);

-- Materialized view for user history
CREATE MATERIALIZED VIEW IF NOT EXISTS user_conversation_history AS
    SELECT * FROM conversation_history
    WHERE user_id IS NOT NULL 
      AND session_id IS NOT NULL
      AND timestamp IS NOT NULL
      AND message_id IS NOT NULL
    PRIMARY KEY (user_id, timestamp, session_id, message_id)
    WITH CLUSTERING ORDER BY (timestamp DESC);

-- Analytics table for aggregated metrics
CREATE TABLE IF NOT EXISTS usage_analytics (
    -- Partition by user and date
    user_id UUID,
    date DATE,
    
    -- Clustering by hour and metric
    hour INT,
    metric_name TEXT,
    
    -- Counter column
    value COUNTER,
    
    PRIMARY KEY ((user_id, date), hour, metric_name)
) WITH CLUSTERING ORDER BY (hour ASC, metric_name ASC)
  AND compression = {'class': 'LZ4Compressor'};

-- Prepared statements for common queries
-- Get recent conversations for a user
-- SELECT * FROM user_conversation_history 
-- WHERE user_id = ? 
-- LIMIT ?;

-- Get session history
-- SELECT * FROM conversation_history 
-- WHERE session_id = ? 
-- LIMIT ?;

-- Update usage counter
-- UPDATE usage_analytics 
-- SET value = value + ? 
-- WHERE user_id = ? 
--   AND date = ? 
--   AND hour = ? 
--   AND metric_name = ?;
```

### **6.5 Redis - Caching and Sessions**

Redis provides our high-performance caching layer:

```python
# Redis data structures and patterns

class RedisSchema:
    """
    Defines Redis key patterns and data structures.
    Consistent naming prevents key collisions.
    """
    
    # Key patterns
    PATTERNS = {
        # User sessions
        'session': 'session:{user_id}',
        'session_data': 'session:data:{session_id}',
        
        # Rate limiting
        'rate_limit': 'rate:{user_id}:{resource}',
        'rate_window': 'rate:window:{user_id}:{resource}:{window}',
        
        # Quota caching
        'quota': 'quota:{user_id}:{resource}',
        'usage': 'usage:{user_id}:{period}',
        
        # Response caching
        'response': 'response:{query_hash}',
        'embedding': 'embedding:{text_hash}',
        
        # Search result caching
        'search': 'search:{query_hash}:{filters_hash}',
        
        # Feature flags
        'feature': 'feature:{flag_name}',
        'user_feature': 'feature:{flag_name}:{user_id}',
        
        # Distributed locks
        'lock': 'lock:{resource}:{id}',
        
        # Task queues
        'task_queue': 'queue:{priority}:{task_type}',
        'task_result': 'result:{task_id}'
    }
    
    # TTL configurations (in seconds)
    TTL = {
        'session': 3600,          # 1 hour
        'rate_limit': 60,         # 1 minute
        'quota': 300,             # 5 minutes
        'response': 1800,         # 30 minutes
        'embedding': 86400,       # 24 hours
        'search': 1800,           # 30 minutes
        'task_result': 3600,      # 1 hour
        'lock': 30                # 30 seconds
    }
    
    @classmethod
    def get_key(cls, pattern: str, **kwargs) -> str:
        """Generate a Redis key from pattern."""
        return cls.PATTERNS[pattern].format(**kwargs)
    
    @classmethod
    def get_ttl(cls, key_type: str) -> int:
        """Get TTL for a key type."""
        return cls.TTL.get(key_type, 3600)

# Example Redis operations
class RedisCache:
    """
    Redis caching operations with proper patterns.
    """
    
    async def cache_response(self, query: str, response: dict, ttl: int = None):
        """Cache a chat response."""
        key = RedisSchema.get_key('response', query_hash=self._hash(query))
        ttl = ttl or RedisSchema.get_ttl('response')
        
        await self.redis.setex(
            key,
            ttl,
            json.dumps(response)
        )
    
    async def get_cached_response(self, query: str) -> dict:
        """Get cached response if available."""
        key = RedisSchema.get_key('response', query_hash=self._hash(query))
        
        cached = await self.redis.get(key)
        if cached:
            # Extend TTL on cache hit
            await self.redis.expire(key, RedisSchema.get_ttl('response'))
            return json.loads(cached)
        
        return None
    
    async def rate_limit_check(self, user_id: str, resource: str, limit: int, window: int):
        """
        Sliding window rate limiting.
        
        Algorithm:
        1. Use sorted set with timestamp scores
        2. Remove old entries outside window
        3. Count remaining entries
        4. Add new entry if under limit
        """
        
        key = RedisSchema.get_key('rate_window', 
                                 user_id=user_id, 
                                 resource=resource,
                                 window=window)
        
        now = time.time()
        window_start = now - window
        
        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current entries
        current_count = await self.redis.zcard(key)
        
        if current_count >= limit:
            return False  # Rate limit exceeded
        
        # Add new entry
        await self.redis.zadd(key, {str(uuid.uuid4()): now})
        
        # Set expiry
        await self.redis.expire(key, window + 1)
        
        return True  # Within limit
```

---

## **7. API Design: The Interface**

### **7.1 RESTful API Principles**

Our API follows REST principles with some pragmatic additions:

```yaml
# OpenAPI 3.0 Specification (excerpt)
openapi: 3.0.0
info:
  title: RAG Chatbot Platform API
  version: 5.0.0
  description: |
    Enterprise-grade RAG chatbot with comprehensive features.
    
    ## Authentication
    All endpoints except /health require JWT authentication.
    Include token in Authorization header: `Bearer <token>`
    
    ## Rate Limiting
    - Default: 100 requests per minute
    - Chat endpoints: 30 requests per minute
    - Search endpoints: 60 requests per minute
    
    ## Errors
    Standard HTTP status codes with detailed error bodies.

servers:
  - url: https://api.chatbot.com/v1
    description: Production
  - url: https://staging-api.chatbot.com/v1
    description: Staging
  - url: http://localhost:8000
    description: Development

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
  
  schemas:
    Error:
      type: object
      required:
        - code
        - message
      properties:
        code:
          type: string
          description: Error code for programmatic handling
        message:
          type: string
          description: Human-readable error message
        details:
          type: object
          description: Additional error context
        request_id:
          type: string
          description: Request ID for support

paths:
  /chat:
    post:
      summary: Send a chat message
      description: |
        Process a user message through the RAG pipeline.
        
        The system will:
        1. Validate the message
        2. Retrieve relevant context
        3. Generate a response
        4. Return with citations
      
      security:
        - bearerAuth: []
      
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - message
              properties:
                message:
                  type: string
                  description: User's message
                  maxLength: 2000
                session_id:
                  type: string
                  format: uuid
                  description: Session ID for conversation continuity
                enable_rag:
                  type: boolean
                  default: true
                  description: Enable retrieval-augmented generation
                stream:
                  type: boolean
                  default: false
                  description: Stream response tokens
                filters:
                  type: object
                  description: Filters for document retrieval
                  properties:
                    category:
                      type: string
                    date_from:
                      type: string
                      format: date
                    date_to:
                      type: string
                      format: date
                max_tokens:
                  type: integer
                  default: 512
                  minimum: 50
                  maximum: 2000
                temperature:
                  type: number
                  default: 0.7
                  minimum: 0.0
                  maximum: 1.0
      
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                type: object
                properties:
                  session_id:
                    type: string
                    format: uuid
                  message_id:
                    type: string
                    format: uuid
                  answer:
                    type: string
                    description: Generated response
                  sources:
                    type: array
                    items:
                      type: object
                      properties:
                        document_id:
                          type: string
                        title:
                          type: string
                        excerpt:
                          type: string
                        relevance_score:
                          type: number
                  metadata:
                    type: object
                    properties:
                      tokens_used:
                        type: integer
                      response_time_ms:
                        type: number
                      model_used:
                        type: string
                      cached:
                        type: boolean
        
        '429':
          description: Rate limit or quota exceeded
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Error'
              example:
                code: QUOTA_EXCEEDED
                message: Monthly message quota exceeded
                details:
                  current_usage: 100
                  limit: 100
                  reset_at: '2025-02-01T00:00:00Z'
```

### **7.2 API Implementation with FastAPI**

Let's implement the API endpoints with production considerations:

```python
from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import time
from contextlib import asynccontextmanager

# Create application with lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Startup tasks:
    - Initialize database connections
    - Load ML models
    - Warm up caches
    
    Shutdown tasks:
    - Close database connections
    - Save model states
    - Flush caches
    """
    
    # Startup
    logger.info("Starting application...")
    
    # Initialize databases
    await postgres_manager.initialize()
    await mongo_manager.connect()
    await redis_client.initialize()
    await scylla_manager.connect()
    
    # Load models
    await embedding_service.initialize()
    await generation_service.initialize()
    
    # Warm up with common queries
    await warmup_cache()
    
    logger.info("Application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    
    # Close connections gracefully
    await postgres_manager.close()
    await mongo_manager.disconnect()
    await redis_client.close()
    await scylla_manager.disconnect()
    
    logger.info("Application shut down")

# Create FastAPI application
app = FastAPI(
    title="RAG Chatbot Platform",
    version="5.0.0",
    description="Enterprise RAG chatbot with vector search and billing",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware for request tracking
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID for tracing."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    
    # Add to logging context
    with logger.contextualize(request_id=request_id):
        response = await call_next(request)
    
    response.headers["X-Request-ID"] = request_id
    return response

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/health/detailed")
async def detailed_health_check():
    """
    Comprehensive health check of all dependencies.
    
    This endpoint checks:
    - Database connections
    - Model availability
    - Cache connectivity
    - Queue depth
    """
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }
    
    # Check PostgreSQL
    try:
        await postgres_manager.execute("SELECT 1")
        health_status["checks"]["postgres"] = "healthy"
    except Exception as e:
        health_status["checks"]["postgres"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check MongoDB
    try:
        await mongo_manager.ping()
        health_status["checks"]["mongodb"] = "healthy"
    except Exception as e:
        health_status["checks"]["mongodb"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        await redis_client.ping()
        health_status["checks"]["redis"] = "healthy"
    except Exception as e:
        health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
        # Redis is non-critical, don't degrade overall status
    
    # Check ML models
    if embedding_service.is_ready():
        health_status["checks"]["embedding_model"] = "healthy"
    else:
        health_status["checks"]["embedding_model"] = "not ready"
        health_status["status"] = "degraded"
    
    if generation_service.is_ready():
        health_status["checks"]["generation_model"] = "healthy"
    else:
        health_status["checks"]["generation_model"] = "not ready"
        health_status["status"] = "degraded"
    
    # Return appropriate status code
    status_code = 200 if health_status["status"] == "healthy" else 503
    
    return JSONResponse(
        content=health_status,
        status_code=status_code
    )

# Include routers
from app.api.endpoints import auth, chat, search, billing, admin

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(billing.router, prefix="/billing", tags=["Billing"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handle uncaught exceptions gracefully.
    
    Important: Never expose internal errors to clients.
    Log full details internally.
    """
    
    logger.error(
        f"Unhandled exception: {exc}",
        exc_info=True,
        request_id=getattr(request.state, 'request_id', 'unknown')
    )
    
    # Return safe error response
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )
```

This enhanced document now includes comprehensive explanations throughout, building concepts gradually from simple to complex. It maintains a teaching-oriented approach with clear explanations of why each design decision was made, how components work together, and what trade-offs are involved. The document serves as both a technical reference and a learning guide for new team members joining the project.

Would you like me to continue with the remaining sections (8-17) with the same level of detail and teaching-oriented approach?

## **8. Security & Governance: Protection Layers**

Security isn't just a feature; it's the foundation of trust in our system. Let's explore how we protect data, users, and the system itself through multiple layers of defense. Think of security as building a castle - we need walls, gates, guards, and even a moat.

### **8.1 Understanding Security Threats**

Before we build defenses, let's understand what we're defending against. Each threat requires specific countermeasures:

```python
class ThreatModel:
    """
    Understanding threats helps us build appropriate defenses.
    This model categorizes threats we need to protect against.
    """
    
    THREATS = {
        'authentication': {
            'description': 'Unauthorized access attempts',
            'examples': ['Password brute force', 'Token theft', 'Session hijacking'],
            'defenses': ['Strong passwords', 'JWT expiration', 'Secure cookies']
        },
        'data_breach': {
            'description': 'Unauthorized data access',
            'examples': ['SQL injection', 'Direct object references', 'Data leakage'],
            'defenses': ['Parameterized queries', 'Access controls', 'Encryption']
        },
        'denial_of_service': {
            'description': 'Service availability attacks',
            'examples': ['Request flooding', 'Resource exhaustion', 'Amplification'],
            'defenses': ['Rate limiting', 'Resource quotas', 'CDN protection']
        },
        'prompt_injection': {
            'description': 'AI-specific attacks',
            'examples': ['Jailbreaking', 'Context manipulation', 'Data extraction'],
            'defenses': ['Input sanitization', 'Output filtering', 'Context isolation']
        },
        'insider_threats': {
            'description': 'Malicious or negligent insiders',
            'examples': ['Data theft', 'Privilege abuse', 'Accidental exposure'],
            'defenses': ['Audit logging', 'Least privilege', 'Data classification']
        }
    }
```

### **8.2 Authentication Layer - Who Are You?**

Authentication is about verifying identity. We use multiple methods to ensure users are who they claim to be:

```python
class AuthenticationService:
    """
    Comprehensive authentication service with multiple verification methods.
    Think of this as the castle's gate keeper - checking everyone's identity.
    """
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.pwd_context = CryptContext(
            schemes=["argon2", "bcrypt"],  # Argon2 is more secure but slower
            default="argon2",
            argon2__rounds=3,
            argon2__memory_cost=65536,
            argon2__parallelism=4
        )
        self.token_manager = TokenManager(config)
        self.mfa_manager = MFAManager()
        
    async def authenticate_user(
        self,
        email: str,
        password: str,
        mfa_code: Optional[str] = None
    ) -> AuthResult:
        """
        Multi-step authentication process.
        
        The authentication journey:
        1. Validate input format
        2. Check user exists and is active
        3. Verify password
        4. Check for MFA requirement
        5. Validate MFA if required
        6. Check for suspicious activity
        7. Create session
        """
        
        # Step 1: Input validation
        if not self._validate_email_format(email):
            # Log potential attack (malformed input might indicate scanning)
            await self._log_auth_attempt(
                email=email,
                success=False,
                reason="invalid_email_format"
            )
            raise AuthenticationError("Invalid credentials")
        
        # Step 2: Get user (timing-safe to prevent user enumeration)
        user = await self._get_user_timing_safe(email)
        if not user:
            # Use same error as wrong password to prevent user enumeration
            raise AuthenticationError("Invalid credentials")
        
        # Step 3: Check account status
        if not user.is_active:
            await self._log_auth_attempt(
                email=email,
                success=False,
                reason="account_inactive"
            )
            raise AuthenticationError("Account inactive")
        
        if user.is_locked:
            # Check if lock period has expired
            if not await self._check_unlock_time(user):
                raise AuthenticationError(
                    f"Account locked. Try again at {user.unlock_time}"
                )
        
        # Step 4: Verify password
        if not await self._verify_password(password, user.hashed_password):
            # Record failed attempt
            await self._record_failed_attempt(user)
            
            # Check if we should lock the account
            if await self._should_lock_account(user):
                await self._lock_account(user)
                raise AuthenticationError("Too many failed attempts. Account locked.")
            
            raise AuthenticationError("Invalid credentials")
        
        # Step 5: Check MFA requirement
        if user.mfa_enabled:
            if not mfa_code:
                # Return partial success, requesting MFA
                return AuthResult(
                    success=False,
                    requires_mfa=True,
                    mfa_session_token=await self._create_mfa_session(user)
                )
            
            # Validate MFA code
            if not await self.mfa_manager.verify_code(user, mfa_code):
                await self._record_failed_attempt(user, mfa_failed=True)
                raise AuthenticationError("Invalid MFA code")
        
        # Step 6: Check for suspicious activity
        risk_score = await self._calculate_risk_score(user, request_context)
        if risk_score > self.config.risk_threshold:
            # Require additional verification for suspicious login
            await self._trigger_additional_verification(user)
            return AuthResult(
                success=False,
                requires_verification=True,
                verification_method="email"
            )
        
        # Step 7: Create session
        tokens = await self._create_session(user)
        
        # Update user's last login
        await self._update_last_login(user)
        
        # Log successful authentication
        await self._log_auth_attempt(
            email=email,
            success=True,
            user_id=user.id
        )
        
        return AuthResult(
            success=True,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            user=user
        )
    
    async def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password with timing attack protection.
        
        Important: We use verify() which is designed to take constant time
        regardless of how many characters match. This prevents timing attacks
        where an attacker could determine password length or partial matches
        based on response time.
        """
        
        # Add small random delay to further obscure timing
        await asyncio.sleep(random.uniform(0.01, 0.03))
        
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    async def _calculate_risk_score(self, user: User, context: RequestContext) -> float:
        """
        Calculate login risk score based on various factors.
        
        Risk factors considered:
        - New device/browser
        - Unusual location
        - Unusual time of day
        - Recent password change
        - Recent failed attempts
        - IP reputation
        """
        
        risk_score = 0.0
        
        # Check if this is a new device
        device_fingerprint = context.get_device_fingerprint()
        if not await self._is_known_device(user, device_fingerprint):
            risk_score += 0.3
        
        # Check location
        current_location = await self._get_location_from_ip(context.ip_address)
        if await self._is_unusual_location(user, current_location):
            risk_score += 0.4
        
        # Check time pattern
        current_hour = datetime.utcnow().hour
        if not await self._is_usual_login_time(user, current_hour):
            risk_score += 0.2
        
        # Check recent security events
        if await self._has_recent_password_change(user):
            risk_score += 0.3
        
        if await self._has_recent_failed_attempts(user):
            risk_score += 0.2
        
        # Check IP reputation
        ip_reputation = await self._check_ip_reputation(context.ip_address)
        if ip_reputation == 'suspicious':
            risk_score += 0.5
        elif ip_reputation == 'malicious':
            risk_score += 1.0
        
        return min(risk_score, 1.0)  # Cap at 1.0
```

### **8.3 Authorization Layer - What Can You Do?**

While authentication verifies identity, authorization determines what actions a user can perform. We implement Role-Based Access Control (RBAC) with fine-grained permissions:

```python
class AuthorizationService:
    """
    Fine-grained authorization system.
    Think of this as the castle's internal security - determining where
    each person can go and what they can do once inside.
    """
    
    def __init__(self):
        # Define roles and their permissions
        self.roles = {
            'guest': {
                'permissions': [
                    'chat:read_public',
                    'search:execute_basic'
                ],
                'description': 'Unauthenticated user with minimal access'
            },
            'user': {
                'inherits': ['guest'],
                'permissions': [
                    'chat:create',
                    'chat:read_own',
                    'search:execute_advanced',
                    'profile:read_own',
                    'profile:update_own'
                ],
                'description': 'Regular authenticated user'
            },
            'pro_user': {
                'inherits': ['user'],
                'permissions': [
                    'chat:bulk_export',
                    'search:use_premium_models',
                    'api:access'
                ],
                'description': 'Paid user with enhanced features'
            },
            'organization_admin': {
                'inherits': ['pro_user'],
                'permissions': [
                    'organization:read',
                    'organization:update',
                    'organization:manage_users',
                    'billing:read',
                    'billing:update'
                ],
                'description': 'Organization administrator'
            },
            'system_admin': {
                'inherits': ['organization_admin'],
                'permissions': [
                    'admin:access_dashboard',
                    'admin:view_all_users',
                    'admin:manage_features',
                    'admin:view_system_metrics',
                    'admin:execute_migrations'
                ],
                'description': 'System administrator with full access'
            }
        }
        
        # Build permission inheritance tree
        self.permission_tree = self._build_permission_tree()
        
    async def check_permission(
        self,
        user: User,
        resource: str,
        action: str,
        context: Optional[Dict] = None
    ) -> bool:
        """
        Check if a user has permission to perform an action on a resource.
        
        Permission check flow:
        1. Build the required permission string
        2. Get user's effective permissions (including inherited)
        3. Check for exact match or wildcard
        4. Apply contextual rules if needed
        5. Log the authorization decision
        """
        
        # Build permission string
        required_permission = f"{resource}:{action}"
        
        # Get user's effective permissions
        user_permissions = await self._get_effective_permissions(user)
        
        # Check for exact match
        if required_permission in user_permissions:
            await self._log_authorization(user, required_permission, True, "exact_match")
            return True
        
        # Check for wildcard permissions
        if self._check_wildcard_permission(required_permission, user_permissions):
            await self._log_authorization(user, required_permission, True, "wildcard_match")
            return True
        
        # Check contextual permissions
        if context and await self._check_contextual_permission(
            user, required_permission, context
        ):
            await self._log_authorization(user, required_permission, True, "contextual")
            return True
        
        # Permission denied
        await self._log_authorization(user, required_permission, False, "no_match")
        return False
    
    async def _check_contextual_permission(
        self,
        user: User,
        permission: str,
        context: Dict
    ) -> bool:
        """
        Check permissions that depend on context.
        
        Examples of contextual permissions:
        - Can only edit own resources
        - Can only access resources in same organization
        - Time-based permissions (business hours only)
        - Location-based permissions
        """
        
        # Check ownership-based permissions
        if permission.endswith(':read_own') or permission.endswith(':update_own'):
            resource_owner = context.get('owner_id')
            if resource_owner and resource_owner == user.id:
                return True
        
        # Check organization-based permissions
        if user.organization_id:
            resource_org = context.get('organization_id')
            if resource_org and resource_org == user.organization_id:
                # User can access resources in their organization
                if permission.startswith('organization:'):
                    return True
        
        # Check time-based permissions
        if context.get('require_business_hours'):
            if not self._is_business_hours():
                return False
        
        # Check rate-limited permissions
        if context.get('rate_limited'):
            if not await self._check_rate_limit(user, permission):
                return False
        
        return False
    
    def require_permission(self, resource: str, action: str):
        """
        Decorator for protecting endpoints with permission checks.
        
        Usage:
        @require_permission('admin', 'access_dashboard')
        async def admin_dashboard(user: User = Depends(get_current_user)):
            return {"message": "Welcome to admin dashboard"}
        """
        
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, user: User = None, **kwargs):
                if not user:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required"
                    )
                
                # Check permission
                has_permission = await self.check_permission(
                    user, resource, action
                )
                
                if not has_permission:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Permission denied: {resource}:{action}"
                    )
                
                return await func(*args, user=user, **kwargs)
            
            return wrapper
        return decorator
```

### **8.4 Data Protection - Encryption and Privacy**

Protecting data requires multiple layers of encryption and careful handling of sensitive information:

```python
class DataProtectionService:
    """
    Comprehensive data protection implementation.
    This is like having locked safes within our castle - even if someone
    gets in, they can't access the valuable contents.
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        
        # Initialize encryption keys from secure storage
        self.master_key = self._load_master_key()
        self.field_encryption_key = self._derive_key(self.master_key, "field_encryption")
        self.file_encryption_key = self._derive_key(self.master_key, "file_encryption")
        
        # Initialize cipher suites
        self.fernet = Fernet(self.field_encryption_key)
        
    async def encrypt_sensitive_field(self, value: str, field_type: str) -> str:
        """
        Encrypt sensitive fields before storage.
        
        Field types and their handling:
        - PII: Personal identifiable information (SSN, passport)
        - PHI: Protected health information
        - FINANCIAL: Credit cards, bank accounts
        - CREDENTIALS: API keys, passwords
        """
        
        if not value:
            return value
        
        # Add metadata for key rotation
        metadata = {
            'version': self.config.encryption_version,
            'type': field_type,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Combine value with metadata
        payload = {
            'value': value,
            'metadata': metadata
        }
        
        # Encrypt
        encrypted = self.fernet.encrypt(
            json.dumps(payload).encode()
        )
        
        # Return base64 encoded for storage
        return base64.b64encode(encrypted).decode()
    
    async def decrypt_sensitive_field(self, encrypted_value: str) -> str:
        """
        Decrypt sensitive fields after retrieval.
        
        Handles key rotation by checking version metadata.
        """
        
        if not encrypted_value:
            return encrypted_value
        
        try:
            # Decode from base64
            encrypted = base64.b64decode(encrypted_value.encode())
            
            # Decrypt
            decrypted = self.fernet.decrypt(encrypted)
            
            # Parse payload
            payload = json.loads(decrypted.decode())
            
            # Check if re-encryption needed (key rotation)
            if payload['metadata']['version'] < self.config.encryption_version:
                # Re-encrypt with new key
                await self._rotate_encryption(encrypted_value, payload)
            
            return payload['value']
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise DataProtectionError("Unable to decrypt field")
    
    async def anonymize_data(self, data: Dict, rules: AnonymizationRules) -> Dict:
        """
        Anonymize data according to privacy rules.
        
        Anonymization techniques:
        - Redaction: Complete removal
        - Masking: Partial hiding (e.g., ***-**-1234)
        - Generalization: Less specific (age 28 -> age 25-30)
        - Pseudonymization: Replace with fake but consistent data
        """
        
        anonymized = {}
        
        for field, value in data.items():
            if field in rules.redact_fields:
                # Complete redaction
                anonymized[field] = "[REDACTED]"
                
            elif field in rules.mask_fields:
                # Partial masking
                anonymized[field] = self._mask_value(value, rules.mask_patterns[field])
                
            elif field in rules.generalize_fields:
                # Generalization
                anonymized[field] = self._generalize_value(value, field)
                
            elif field in rules.pseudonymize_fields:
                # Pseudonymization (consistent fake data)
                anonymized[field] = await self._pseudonymize_value(value, field)
                
            else:
                # No anonymization needed
                anonymized[field] = value
        
        return anonymized
    
    def _mask_value(self, value: str, pattern: str) -> str:
        """
        Apply masking pattern to sensitive value.
        
        Examples:
        - SSN: 123-45-6789 -> ***-**-6789
        - Credit Card: 1234-5678-9012-3456 -> ****-****-****-3456
        - Email: user@example.com -> u***@example.com
        """
        
        if pattern == 'ssn':
            if len(value) >= 4:
                return f"***-**-{value[-4:]}"
        elif pattern == 'credit_card':
            if len(value) >= 4:
                return f"****-****-****-{value[-4:]}"
        elif pattern == 'email':
            parts = value.split('@')
            if len(parts) == 2:
                username = parts[0]
                if len(username) > 1:
                    return f"{username[0]}***@{parts[1]}"
        
        return "****"
    
    async def enforce_data_retention(self, user_id: str):
        """
        Enforce data retention policies.
        
        Retention rules:
        - Conversation history: 180 days
        - Usage logs: 90 days
        - Audit logs: 7 years (regulatory requirement)
        - Deleted user data: 30 days (soft delete period)
        """
        
        retention_policies = {
            'conversations': timedelta(days=180),
            'usage_logs': timedelta(days=90),
            'audit_logs': timedelta(days=2555),  # 7 years
            'deleted_users': timedelta(days=30)
        }
        
        now = datetime.utcnow()
        
        for data_type, retention_period in retention_policies.items():
            cutoff_date = now - retention_period
            
            if data_type == 'conversations':
                # Delete old conversations
                await self._delete_old_conversations(user_id, cutoff_date)
                
            elif data_type == 'usage_logs':
                # Archive and delete old usage logs
                await self._archive_old_usage_logs(user_id, cutoff_date)
                
            elif data_type == 'deleted_users':
                # Permanently delete soft-deleted users
                await self._purge_deleted_users(cutoff_date)
```

### **8.5 Security Monitoring and Incident Response**

Security isn't just about prevention; it's also about detection and response. We need to know when something suspicious is happening:

```python
class SecurityMonitoringService:
    """
    Real-time security monitoring and incident response.
    This is like having security cameras and guards watching for suspicious activity.
    """
    
    def __init__(self):
        self.alert_manager = AlertManager()
        self.incident_recorder = IncidentRecorder()
        
        # Define security events to monitor
        self.security_events = {
            'multiple_failed_logins': {
                'threshold': 5,
                'window': 300,  # 5 minutes
                'severity': 'medium',
                'action': 'lock_account'
            },
            'privilege_escalation_attempt': {
                'threshold': 1,
                'window': 0,
                'severity': 'high',
                'action': 'alert_and_block'
            },
            'data_exfiltration_attempt': {
                'threshold': 100,  # requests
                'window': 60,  # 1 minute
                'severity': 'critical',
                'action': 'immediate_block'
            },
            'suspicious_api_pattern': {
                'threshold': 50,
                'window': 60,
                'severity': 'medium',
                'action': 'rate_limit'
            }
        }
    
    async def monitor_security_event(
        self,
        event_type: str,
        user_id: str,
        metadata: Dict
    ):
        """
        Monitor and respond to security events in real-time.
        
        The monitoring process:
        1. Record the event
        2. Check if threshold is exceeded
        3. Calculate severity
        4. Take appropriate action
        5. Alert security team if needed
        """
        
        # Record event
        await self._record_event(event_type, user_id, metadata)
        
        # Get event configuration
        event_config = self.security_events.get(event_type)
        if not event_config:
            return
        
        # Check threshold
        event_count = await self._count_recent_events(
            event_type,
            user_id,
            event_config['window']
        )
        
        if event_count >= event_config['threshold']:
            # Threshold exceeded - create incident
            incident = await self._create_incident(
                event_type=event_type,
                user_id=user_id,
                severity=event_config['severity'],
                event_count=event_count,
                metadata=metadata
            )
            
            # Take action based on severity and type
            await self._respond_to_incident(incident, event_config['action'])
            
            # Alert security team
            if event_config['severity'] in ['high', 'critical']:
                await self._alert_security_team(incident)
    
    async def _respond_to_incident(self, incident: Incident, action: str):
        """
        Automated incident response based on action type.
        
        Response actions:
        - lock_account: Temporarily lock user account
        - alert_and_block: Alert team and block user
        - immediate_block: Block immediately without delay
        - rate_limit: Apply stricter rate limits
        - investigate: Flag for manual investigation
        """
        
        if action == 'lock_account':
            await self._lock_user_account(
                incident.user_id,
                duration=timedelta(minutes=30),
                reason=f"Security incident: {incident.type}"
            )
            
        elif action == 'alert_and_block':
            # Alert first
            await self.alert_manager.send_alert(
                level='HIGH',
                title=f"Security Incident: {incident.type}",
                details=incident.to_dict()
            )
            
            # Then block
            await self._block_user(incident.user_id)
            
        elif action == 'immediate_block':
            # Block immediately
            await self._block_user(incident.user_id, immediate=True)
            
            # Alert after blocking
            await self.alert_manager.send_critical_alert(
                title=f"CRITICAL: User blocked for {incident.type}",
                details=incident.to_dict()
            )
            
        elif action == 'rate_limit':
            # Apply stricter rate limits
            await self._apply_strict_rate_limit(
                incident.user_id,
                limit=10,  # 10 requests per minute
                duration=timedelta(hours=1)
            )
            
        elif action == 'investigate':
            # Flag for manual investigation
            await self._flag_for_investigation(incident)
    
    async def detect_anomalies(self, user_id: str, activity: UserActivity):
        """
        Detect anomalous behavior using pattern analysis.
        
        Anomaly detection considers:
        - Unusual access patterns
        - Abnormal data volumes
        - Suspicious query patterns
        - Time-based anomalies
        - Geographic anomalies
        """
        
        # Get user's normal behavior profile
        profile = await self._get_user_behavior_profile(user_id)
        
        # Calculate anomaly scores for different dimensions
        anomaly_scores = {
            'access_pattern': self._calculate_access_anomaly(activity, profile),
            'data_volume': self._calculate_volume_anomaly(activity, profile),
            'query_pattern': self._calculate_query_anomaly(activity, profile),
            'temporal': self._calculate_temporal_anomaly(activity, profile),
            'geographic': self._calculate_geographic_anomaly(activity, profile)
        }
        
        # Calculate overall anomaly score
        overall_score = sum(anomaly_scores.values()) / len(anomaly_scores)
        
        # Take action if anomaly detected
        if overall_score > 0.7:
            await self.monitor_security_event(
                'anomalous_behavior',
                user_id,
                {
                    'anomaly_scores': anomaly_scores,
                    'overall_score': overall_score,
                    'activity': activity.to_dict()
                }
            )
        
        return anomaly_scores
```

---

## **9. Getting Started: Setup Guide**

Now let's walk through setting up the entire system from scratch. This section is designed to get new team members productive quickly while understanding what each step accomplishes.

### **9.1 Prerequisites and System Requirements**

Before we begin, let's ensure your development environment is properly configured. Think of this as gathering all your tools before starting a complex project:

```bash
#!/bin/bash
# System requirement checker script
# Save this as: scripts/check_requirements.sh

echo "🔍 Checking system requirements for RAG Chatbot Platform..."
echo "=================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track if all requirements are met
ALL_REQUIREMENTS_MET=true

# Function to check command existence
check_command() {
    local cmd=$1
    local required_version=$2
    local install_hint=$3
    
    if command -v $cmd &> /dev/null; then
        version=$(eval "$cmd --version 2>&1 | head -n 1")
        echo -e "${GREEN}✓${NC} $cmd found: $version"
        
        # Version check if required
        if [ ! -z "$required_version" ]; then
            # Extract version number and compare
            # This is simplified - you might need more complex parsing
            current_version=$(echo $version | grep -oE '[0-9]+\.[0-9]+' | head -1)
            if [ "$(printf '%s\n' "$required_version" "$current_version" | sort -V | head -n1)" = "$required_version" ]; then
                echo -e "  ${GREEN}Version OK${NC} (required: $required_version+)"
            else
                echo -e "  ${YELLOW}Warning:${NC} Version $current_version might be too old (required: $required_version+)"
            fi
        fi
    else
        echo -e "${RED}✗${NC} $cmd not found"
        echo -e "  ${YELLOW}Install hint:${NC} $install_hint"
        ALL_REQUIREMENTS_MET=false
    fi
}

# Check system resources
echo -e "\n📊 System Resources:"
echo "-------------------"

# Check RAM
total_ram=$(free -g | awk '/^Mem:/{print $2}')
if [ $total_ram -ge 16 ]; then
    echo -e "${GREEN}✓${NC} RAM: ${total_ram}GB (16GB+ recommended)"
else
    echo -e "${YELLOW}⚠${NC} RAM: ${total_ram}GB (16GB recommended, system might be slow)"
fi

# Check disk space
available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ $available_space -ge 50 ]; then
    echo -e "${GREEN}✓${NC} Disk Space: ${available_space}GB available (50GB+ recommended)"
else
    echo -e "${YELLOW}⚠${NC} Disk Space: ${available_space}GB available (50GB recommended)"
fi

# Check CPU cores
cpu_cores=$(nproc)
if [ $cpu_cores -ge 4 ]; then
    echo -e "${GREEN}✓${NC} CPU Cores: $cpu_cores (4+ recommended)"
else
    echo -e "${YELLOW}⚠${NC} CPU Cores: $cpu_cores (4+ recommended for better performance)"
fi

# Check required software
echo -e "\n🔧 Required Software:"
echo "--------------------"

check_command "docker" "20.10" "https://docs.docker.com/get-docker/"
check_command "docker-compose" "2.0" "Included with Docker Desktop"
check_command "python3" "3.11" "https://www.python.org/downloads/"
check_command "git" "" "https://git-scm.com/downloads"
check_command "curl" "" "Usually pre-installed"
check_command "jq" "" "apt-get install jq or brew install jq"

# Check Python packages
echo -e "\n🐍 Python Environment:"
echo "--------------------"

if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python version: $python_version"
    
    # Check if pip is available
    if python3 -m pip --version &> /dev/null; then
        echo -e "${GREEN}✓${NC} pip is available"
    else
        echo -e "${RED}✗${NC} pip not found"
        echo -e "  ${YELLOW}Install:${NC} python3 -m ensurepip"
        ALL_REQUIREMENTS_MET=false
    fi
else
    echo -e "${RED}✗${NC} Python 3 not found"
    ALL_REQUIREMENTS_MET=false
fi

# Check Docker daemon
echo -e "\n🐳 Docker Status:"
echo "----------------"

if docker info &> /dev/null; then
    echo -e "${GREEN}✓${NC} Docker daemon is running"
    
    # Check Docker Compose
    if docker compose version &> /dev/null; then
        compose_version=$(docker compose version --short)
        echo -e "${GREEN}✓${NC} Docker Compose V2: $compose_version"
    elif docker-compose --version &> /dev/null; then
        echo -e "${YELLOW}⚠${NC} Using Docker Compose V1 (V2 recommended)"
    fi
else
    echo -e "${RED}✗${NC} Docker daemon is not running"
    echo -e "  ${YELLOW}Start Docker:${NC} sudo systemctl start docker (Linux) or open Docker Desktop"
    ALL_REQUIREMENTS_MET=false
fi

# Check ports
echo -e "\n🔌 Port Availability:"
echo "--------------------"

check_port() {
    local port=$1
    local service=$2
    
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠${NC} Port $port is in use (needed for $service)"
        echo -e "  ${YELLOW}Check:${NC} lsof -i :$port"
    else
        echo -e "${GREEN}✓${NC} Port $port is available ($service)"
    fi
}

check_port 8000 "FastAPI"
check_port 5432 "PostgreSQL"
check_port 27017 "MongoDB"
check_port 6379 "Redis"
check_port 9042 "ScyllaDB"

# Final verdict
echo -e "\n📋 Summary:"
echo "----------"

if [ "$ALL_REQUIREMENTS_MET" = true ]; then
    echo -e "${GREEN}✅ All requirements met! You're ready to proceed.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some requirements are missing. Please install them before continuing.${NC}"
    exit 1
fi
```

### **9.2 Step-by-Step Installation Process**

Now let's walk through the complete installation process. Each step is explained so you understand not just what to do, but why:

```bash
# Step 1: Create project structure
# This creates our workspace with all necessary directories

echo "📁 Creating project structure..."
mkdir -p rag-chatbot-platform/{app,scripts,tests,migrations,docs}
cd rag-chatbot-platform

# Create subdirectories for better organization
mkdir -p app/{api,core,database,services,models,utils}
mkdir -p app/api/endpoints
mkdir -p tests/{unit,integration,e2e}
mkdir -p scripts/{setup,maintenance,monitoring}

echo "✓ Project structure created"

# Step 2: Initialize Git repository
# Version control is essential from day one

echo "🔧 Initializing Git repository..."
git init
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Environment
.env
.env.local
.env.*.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/
*.log

# Data
data/
*.db
*.sqlite

# Docker
.docker/

# OS
.DS_Store
Thumbs.db
EOF

git add .
git commit -m "Initial commit: Project structure"
echo "✓ Git repository initialized"
```

### **9.3 Environment Configuration Deep Dive**

Understanding environment configuration is crucial. Let's create a comprehensive configuration system:

```python
# app/core/config_manager.py

import os
from typing import Any, Dict, Optional
from pathlib import Path
import json
import yaml
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class ConfigurationManager:
    """
    Sophisticated configuration management system.
    
    This manager handles:
    - Multiple environment support (dev, staging, prod)
    - Secret encryption/decryption
    - Configuration validation
    - Dynamic reloading
    - Environment variable override
    
    Think of this as the control panel for your entire system.
    """
    
    def __init__(self, environment: str = None):
        """
        Initialize configuration for specific environment.
        
        The configuration hierarchy:
        1. Default values (base config)
        2. Environment-specific config file
        3. Environment variables
        4. Runtime overrides
        """
        
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config_dir = Path('config')
        self.secrets_key = self._load_secrets_key()
        
        # Load configuration in order of precedence
        self.config = {}
        self._load_base_config()
        self._load_environment_config()
        self._load_env_variables()
        self._validate_config()
        
        logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _load_base_config(self):
        """
        Load base configuration that applies to all environments.
        
        Base config contains:
        - Default values
        - Feature flags
        - System constants
        """
        
        base_config = {
            # Application settings
            'app': {
                'name': 'RAG Chatbot Platform',
                'version': '5.0.0',
                'debug': False,
                'testing': False
            },
            
            # API configuration
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'cors_origins': ['http://localhost:3000'],
                'request_timeout': 30,
                'max_request_size': 10 * 1024 * 1024  # 10MB
            },
            
            # Security defaults
            'security': {
                'jwt_algorithm': 'HS256',
                'jwt_expiration_minutes': 30,
                'bcrypt_rounds': 12,
                'password_min_length': 8,
                'max_login_attempts': 5,
                'account_lock_duration': 1800  # 30 minutes
            },
            
            # Database defaults
            'databases': {
                'postgres': {
                    'pool_size': 20,
                    'max_overflow': 40,
                    'pool_timeout': 30,
                    'echo': False
                },
                'mongodb': {
                    'max_pool_size': 50,
                    'min_pool_size': 10,
                    'server_selection_timeout': 5000
                },
                'redis': {
                    'max_connections': 50,
                    'decode_responses': True,
                    'socket_timeout': 5
                },
                'scylla': {
                    'consistency_level': 'LOCAL_QUORUM',
                    'request_timeout': 10
                }
            },
            
            # AI Model defaults
            'models': {
                'embedding': {
                    'name': 'sentence-transformers/all-mpnet-base-v2',
                    'dimension': 768,
                    'max_length': 512,
                    'batch_size': 32
                },
                'generation': {
                    'name': 'Qwen/Qwen3-1.7B',
                    'max_tokens': 512,
                    'temperature': 0.7,
                    'top_p': 0.9
                }
            },
            
            # Feature flags
            'features': {
                'enable_caching': True,
                'enable_rate_limiting': True,
                'enable_quota_enforcement': True,
                'enable_audit_logging': True,
                'enable_analytics': True
            },
            
            # Monitoring
            'monitoring': {
                'metrics_enabled': True,
                'tracing_enabled': True,
                'log_level': 'INFO',
                'slow_query_threshold': 1000  # milliseconds
            }
        }
        
        self.config = base_config
    
    def _load_environment_config(self):
        """
        Load environment-specific configuration.
        
        Environment configs override base config with
        environment-specific values like:
        - Database connections
        - External service URLs
        - Feature toggles
        """
        
        config_file = self.config_dir / f"{self.environment}.yaml"
        
        if not config_file.exists():
            logger.warning(f"No config file found for environment: {self.environment}")
            return
        
        with open(config_file, 'r') as f:
            env_config = yaml.safe_load(f)
        
        # Merge with base config (environment config takes precedence)
        self.config = self._deep_merge(self.config, env_config)
    
    def _load_env_variables(self):
        """
        Override configuration with environment variables.
        
        Environment variables follow the pattern:
        APP_SECTION_KEY=value
        
        Examples:
        APP_API_PORT=8080
        APP_SECURITY_JWT_SECRET=my-secret
        APP_DATABASES_POSTGRES_HOST=db.example.com
        """
        
        prefix = 'APP_'
        
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            
            # Parse environment variable into config path
            # APP_API_PORT -> ['api', 'port']
            path = key[len(prefix):].lower().split('_')
            
            # Set value in config
            self._set_nested_value(self.config, path, self._parse_env_value(value))
    
    def _parse_env_value(self, value: str) -> Any:
        """
        Parse environment variable value to appropriate type.
        
        Handles:
        - Booleans: 'true'/'false'
        - Numbers: integers and floats
        - JSON: arrays and objects
        - Strings: everything else
        """
        
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # JSON
        if value.startswith('[') or value.startswith('{'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String
        return value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path.
        
        Path uses dot notation: 'api.port' or 'databases.postgres.host'
        """
        
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_secret(self, name: str) -> str:
        """
        Get decrypted secret value.
        
        Secrets are stored encrypted and only decrypted when needed.
        This prevents accidental logging or exposure.
        """
        
        encrypted_value = self.get(f'secrets.{name}')
        if not encrypted_value:
            raise ValueError(f"Secret '{name}' not found")
        
        # Decrypt secret
        fernet = Fernet(self.secrets_key)
        decrypted = fernet.decrypt(encrypted_value.encode()).decode()
        
        return decrypted
    
    def _validate_config(self):
        """
        Validate configuration for required values and correct types.
        
        This prevents the application from starting with invalid config,
        catching errors early rather than at runtime.
        """
        
        required_fields = [
            'app.name',
            'api.port',
            'security.jwt_algorithm'
        ]
        
        for field in required_fields:
            if self.get(field) is None:
                raise ValueError(f"Required configuration field missing: {field}")
        
        # Validate types
        if not isinstance(self.get('api.port'), int):
            raise TypeError("api.port must be an integer")
        
        if not isinstance(self.get('api.cors_origins'), list):
            raise TypeError("api.cors_origins must be a list")
        
        # Validate value ranges
        if self.get('api.port') < 1 or self.get('api.port') > 65535:
            raise ValueError("api.port must be between 1 and 65535")
        
        if self.get('security.bcrypt_rounds') < 4 or self.get('security.bcrypt_rounds') > 31:
            raise ValueError("security.bcrypt_rounds must be between 4 and 31")
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Override values take precedence, but nested dictionaries
        are merged rather than replaced entirely.
        """
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result

# Create environment-specific config files
def create_config_files():
    """
    Create template configuration files for different environments.
    """
    
    os.makedirs('config', exist_ok=True)
    
    # Development configuration
    dev_config = """
# Development Environment Configuration
app:
  debug: true
  
api:
  host: localhost
  
databases:
  postgres:
    host: localhost
    port: 5432
    database: chatbot_dev
    username: chatbot_user
    echo: true  # Log SQL queries in development
    
  mongodb:
    uri: mongodb://localhost:27017/chatbot_dev
    
  redis:
    host: localhost
    port: 6379
    db: 0
    
  scylla:
    hosts:
      - 127.0.0.1
    keyspace: chatbot_dev

features:
  enable_debug_endpoints: true
  enable_swagger_ui: true

monitoring:
  log_level: DEBUG
"""
    
    with open('config/development.yaml', 'w') as f:
        f.write(dev_config)
    
    # Production configuration
    prod_config = """
# Production Environment Configuration
app:
  debug: false
  
api:
  host: 0.0.0.0
  workers: 8
  
databases:
  postgres:
    host: ${DB_HOST}  # Use environment variable
    port: 5432
    database: chatbot_prod
    username: ${DB_USER}
    password: ${DB_PASSWORD}
    echo: false
    pool_size: 50
    
  mongodb:
    uri: ${MONGO_URI}
    
  redis:
    host: ${REDIS_HOST}
    port: 6379
    password: ${REDIS_PASSWORD}
    db: 0
    
  scylla:
    hosts:
      - ${SCYLLA_NODE1}
      - ${SCYLLA_NODE2}
      - ${SCYLLA_NODE3}
    keyspace: chatbot_prod
    username: ${SCYLLA_USER}
    password: ${SCYLLA_PASSWORD}

features:
  enable_debug_endpoints: false
  enable_swagger_ui: false

monitoring:
  log_level: WARNING
  metrics_endpoint: ${METRICS_URL}
  tracing_endpoint: ${TRACING_URL}
"""
    
    with open('config/production.yaml', 'w') as f:
        f.write(prod_config)
    
    print("✓ Configuration files created")

# Usage example
if __name__ == "__main__":
    # Create config files
    create_config_files()
    
    # Load configuration
    config = ConfigurationManager(environment='development')
    
    # Access configuration values
    print(f"API Port: {config.get('api.port')}")
    print(f"Database Host: {config.get('databases.postgres.host')}")
    print(f"Debug Mode: {config.get('app.debug')}")
```

---

## **10. Production Deployment: Going Live**

Taking a system from development to production requires careful planning and execution. This section covers everything you need to deploy the system reliably at scale.

### **10.1 Production Architecture Overview**

Production deployment is fundamentally different from development. We need to consider high availability, disaster recovery, and zero-downtime deployments:

```yaml
# kubernetes/production-architecture.yaml

# This defines our production Kubernetes architecture
# Think of this as the blueprint for our production infrastructure

apiVersion: v1
kind: Namespace
metadata:
  name: rag-chatbot-prod
  labels:
    environment: production
    app: rag-chatbot

---
# ConfigMap for application configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: rag-chatbot-prod
data:
  # Non-sensitive configuration
  ENVIRONMENT: "production"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  LOG_LEVEL: "INFO"
  
  # Feature flags
  ENABLE_CACHING: "true"
  ENABLE_RATE_LIMITING: "true"
  ENABLE_QUOTA_ENFORCEMENT: "true"
  
  # Model configuration
  EMBEDDING_MODEL: "sentence-transformers/all-mpnet-base-v2"
  GENERATION_MODEL: "Qwen/Qwen3-1.7B"
  
---
# Secret for sensitive data
apiVersion: v1
kind: Secret
metadata:
  name: app-secrets
  namespace: rag-chatbot-prod
type: Opaque
stringData:
  # Database credentials
  POSTGRES_PASSWORD: "your-secure-password"
  MONGO_URI: "mongodb+srv://user:pass@cluster.mongodb.net/"
  REDIS_PASSWORD: "your-redis-password"
  
  # Security keys
  JWT_SECRET_KEY: "your-jwt-secret-key"
  ENCRYPTION_KEY: "your-encryption-key"
  
  # External services
  STRIPE_API_KEY: "sk_live_..."
  OPENAI_API_KEY: "sk-..."

---
# Deployment for the main API service
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-deployment
  namespace: rag-chatbot-prod
  labels:
    app: rag-chatbot-api
    version: v5.0.0
spec:
  replicas: 3  # Start with 3 replicas
  
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Allow 1 extra pod during update
      maxUnavailable: 0  # Don't take down pods during update
  
  selector:
    matchLabels:
      app: rag-chatbot-api
  
  template:
    metadata:
      labels:
        app: rag-chatbot-api
        version: v5.0.0
      annotations:
        # Force pod restart on config change
        configHash: "{{ .Values.configHash }}"
    
    spec:
      # Service account for pod permissions
      serviceAccountName: rag-chatbot-sa
      
      # Anti-affinity to spread pods across nodes
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - rag-chatbot-api
              topologyKey: kubernetes.io/hostname
      
      # Init container to wait for databases
      initContainers:
      - name: wait-for-db
        image: busybox:1.35
        command: ['sh', '-c']
        args:
        - |
          echo "Waiting for databases to be ready..."
          # Check PostgreSQL
          until nc -z postgres-service 5432; do
            echo "Waiting for PostgreSQL..."
            sleep 2
          done
          # Check MongoDB
          until nc -z mongodb-service 27017; do
            echo "Waiting for MongoDB..."
            sleep 2
          done
          echo "All databases are ready!"
      
      containers:
      - name: api
        image: your-registry/rag-chatbot-api:v5.0.0
        
        # Resource requests and limits
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            ephemeral-storage: "1Gi"
          limits:
            memory: "4Gi"
            cpu: "2000m"
            ephemeral-storage: "2Gi"
        
        # Container ports
        ports:
        - containerPort: 8000
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        
        # Environment variables from ConfigMap and Secrets
        envFrom:
        - configMapRef:
            name: app-config
        - secretRef:
            name: app-secrets
        
        # Additional environment variables
        env:
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        
        # Health checks
        livenessProbe:
          httpGet:
            path: /health/live
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /health/ready
            port: http
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3
        
        # Startup probe for slow-starting containers
        startupProbe:
          httpGet:
            path: /health/startup
            port: http
          initialDelaySeconds: 0
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 30  # 5 minutes to start
        
        # Volume mounts
        volumeMounts:
        - name: models-cache
          mountPath: /app/models
        - name: temp-storage
          mountPath: /tmp
        
        # Security context
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
      
      # Volumes
      volumes:
      - name: models-cache
        persistentVolumeClaim:
          claimName: models-pvc
      - name: temp-storage
        emptyDir:
          sizeLimit: 1Gi

---
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: rag-chatbot-prod
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-deployment
  
  minReplicas: 3
  maxReplicas: 20
  
  metrics:
  # Scale based on CPU
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  
  # Scale based on memory
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  
  # Scale based on custom metrics (requests per second)
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100  # Double the pods
        periodSeconds: 60
      - type: Pods
        value: 4    # Or add 4 pods
        periodSeconds: 60
      selectPolicy: Max  # Use whichever is larger
    
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 minutes before scaling down
      policies:
      - type: Percent
        value: 50   # Remove half the pods
        periodSeconds: 60
      - type: Pods
        value: 2    # Or remove 2 pods
        periodSeconds: 60
      selectPolicy: Min  # Use whichever is smaller
```

### **10.2 Deployment Pipeline and CI/CD**

A robust deployment pipeline ensures that code changes are tested and deployed safely:

```python
# .github/workflows/production-deployment.yaml

name: Production Deployment Pipeline

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

env:
  REGISTRY: your-registry.com
  IMAGE_NAME: rag-chatbot-api
  CLUSTER_NAME: production-cluster
  NAMESPACE: rag-chatbot-prod

jobs:
  # Job 1: Run tests
  test:
    name: Run Test Suite
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        cache: 'pip'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit -v --cov=app --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration -v
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
  
  # Job 2: Security scanning
  security:
    name: Security Scanning
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Run Trivy security scan
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Run Bandit security linter
      run: |
        pip install bandit
        bandit -r app/ -f json -o bandit-results.json
    
    - name: Check for secrets
      uses: trufflesecurity/trufflehog@main
      with:
        path: ./
        base: main
        head: HEAD
  
  # Job 3: Build and push Docker image
  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [test, security]
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Log in to registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ secrets.REGISTRY_USERNAME }}
        password: ${{ secrets.REGISTRY_PASSWORD }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          VERSION=${{ github.sha }}
          BUILD_DATE=${{ github.event.head_commit.timestamp }}
  
  # Job 4: Deploy to staging
  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build
    environment:
      name: staging
      url: https://staging.chatbot.example.com
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'latest'
    
    - name: Set up Kubernetes config
      run: |
        echo "${{ secrets.STAGING_KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Deploy to staging
      run: |
        # Update image tag in deployment
        kubectl set image deployment/api-deployment \
          api=${{ needs.build.outputs.image-tag }} \
          -n rag-chatbot-staging
        
        # Wait for rollout to complete
        kubectl rollout status deployment/api-deployment \
          -n rag-chatbot-staging \
          --timeout=10m
    
    - name: Run smoke tests
      run: |
        # Wait for service to be ready
        sleep 30
        
        # Run smoke tests against staging
        python scripts/smoke_tests.py --url https://staging.chatbot.example.com
  
  # Job 5: Deploy to production
  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [build, deploy-staging]
    environment:
      name: production
      url: https://chatbot.example.com
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
    
    - name: Configure kubectl
      uses: azure/setup-kubectl@v3
    
    - name: Set up Kubernetes config
      run: |
        echo "${{ secrets.PROD_KUBECONFIG }}" | base64 -d > kubeconfig
        export KUBECONFIG=kubeconfig
    
    - name: Create deployment backup
      run: |
        # Backup current deployment state
        kubectl get deployment api-deployment \
          -n ${{ env.NAMESPACE }} \
          -o yaml > deployment-backup-$(date +%Y%m%d-%H%M%S).yaml
    
    - name: Deploy to production (canary)
      run: |
        # Deploy as canary first (10% traffic)
        kubectl apply -f kubernetes/canary-deployment.yaml
        
        # Update canary image
        kubectl set image deployment/api-deployment-canary \
          api=${{ needs.build.outputs.image-tag }} \
          -n ${{ env.NAMESPACE }}
        
        # Wait for canary rollout
        kubectl rollout status deployment/api-deployment-canary \
          -n ${{ env.NAMESPACE }} \
          --timeout=10m
    
    - name: Validate canary deployment
      run: |
        # Run validation tests against canary
        python scripts/canary_validation.py \
          --url https://chatbot.example.com \
          --canary-weight 0.1
        
        # Check error rates
        python scripts/check_metrics.py \
          --deployment canary \
          --threshold 0.01
    
    - name: Promote to full deployment
      if: success()
      run: |
        # Update main deployment
        kubectl set image deployment/api-deployment \
          api=${{ needs.build.outputs.image-tag }} \
          -n ${{ env.NAMESPACE }}
        
        # Wait for rollout
        kubectl rollout status deployment/api-deployment \
          -n ${{ env.NAMESPACE }} \
          --timeout=15m
        
        # Remove canary deployment
        kubectl delete deployment api-deployment-canary \
          -n ${{ env.NAMESPACE }}
    
    - name: Rollback on failure
      if: failure()
      run: |
        # Rollback to previous version
        kubectl rollout undo deployment/api-deployment \
          -n ${{ env.NAMESPACE }}
        
        # Remove canary
        kubectl delete deployment api-deployment-canary \
          -n ${{ env.NAMESPACE }} \
          --ignore-not-found=true
        
        # Alert team
        curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
          -H 'Content-Type: application/json' \
          -d '{"text":"⚠️ Production deployment failed and was rolled back!"}'
```

## **11. Performance Optimization: Speed & Efficiency**

Performance optimization is like tuning a high-performance engine – every small improvement compounds to create a dramatically better experience. Let's explore how to make our system not just work, but work brilliantly fast.

### **11.1 Understanding Performance Bottlenecks**

Before we can optimize, we need to understand where our system spends its time. Think of this as being a detective, looking for clues about what's slowing things down:

```python
class PerformanceProfiler:
    """
    Comprehensive performance profiling system.
    
    This profiler helps us understand where time is being spent in our system,
    like having a stopwatch at every stage of a relay race.
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.active_traces = {}
        
    @contextmanager
    def trace(self, operation_name: str, metadata: Dict = None):
        """
        Context manager for tracing operation performance.
        
        Usage:
        with profiler.trace('database_query', {'query_type': 'vector_search'}):
            # Your operation here
            results = await database.search(...)
        
        This records how long the operation takes and stores it for analysis.
        """
        
        trace_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        
        # Store active trace information
        self.active_traces[trace_id] = {
            'operation': operation_name,
            'start_time': start_time,
            'metadata': metadata or {}
        }
        
        try:
            yield trace_id
        finally:
            # Calculate duration
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Store metric
            self.metrics[operation_name].append({
                'duration_ms': duration_ms,
                'timestamp': datetime.utcnow(),
                'metadata': metadata or {},
                'trace_id': trace_id
            })
            
            # Clean up active trace
            del self.active_traces[trace_id]
            
            # Log slow operations
            if duration_ms > self._get_threshold(operation_name):
                logger.warning(
                    f"Slow operation detected: {operation_name} took {duration_ms:.2f}ms",
                    extra={'trace_id': trace_id, 'metadata': metadata}
                )
    
    def _get_threshold(self, operation_name: str) -> float:
        """
        Get performance threshold for different operations.
        
        Different operations have different acceptable durations:
        - Database queries: 100ms
        - API calls: 500ms
        - ML inference: 1000ms
        - Cache operations: 10ms
        """
        
        thresholds = {
            'database_query': 100,
            'api_call': 500,
            'ml_inference': 1000,
            'cache_operation': 10,
            'embedding_generation': 50,
            'vector_search': 200,
            'text_generation': 3000
        }
        
        # Default threshold
        return thresholds.get(operation_name, 200)
    
    def get_statistics(self, operation_name: str) -> Dict:
        """
        Calculate statistics for a specific operation.
        
        This gives us insights like:
        - How fast is it on average?
        - What's the slowest it's been?
        - Is it getting slower over time?
        """
        
        if operation_name not in self.metrics:
            return {}
        
        durations = [m['duration_ms'] for m in self.metrics[operation_name]]
        
        return {
            'count': len(durations),
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'p95': np.percentile(durations, 95),
            'p99': np.percentile(durations, 99),
            'min': min(durations),
            'max': max(durations),
            'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0
        }
    
    def identify_bottlenecks(self) -> List[Dict]:
        """
        Identify the biggest performance bottlenecks.
        
        This is like finding the slowest runner in a relay team -
        they're the ones we need to help improve first.
        """
        
        bottlenecks = []
        
        for operation_name in self.metrics:
            stats = self.get_statistics(operation_name)
            
            # Calculate bottleneck score
            # Higher score = bigger bottleneck
            bottleneck_score = (
                stats['p95'] * 0.5 +  # How slow is it usually?
                stats['max'] * 0.3 +   # How slow can it get?
                stats['count'] * 0.2   # How often does it happen?
            )
            
            bottlenecks.append({
                'operation': operation_name,
                'score': bottleneck_score,
                'stats': stats,
                'impact': self._calculate_impact(operation_name, stats)
            })
        
        # Sort by bottleneck score
        bottlenecks.sort(key=lambda x: x['score'], reverse=True)
        
        return bottlenecks
    
    def _calculate_impact(self, operation_name: str, stats: Dict) -> str:
        """
        Calculate the business impact of this bottleneck.
        
        Some slowdowns matter more than others:
        - User-facing operations are critical
        - Background tasks can be slower
        - Admin operations have lower priority
        """
        
        # Total time spent on this operation
        total_time = stats['count'] * stats['mean']
        
        # Criticality multiplier based on operation type
        criticality = {
            'chat_response': 10,      # User-facing, critical
            'vector_search': 8,        # Part of user flow
            'database_query': 7,       # Can block operations
            'cache_operation': 5,      # Should be fast
            'background_task': 2,      # Not user-facing
            'admin_operation': 1       # Low priority
        }.get(operation_name, 3)
        
        impact_score = total_time * criticality
        
        if impact_score > 10000:
            return "CRITICAL"
        elif impact_score > 5000:
            return "HIGH"
        elif impact_score > 1000:
            return "MEDIUM"
        else:
            return "LOW"
```

### **11.2 Caching Strategy - The Art of Remembering**

Caching is one of the most powerful optimization techniques. It's like having a really good memory – you don't need to figure out the same thing twice:

```python
class IntelligentCacheManager:
    """
    Multi-tier caching system with intelligent invalidation.
    
    Think of this as a hierarchical memory system:
    - L1 Cache: In-memory (fastest, smallest)
    - L2 Cache: Redis (fast, medium)
    - L3 Cache: Database cache tables (slower, largest)
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.l1_cache = LRUCache(maxsize=config.l1_size)  # In-memory
        self.l2_cache = RedisCache(config.redis_config)   # Redis
        self.l3_cache = DatabaseCache(config.db_config)   # Database
        
        # Cache statistics for optimization
        self.stats = CacheStatistics()
        
    async def get(
        self,
        key: str,
        cache_level: str = 'auto'
    ) -> Optional[Any]:
        """
        Get value from cache with multi-tier lookup.
        
        The lookup process:
        1. Check L1 (memory) - microseconds
        2. Check L2 (Redis) - milliseconds
        3. Check L3 (database) - tens of milliseconds
        4. Cache miss - need to compute
        
        Each level acts as a fallback for the previous one.
        """
        
        # Generate cache key with version
        versioned_key = self._version_key(key)
        
        # Try L1 cache (in-memory)
        value = self.l1_cache.get(versioned_key)
        if value is not None:
            self.stats.record_hit('l1', key)
            return value
        
        # Try L2 cache (Redis)
        value = await self.l2_cache.get(versioned_key)
        if value is not None:
            # Promote to L1
            self.l1_cache.set(versioned_key, value)
            self.stats.record_hit('l2', key)
            return value
        
        # Try L3 cache (database)
        if cache_level != 'memory_only':
            value = await self.l3_cache.get(versioned_key)
            if value is not None:
                # Promote to L1 and L2
                self.l1_cache.set(versioned_key, value)
                await self.l2_cache.set(versioned_key, value, ttl=300)
                self.stats.record_hit('l3', key)
                return value
        
        # Cache miss
        self.stats.record_miss(key)
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_level: str = 'all'
    ):
        """
        Set value in cache with intelligent tier selection.
        
        Caching decisions based on value characteristics:
        - Small, frequently accessed: All levels
        - Large, infrequently accessed: L3 only
        - Temporary: L1 and L2 only
        - Critical: All levels with extended TTL
        """
        
        versioned_key = self._version_key(key)
        
        # Determine caching strategy
        strategy = self._determine_cache_strategy(key, value)
        
        # Apply to appropriate cache levels
        if strategy.use_l1:
            self.l1_cache.set(versioned_key, value)
        
        if strategy.use_l2:
            await self.l2_cache.set(
                versioned_key,
                value,
                ttl=ttl or strategy.default_ttl
            )
        
        if strategy.use_l3 and cache_level != 'memory_only':
            await self.l3_cache.set(
                versioned_key,
                value,
                ttl=ttl or strategy.default_ttl * 10  # Longer TTL for L3
            )
        
        # Update statistics
        self.stats.record_set(key, strategy)
    
    def _determine_cache_strategy(self, key: str, value: Any) -> CacheStrategy:
        """
        Determine optimal caching strategy based on data characteristics.
        
        Factors considered:
        - Size of data
        - Access frequency (from historical stats)
        - Data type (embeddings, responses, etc.)
        - Computational cost to regenerate
        """
        
        # Estimate size
        size_bytes = self._estimate_size(value)
        
        # Get access frequency from statistics
        access_frequency = self.stats.get_access_frequency(key)
        
        # Determine data type from key pattern
        data_type = self._infer_data_type(key)
        
        # Build strategy
        strategy = CacheStrategy()
        
        # Small, frequently accessed data goes everywhere
        if size_bytes < 1024 and access_frequency > 10:  # < 1KB, accessed often
            strategy.use_l1 = True
            strategy.use_l2 = True
            strategy.use_l3 = True
            strategy.default_ttl = 3600  # 1 hour
        
        # Embeddings: Medium size, moderate access
        elif data_type == 'embedding':
            strategy.use_l1 = False  # Too large for memory
            strategy.use_l2 = True
            strategy.use_l3 = True
            strategy.default_ttl = 86400  # 24 hours (embeddings don't change)
        
        # Chat responses: Variable size, frequent access
        elif data_type == 'response':
            strategy.use_l1 = size_bytes < 10240  # Only if < 10KB
            strategy.use_l2 = True
            strategy.use_l3 = False  # Responses can be regenerated
            strategy.default_ttl = 1800  # 30 minutes
        
        # Search results: Large, moderate access
        elif data_type == 'search_results':
            strategy.use_l1 = False
            strategy.use_l2 = True
            strategy.use_l3 = True
            strategy.default_ttl = 900  # 15 minutes
        
        # Default strategy
        else:
            strategy.use_l1 = size_bytes < 5120  # < 5KB
            strategy.use_l2 = True
            strategy.use_l3 = size_bytes < 102400  # < 100KB
            strategy.default_ttl = 600  # 10 minutes
        
        return strategy
    
    async def invalidate(self, pattern: str):
        """
        Invalidate cache entries matching pattern.
        
        Cache invalidation is one of the hardest problems in computer science.
        We need to be careful to invalidate everything that might be stale,
        but not more than necessary.
        """
        
        # Invalidate from all levels
        self._invalidate_l1(pattern)
        await self._invalidate_l2(pattern)
        await self._invalidate_l3(pattern)
        
        # Log invalidation
        logger.info(f"Cache invalidated for pattern: {pattern}")
        
        # Update statistics
        self.stats.record_invalidation(pattern)
    
    def _version_key(self, key: str) -> str:
        """
        Add version to cache key for easy invalidation.
        
        By including a version in the key, we can invalidate
        entire categories of cached data by changing the version.
        """
        
        # Get current version for key type
        key_type = key.split(':')[0]
        version = self.config.versions.get(key_type, 'v1')
        
        return f"{version}:{key}"
```

### **11.3 Database Query Optimization**

Database queries are often the biggest bottleneck. Let's optimize them like a surgeon – precise, careful, and effective:

```python
class QueryOptimizer:
    """
    Database query optimization system.
    
    This optimizer acts like a GPS for database queries,
    finding the fastest route to get your data.
    """
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.query_cache = {}
        self.execution_stats = defaultdict(list)
        
    async def optimize_query(self, query: str, params: Dict = None) -> OptimizedQuery:
        """
        Optimize a database query for better performance.
        
        Optimization techniques:
        1. Query rewriting for efficiency
        2. Index usage validation
        3. Join order optimization
        4. Subquery elimination
        5. Predicate pushdown
        """
        
        # Parse the query
        parsed = self._parse_query(query)
        
        # Check if we've seen this query pattern before
        query_pattern = self._extract_pattern(parsed)
        if query_pattern in self.query_cache:
            cached_optimization = self.query_cache[query_pattern]
            return cached_optimization.apply(params)
        
        # Analyze query execution plan
        execution_plan = await self._get_execution_plan(query)
        
        # Identify optimization opportunities
        optimizations = []
        
        # Check for missing indexes
        if self._needs_index(execution_plan):
            optimizations.append(self._suggest_index(parsed))
        
        # Check for inefficient joins
        if self._has_inefficient_joins(execution_plan):
            optimizations.append(self._optimize_joins(parsed))
        
        # Check for N+1 queries
        if self._detect_n_plus_one(query_pattern):
            optimizations.append(self._batch_queries(parsed))
        
        # Apply optimizations
        optimized_query = self._apply_optimizations(query, optimizations)
        
        # Cache the optimization
        self.query_cache[query_pattern] = OptimizedQuery(
            original=query,
            optimized=optimized_query,
            optimizations=optimizations
        )
        
        return optimized_query
    
    async def _get_execution_plan(self, query: str) -> ExecutionPlan:
        """
        Get the execution plan for a query.
        
        The execution plan tells us:
        - Which indexes are being used
        - How tables are being joined
        - Where the most time is spent
        - How many rows are being processed
        """
        
        # PostgreSQL example
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
        
        result = await self.db.execute(explain_query)
        plan = json.loads(result[0]['QUERY PLAN'])
        
        return ExecutionPlan(plan)
    
    def _needs_index(self, plan: ExecutionPlan) -> bool:
        """
        Determine if query would benefit from an index.
        
        Signs that an index is needed:
        - Sequential scans on large tables
        - High cost estimates
        - Large number of rows filtered after retrieval
        """
        
        for node in plan.nodes:
            if node.type == 'Seq Scan' and node.rows > 10000:
                # Sequential scan on large table
                return True
            
            if node.filter_removed_rows > node.rows * 0.5:
                # Filtering out more than half the rows
                return True
            
            if node.cost > 1000:
                # High cost operation
                return True
        
        return False
    
    def _suggest_index(self, parsed_query: ParsedQuery) -> Optimization:
        """
        Suggest indexes that would improve query performance.
        
        Index selection is an art:
        - Columns in WHERE clauses need indexes
        - Join columns benefit from indexes
        - Sort columns should be indexed
        - But too many indexes slow down writes
        """
        
        suggestions = []
        
        # Analyze WHERE clauses
        for condition in parsed_query.where_conditions:
            if condition.is_equality:
                # Equality conditions benefit from B-tree indexes
                suggestions.append(
                    f"CREATE INDEX idx_{condition.table}_{condition.column} "
                    f"ON {condition.table}({condition.column});"
                )
            elif condition.is_range:
                # Range conditions also use B-tree
                suggestions.append(
                    f"CREATE INDEX idx_{condition.table}_{condition.column}_range "
                    f"ON {condition.table}({condition.column});"
                )
            elif condition.is_pattern:
                # Pattern matching needs special indexes
                suggestions.append(
                    f"CREATE INDEX idx_{condition.table}_{condition.column}_pattern "
                    f"ON {condition.table}({condition.column} text_pattern_ops);"
                )
        
        # Analyze JOIN conditions
        for join in parsed_query.joins:
            suggestions.append(
                f"CREATE INDEX idx_{join.table}_{join.column} "
                f"ON {join.table}({join.column});"
            )
        
        # Composite indexes for multi-column conditions
        if len(parsed_query.where_conditions) > 1:
            columns = [c.column for c in parsed_query.where_conditions 
                      if c.table == parsed_query.main_table]
            if columns:
                suggestions.append(
                    f"CREATE INDEX idx_{parsed_query.main_table}_composite "
                    f"ON {parsed_query.main_table}({', '.join(columns)});"
                )
        
        return Optimization(
            type='index',
            suggestions=suggestions,
            estimated_improvement='50-90% reduction in query time'
        )
```

### **11.4 Model Optimization - Making AI Faster**

Making machine learning models run faster is crucial for a responsive system. Let's explore optimization techniques that can dramatically improve inference speed:

```python
class ModelOptimizationPipeline:
    """
    Comprehensive model optimization pipeline.
    
    Think of this as tuning a Formula 1 car - every optimization
    makes the model a little bit faster, and they all add up.
    """
    
    def __init__(self):
        self.optimization_techniques = []
        self.benchmark_results = {}
        
    async def optimize_model(
        self,
        model: Any,
        model_type: str,
        target_hardware: str = 'auto'
    ) -> OptimizedModel:
        """
        Apply multiple optimization techniques to a model.
        
        The optimization journey:
        1. Quantization - Reduce precision for speed
        2. Pruning - Remove unnecessary connections
        3. Knowledge Distillation - Create smaller student model
        4. Operator Fusion - Combine operations
        5. Hardware Optimization - Tune for specific hardware
        """
        
        # Detect target hardware if not specified
        if target_hardware == 'auto':
            target_hardware = self._detect_hardware()
        
        logger.info(f"Starting optimization for {model_type} on {target_hardware}")
        
        # Benchmark original model
        original_metrics = await self._benchmark_model(model, "original")
        
        # Apply optimizations based on model type and hardware
        optimized_model = model
        
        # 1. Quantization - Convert to lower precision
        if self._should_quantize(model_type, target_hardware):
            optimized_model = await self._apply_quantization(
                optimized_model,
                target_hardware
            )
            metrics = await self._benchmark_model(optimized_model, "quantized")
            self._log_improvement("Quantization", original_metrics, metrics)
        
        # 2. Model Pruning - Remove redundant weights
        if self._should_prune(model_type):
            optimized_model = await self._apply_pruning(
                optimized_model,
                sparsity=0.5  # Remove 50% of weights
            )
            metrics = await self._benchmark_model(optimized_model, "pruned")
            self._log_improvement("Pruning", original_metrics, metrics)
        
        # 3. Graph Optimization - Optimize computation graph
        if self._supports_graph_optimization(model_type):
            optimized_model = await self._optimize_graph(optimized_model)
            metrics = await self._benchmark_model(optimized_model, "graph_optimized")
            self._log_improvement("Graph Optimization", original_metrics, metrics)
        
        # 4. Hardware-Specific Optimization
        optimized_model = await self._hardware_specific_optimization(
            optimized_model,
            target_hardware
        )
        
        # Final benchmarking
        final_metrics = await self._benchmark_model(optimized_model, "final")
        
        # Create optimization report
        report = self._create_optimization_report(
            original_metrics,
            final_metrics,
            self.optimization_techniques
        )
        
        return OptimizedModel(
            model=optimized_model,
            metrics=final_metrics,
            report=report
        )
    
    async def _apply_quantization(self, model: Any, hardware: str) -> Any:
        """
        Apply quantization to reduce model precision.
        
        Quantization is like using fewer decimal places in calculations -
        less precise but much faster:
        
        FP32 (32-bit float): 3.141592653589793
        FP16 (16-bit float): 3.14159
        INT8 (8-bit integer): 3
        
        The trick is finding the right balance between speed and accuracy.
        """
        
        if hardware == 'gpu':
            # GPU supports mixed precision well
            return self._apply_mixed_precision(model)
        elif hardware == 'cpu':
            # CPU benefits from INT8 quantization
            return self._apply_int8_quantization(model)
        elif hardware == 'edge':
            # Edge devices need aggressive quantization
            return self._apply_int4_quantization(model)
        else:
            # Default to INT8
            return self._apply_int8_quantization(model)
    
    def _apply_int8_quantization(self, model: Any) -> Any:
        """
        Convert model to INT8 precision.
        
        This can give 2-4x speedup with minimal accuracy loss.
        """
        
        import torch.quantization as quant
        
        # Prepare model for quantization
        model.eval()
        
        # Fuse operations (Conv + BatchNorm + ReLU -> ConvBNReLU)
        model = quant.fuse_modules(model, [['conv', 'bn', 'relu']])
        
        # Prepare for quantization
        model.qconfig = quant.get_default_qconfig('fbgemm')
        quant.prepare(model, inplace=True)
        
        # Calibrate with representative data
        calibration_data = self._get_calibration_data()
        with torch.no_grad():
            for batch in calibration_data:
                model(batch)
        
        # Convert to quantized model
        quantized_model = quant.convert(model, inplace=False)
        
        return quantized_model
    
    async def _apply_pruning(self, model: Any, sparsity: float) -> Any:
        """
        Remove redundant weights from the model.
        
        Pruning is like removing unnecessary parts from a machine -
        it still works the same but runs more efficiently.
        
        We identify weights that contribute least to the output
        and set them to zero, making the model sparse.
        """
        
        import torch.nn.utils.prune as prune
        
        # Apply magnitude-based pruning
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prune.l1_unstructured(
                    module,
                    name='weight',
                    amount=sparsity
                )
        
        # Remove pruning reparametrization for inference
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                prune.remove(module, 'weight')
        
        return model
    
    async def _benchmark_model(self, model: Any, stage: str) -> ModelMetrics:
        """
        Benchmark model performance.
        
        We measure:
        - Inference latency (how fast)
        - Throughput (requests per second)
        - Memory usage (RAM/VRAM)
        - Accuracy (quality of outputs)
        """
        
        metrics = ModelMetrics()
        
        # Prepare test data
        test_inputs = self._get_benchmark_inputs()
        
        # Measure latency
        latencies = []
        for input_data in test_inputs:
            start = time.perf_counter()
            _ = model(input_data)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
        
        metrics.latency_p50 = np.percentile(latencies, 50)
        metrics.latency_p95 = np.percentile(latencies, 95)
        metrics.latency_p99 = np.percentile(latencies, 99)
        
        # Measure throughput
        batch_size = 32
        start = time.perf_counter()
        for _ in range(100):
            _ = model(self._get_batch(batch_size))
        end = time.perf_counter()
        
        metrics.throughput = (100 * batch_size) / (end - start)
        
        # Measure memory usage
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = model(test_inputs[0])
            metrics.memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            import psutil
            process = psutil.Process()
            metrics.memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Store results
        self.benchmark_results[stage] = metrics
        
        return metrics
```

---

## **12. Operations & Reliability: Keeping It Running**

Running a production system is like conducting an orchestra – every component needs to work in harmony, and when something goes wrong, you need to know immediately and fix it gracefully. Let's explore how to build a system that practically runs itself.

### **12.1 Observability - Seeing Everything**

You can't fix what you can't see. Observability gives us eyes into every corner of our system:

```python
class ObservabilityPlatform:
    """
    Comprehensive observability system providing complete visibility.
    
    Think of this as having cameras, microphones, and sensors throughout
    a building - we can see, hear, and sense everything that's happening.
    """
    
    def __init__(self, config: ObservabilityConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.trace_collector = TraceCollector()
        self.log_aggregator = LogAggregator()
        self.alert_manager = AlertManager()
        
    def setup_instrumentation(self, app: FastAPI):
        """
        Instrument the application for complete observability.
        
        This adds monitoring to every part of the application,
        like adding sensors to every component of a machine.
        """
        
        # Add request tracing
        @app.middleware("http")
        async def trace_requests(request: Request, call_next):
            """
            Trace every HTTP request through the system.
            
            This creates a breadcrumb trail showing exactly
            how each request flows through our services.
            """
            
            # Generate or extract trace ID
            trace_id = request.headers.get('X-Trace-ID', str(uuid.uuid4()))
            
            # Start span for this request
            with self.trace_collector.start_span(
                name=f"{request.method} {request.url.path}",
                trace_id=trace_id
            ) as span:
                # Add request metadata
                span.set_attributes({
                    'http.method': request.method,
                    'http.url': str(request.url),
                    'http.scheme': request.url.scheme,
                    'http.host': request.url.hostname,
                    'http.target': request.url.path,
                    'http.user_agent': request.headers.get('User-Agent'),
                    'peer.ip': request.client.host
                })
                
                # Time the request
                start_time = time.perf_counter()
                
                try:
                    # Process request
                    response = await call_next(request)
                    
                    # Record success metrics
                    duration = time.perf_counter() - start_time
                    
                    span.set_attributes({
                        'http.status_code': response.status_code,
                        'http.response_time_ms': duration * 1000
                    })
                    
                    # Record metrics
                    self.metrics_collector.record_request(
                        method=request.method,
                        endpoint=request.url.path,
                        status=response.status_code,
                        duration=duration
                    )
                    
                    # Add trace ID to response
                    response.headers['X-Trace-ID'] = trace_id
                    
                    return response
                    
                except Exception as e:
                    # Record error metrics
                    duration = time.perf_counter() - start_time
                    
                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    
                    # Record error metrics
                    self.metrics_collector.record_error(
                        method=request.method,
                        endpoint=request.url.path,
                        error_type=type(e).__name__,
                        duration=duration
                    )
                    
                    # Re-raise exception
                    raise
        
        # Add database query monitoring
        self._instrument_database()
        
        # Add cache monitoring
        self._instrument_cache()
        
        # Add model inference monitoring
        self._instrument_models()
        
        # Add custom business metrics
        self._setup_business_metrics()
    
    def _instrument_database(self):
        """
        Add monitoring to all database operations.
        
        This helps us understand:
        - Which queries are slow
        - How often queries fail
        - Connection pool health
        - Transaction patterns
        """
        
        from sqlalchemy import event
        
        @event.listens_for(Engine, "before_execute")
        def receive_before_execute(conn, clauseelement, multiparams, params, execution_options):
            """Start timing database query."""
            conn.info['query_start_time'] = time.perf_counter()
            conn.info['query_text'] = str(clauseelement)
        
        @event.listens_for(Engine, "after_execute")
        def receive_after_execute(conn, clauseelement, multiparams, params, execution_options, result):
            """Record database query metrics."""
            duration = time.perf_counter() - conn.info.get('query_start_time', 0)
            
            # Parse query type
            query_text = conn.info.get('query_text', '')
            query_type = self._get_query_type(query_text)
            
            # Record metrics
            self.metrics_collector.record_database_query(
                query_type=query_type,
                duration=duration,
                rows_affected=result.rowcount if hasattr(result, 'rowcount') else 0
            )
            
            # Log slow queries
            if duration > self.config.slow_query_threshold:
                logger.warning(
                    f"Slow query detected",
                    extra={
                        'query': query_text[:500],  # Truncate long queries
                        'duration_ms': duration * 1000,
                        'query_type': query_type
                    }
                )
    
    def create_dashboard(self) -> Dashboard:
        """
        Create comprehensive monitoring dashboard.
        
        A good dashboard is like a car's instrument panel -
        it shows you everything important at a glance.
        """
        
        dashboard = Dashboard(title="RAG Chatbot System")
        
        # System Health Overview
        dashboard.add_panel(
            HealthPanel(
                title="System Health",
                metrics=[
                    "up{job='api'}",  # Service availability
                    "rate(errors_total[5m])",  # Error rate
                    "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))"  # P95 latency
                ]
            )
        )
        
        # Request Metrics
        dashboard.add_panel(
            GraphPanel(
                title="Request Rate",
                query="rate(http_requests_total[5m])",
                legend=['{{method}} {{endpoint}}']
            )
        )
        
        # Database Performance
        dashboard.add_panel(
            GraphPanel(
                title="Database Query Performance",
                queries=[
                    "histogram_quantile(0.95, rate(db_query_duration_seconds_bucket[5m]))",
                    "histogram_quantile(0.99, rate(db_query_duration_seconds_bucket[5m]))"
                ],
                legend=['P95', 'P99']
            )
        )
        
        # Model Performance
        dashboard.add_panel(
            GraphPanel(
                title="Model Inference Latency",
                queries=[
                    "histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket{model='embedding'}[5m]))",
                    "histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket{model='generation'}[5m]))"
                ],
                legend=['Embedding', 'Generation']
            )
        )
        
        # Business Metrics
        dashboard.add_panel(
            StatPanel(
                title="Active Users",
                query="count(increase(user_activity_total[1h]) > 0)"
            )
        )
        
        dashboard.add_panel(
            GraphPanel(
                title="Messages per Hour",
                query="sum(rate(chat_messages_total[1h])) * 3600"
            )
        )
        
        # Resource Usage
        dashboard.add_panel(
            GraphPanel(
                title="Resource Utilization",
                queries=[
                    "avg(rate(container_cpu_usage_seconds_total[5m])) * 100",
                    "avg(container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100"
                ],
                legend=['CPU %', 'Memory %']
            )
        )
        
        return dashboard
```

### **12.2 Incident Response - When Things Go Wrong**

Even the best systems have problems. What matters is how quickly and effectively we respond:

```python
class IncidentResponseSystem:
    """
    Automated incident detection and response system.
    
    This is like having a team of doctors on call 24/7,
    ready to diagnose and treat any problems that arise.
    """
    
    def __init__(self):
        self.incident_detector = IncidentDetector()
        self.response_orchestrator = ResponseOrchestrator()
        self.runbook_executor = RunbookExecutor()
        self.notification_service = NotificationService()
        
    async def monitor_and_respond(self):
        """
        Continuously monitor for incidents and respond automatically.
        
        The incident response loop:
        1. Detect anomalies or failures
        2. Classify the incident
        3. Execute appropriate runbook
        4. Notify relevant people
        5. Track resolution
        """
        
        while True:
            # Check for incidents
            incidents = await self.incident_detector.check_for_incidents()
            
            for incident in incidents:
                # Don't create duplicate incidents
                if not await self._is_duplicate(incident):
                    await self._handle_incident(incident)
            
            # Check every 30 seconds
            await asyncio.sleep(30)
    
    async def _handle_incident(self, incident: Incident):
        """
        Handle a detected incident.
        
        Incident handling is like emergency response:
        1. Assess severity
        2. Take immediate action
        3. Call for help if needed
        4. Document everything
        """
        
        logger.info(f"Incident detected: {incident.title}")
        
        # Create incident record
        incident_id = await self._create_incident_record(incident)
        
        # Determine severity
        severity = await self._assess_severity(incident)
        incident.severity = severity
        
        # Get appropriate runbook
        runbook = self._get_runbook(incident.type, severity)
        
        if runbook:
            # Execute automated response
            logger.info(f"Executing runbook: {runbook.name}")
            
            try:
                result = await self.runbook_executor.execute(
                    runbook,
                    incident
                )
                
                if result.success:
                    logger.info(f"Runbook executed successfully")
                    
                    # Check if incident is resolved
                    if await self._verify_resolution(incident):
                        await self._mark_resolved(incident_id)
                        await self._notify_resolution(incident)
                        return
                else:
                    logger.warning(f"Runbook execution failed: {result.error}")
                    
            except Exception as e:
                logger.error(f"Error executing runbook: {e}")
        
        # If we get here, automated response didn't resolve the issue
        # Escalate to humans
        await self._escalate_incident(incident)
    
    async def _assess_severity(self, incident: Incident) -> Severity:
        """
        Assess the severity of an incident.
        
        Severity levels:
        - CRITICAL: System down or data loss risk
        - HIGH: Major functionality impaired
        - MEDIUM: Performance degraded
        - LOW: Minor issues
        
        Factors considered:
        - Number of users affected
        - Business impact
        - Data integrity risk
        - Security implications
        """
        
        severity_score = 0
        
        # Check user impact
        affected_users = await self._count_affected_users(incident)
        if affected_users > 1000:
            severity_score += 40
        elif affected_users > 100:
            severity_score += 20
        elif affected_users > 10:
            severity_score += 10
        
        # Check business impact
        if incident.affects_billing:
            severity_score += 30
        if incident.affects_authentication:
            severity_score += 30
        if incident.affects_core_functionality:
            severity_score += 25
        
        # Check data risk
        if incident.data_loss_risk:
            severity_score += 50
        if incident.data_corruption_risk:
            severity_score += 40
        
        # Check security implications
        if incident.security_breach_possible:
            severity_score += 50
        
        # Map score to severity
        if severity_score >= 70:
            return Severity.CRITICAL
        elif severity_score >= 50:
            return Severity.HIGH
        elif severity_score >= 30:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _get_runbook(self, incident_type: str, severity: Severity) -> Runbook:
        """
        Get the appropriate runbook for an incident.
        
        Runbooks are like recipes for fixing problems -
        step-by-step instructions that can be executed automatically.
        """
        
        runbooks = {
            'database_connection_failure': DatabaseConnectionRunbook(),
            'high_error_rate': HighErrorRateRunbook(),
            'memory_leak': MemoryLeakRunbook(),
            'slow_response_time': SlowResponseRunbook(),
            'redis_unavailable': RedisFailoverRunbook(),
            'model_loading_failure': ModelRecoveryRunbook(),
            'rate_limit_breach': RateLimitRunbook(),
            'disk_space_low': DiskCleanupRunbook()
        }
        
        return runbooks.get(incident_type)
```

### **12.3 Disaster Recovery - Preparing for the Worst**

Disaster recovery is about preparing for catastrophic failures. It's like having fire drills - you hope you never need them, but you practice anyway:

```python
class DisasterRecoveryManager:
    """
    Comprehensive disaster recovery system.
    
    This ensures we can recover from even the worst failures:
    - Data center outages
    - Database corruption
    - Ransomware attacks
    - Human errors
    """
    
    def __init__(self, config: DRConfig):
        self.config = config
        self.backup_manager = BackupManager()
        self.replication_manager = ReplicationManager()
        self.failover_coordinator = FailoverCoordinator()
        
    async def setup_disaster_recovery(self):
        """
        Set up comprehensive disaster recovery capabilities.
        
        DR setup includes:
        1. Automated backups
        2. Cross-region replication
        3. Failover procedures
        4. Recovery testing
        """
        
        # Configure automated backups
        await self._setup_backups()
        
        # Set up replication
        await self._setup_replication()
        
        # Configure failover
        await self._setup_failover()
        
        # Schedule recovery testing
        await self._schedule_dr_tests()
    
    async def _setup_backups(self):
        """
        Configure automated backup system.
        
        Backup strategy follows 3-2-1 rule:
        - 3 copies of data
        - 2 different storage media
        - 1 offsite backup
        """
        
        # Database backups
        backup_configs = [
            BackupConfig(
                name="postgres_backup",
                source="postgresql://primary",
                destinations=[
                    "s3://backups/postgres/",
                    "azure://backups/postgres/",
                    "local://backup/postgres/"
                ],
                schedule="0 */6 * * *",  # Every 6 hours
                retention=RetentionPolicy(
                    hourly=24,    # Keep 24 hourly backups
                    daily=30,     # Keep 30 daily backups
                    weekly=12,    # Keep 12 weekly backups
                    monthly=12    # Keep 12 monthly backups
                ),
                encryption=True,
                compression=True
            ),
            BackupConfig(
                name="mongodb_backup",
                source="mongodb://primary",
                destinations=[
                    "s3://backups/mongodb/",
                    "azure://backups/mongodb/"
                ],
                schedule="0 */4 * * *",  # Every 4 hours
                retention=RetentionPolicy(
                    hourly=12,
                    daily=14,
                    weekly=8
                ),
                encryption=True,
                compression=True
            )
        ]
        
        for config in backup_configs:
            await self.backup_manager.configure_backup(config)
        
        logger.info("Backup system configured")
    
    async def perform_failover(self, reason: str):
        """
        Perform failover to disaster recovery site.
        
        Failover is like switching to a backup generator when
        the power goes out - it needs to be fast and reliable.
        """
        
        logger.critical(f"Initiating failover: {reason}")
        
        # Create failover plan
        plan = FailoverPlan()
        
        # Step 1: Verify DR site readiness
        dr_ready = await self._verify_dr_site()
        if not dr_ready:
            raise FailoverError("DR site not ready")
        
        # Step 2: Stop writes to primary
        await self._stop_primary_writes()
        plan.record_step("Primary writes stopped")
        
        # Step 3: Ensure replication is caught up
        await self._wait_for_replication_sync(timeout=300)
        plan.record_step("Replication synchronized")
        
        # Step 4: Promote DR databases
        await self._promote_dr_databases()
        plan.record_step("DR databases promoted")
        
        # Step 5: Update DNS to point to DR site
        await self._update_dns_records()
        plan.record_step("DNS updated")
        
        # Step 6: Verify DR site functionality
        await self._verify_dr_functionality()
        plan.record_step("DR site verified")
        
        # Step 7: Notify stakeholders
        await self._notify_failover_complete()
        
        logger.info(f"Failover completed successfully in {plan.duration} seconds")
        
        return plan
    
    async def test_recovery_procedures(self):
        """
        Test disaster recovery procedures.
        
        Regular DR testing is critical - it's like fire drills,
        you want to find problems during practice, not during a real emergency.
        """
        
        test_results = []
        
        # Test 1: Backup restoration
        logger.info("Testing backup restoration...")
        
        test_backup = await self.backup_manager.get_latest_backup("postgres")
        restore_result = await self._test_restore(test_backup, target="test_db")
        
        test_results.append({
            'test': 'backup_restoration',
            'success': restore_result.success,
            'duration': restore_result.duration,
            'notes': restore_result.notes
        })
        
        # Test 2: Replication lag
        logger.info("Testing replication lag...")
        
        lag = await self.replication_manager.measure_lag()
        
        test_results.append({
            'test': 'replication_lag',
            'success': lag < self.config.max_acceptable_lag,
            'lag_seconds': lag,
            'threshold': self.config.max_acceptable_lag
        })
        
        # Test 3: Failover simulation
        logger.info("Testing failover simulation...")
        
        # Create isolated test environment
        test_env = await self._create_test_environment()
        
        try:
            failover_result = await self._simulate_failover(test_env)
            
            test_results.append({
                'test': 'failover_simulation',
                'success': failover_result.success,
                'duration': failover_result.duration,
                'issues': failover_result.issues
            })
            
        finally:
            await self._cleanup_test_environment(test_env)
        
        # Generate test report
        report = self._generate_dr_test_report(test_results)
        
        # Alert if any tests failed
        if any(not r['success'] for r in test_results):
            await self._alert_dr_test_failure(report)
        
        return report
```

---

## **13. Scaling Strategies: Growing the System**

Scaling is about growing gracefully. Like a city that expands from a village to a metropolis, our system needs to handle growth without losing its essential qualities.

### **13.1 Horizontal Scaling - Adding More Servers**

Horizontal scaling is like adding more checkout lanes at a grocery store - more servers to handle more requests:

```python
class HorizontalScalingOrchestrator:
    """
    Manages horizontal scaling of services.
    
    Think of this as a smart traffic controller that adds more roads
    when traffic gets heavy and removes them when it's light.
    """
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.metrics_provider = MetricsProvider()
        self.orchestrator = KubernetesOrchestrator()
        self.load_balancer = LoadBalancerManager()
        
    async def auto_scale_service(self, service_name: str):
        """
        Automatically scale a service based on metrics.
        
        Scaling decisions consider:
        - Current load (CPU, memory, requests)
        - Historical patterns
        - Cost constraints
        - Business priorities
        """
        
        while True:
            # Get current metrics
            metrics = await self.metrics_provider.get_service_metrics(service_name)
            
            # Get current scale
            current_replicas = await self.orchestrator.get_replica_count(service_name)
            
            # Calculate desired scale
            desired_replicas = await self._calculate_desired_replicas(
                service_name,
                metrics,
                current_replicas
            )
            
            # Apply scaling decision
            if desired_replicas != current_replicas:
                await self._scale_service(
                    service_name,
                    current_replicas,
                    desired_replicas
                )
            
            # Wait before next check
            await asyncio.sleep(self.config.scaling_interval)
    
    async def _calculate_desired_replicas(
        self,
        service_name: str,
        metrics: ServiceMetrics,
        current_replicas: int
    ) -> int:
        """
        Calculate the desired number of replicas.
        
        This is like deciding how many cashiers to have open:
        - Too few: long lines (high latency)
        - Too many: wasted resources (high cost)
        - Just right: good service at reasonable cost
        """
        
        # Get service configuration
        service_config = self.config.services[service_name]
        
        # Initialize scaling score
        scaling_score = 0
        
        # Check CPU utilization
        if metrics.cpu_utilization > service_config.scale_up_cpu_threshold:
            # CPU is high, need more instances
            cpu_pressure = (metrics.cpu_utilization - service_config.scale_up_cpu_threshold) / 10
            scaling_score += cpu_pressure
            
        elif metrics.cpu_utilization < service_config.scale_down_cpu_threshold:
            # CPU is low, might need fewer instances
            cpu_slack = (service_config.scale_down_cpu_threshold - metrics.cpu_utilization) / 10
            scaling_score -= cpu_slack
        
        # Check memory utilization
        if metrics.memory_utilization > service_config.scale_up_memory_threshold:
            memory_pressure = (metrics.memory_utilization - service_config.scale_up_memory_threshold) / 10
            scaling_score += memory_pressure
            
        elif metrics.memory_utilization < service_config.scale_down_memory_threshold:
            memory_slack = (service_config.scale_down_memory_threshold - metrics.memory_utilization) / 10
            scaling_score -= memory_slack
        
        # Check request rate
        requests_per_instance = metrics.request_rate / current_replicas
        optimal_requests_per_instance = service_config.optimal_requests_per_instance
        
        if requests_per_instance > optimal_requests_per_instance * 1.2:
            # Too many requests per instance
            request_pressure = (requests_per_instance / optimal_requests_per_instance) - 1
            scaling_score += request_pressure
            
        elif requests_per_instance < optimal_requests_per_instance * 0.5:
            # Too few requests per instance
            request_slack = 1 - (requests_per_instance / optimal_requests_per_instance)
            scaling_score -= request_slack
        
        # Check response time
        if metrics.p95_latency > service_config.latency_slo:
            # Latency is too high
            latency_pressure = (metrics.p95_latency - service_config.latency_slo) / service_config.latency_slo
            scaling_score += latency_pressure * 2  # Weight latency heavily
        
        # Consider time of day patterns
        time_factor = await self._get_time_based_factor(service_name)
        scaling_score *= time_factor
        
        # Consider cost constraints
        if await self._would_exceed_budget(service_name, current_replicas + 1):
            # Don't scale up if it would exceed budget
            scaling_score = min(scaling_score, 0)
        
        # Calculate desired replicas
        if scaling_score > 0.3:
            # Scale up
            scale_factor = min(2.0, 1.0 + scaling_score)  # Max 2x scale up
            desired = int(current_replicas * scale_factor)
            
        elif scaling_score < -0.3:
            # Scale down
            scale_factor = max(0.5, 1.0 + scaling_score)  # Max 50% scale down
            desired = int(current_replicas * scale_factor)
            
        else:
            # No change needed
            desired = current_replicas
        
        # Apply min/max constraints
        desired = max(service_config.min_replicas, desired)
        desired = min(service_config.max_replicas, desired)
        
        return desired
    
    async def _scale_service(
        self,
        service_name: str,
        current_replicas: int,
        desired_replicas: int
    ):
        """
        Scale a service to the desired number of replicas.
        
        Scaling needs to be done carefully:
        - Gradual scaling to avoid thundering herd
        - Health checks before routing traffic
        - Graceful shutdown of removed instances
        """
        
        logger.info(
            f"Scaling {service_name} from {current_replicas} to {desired_replicas} replicas"
        )
        
        if desired_replicas > current_replicas:
            # Scaling up
            await self._scale_up(service_name, current_replicas, desired_replicas)
            
        else:
            # Scaling down
            await self._scale_down(service_name, current_replicas, desired_replicas)
    
    async def _scale_up(
        self,
        service_name: str,
        current: int,
        desired: int
    ):
        """
        Scale up a service by adding instances.
        
        Scaling up is like opening new checkout lanes:
        1. Open the new lane (start instance)
        2. Get it ready (warm up)
        3. Put up the "open" sign (add to load balancer)
        """
        
        # Calculate how many instances to add
        to_add = desired - current
        
        # Add instances gradually
        for i in range(to_add):
            # Start new instance
            instance_id = await self.orchestrator.add_instance(service_name)
            
            # Wait for instance to be ready
            await self._wait_for_instance_ready(instance_id)
            
            # Warm up the instance (send test requests)
            await self._warm_up_instance(instance_id)
            
            # Add to load balancer
            await self.load_balancer.add_target(service_name, instance_id)
            
            logger.info(f"Added instance {instance_id} to {service_name}")
            
            # Brief pause between instances to avoid thundering herd
            if i < to_add - 1:
                await asyncio.sleep(5)
    
    async def _scale_down(
        self,
        service_name: str,
        current: int,
        desired: int
    ):
        """
        Scale down a service by removing instances.
        
        Scaling down needs extra care:
        1. Stop sending new traffic (remove from load balancer)
        2. Wait for existing requests to complete
        3. Gracefully shut down
        """
        
        # Calculate how many instances to remove
        to_remove = current - desired
        
        # Get instances to remove (prefer least loaded)
        instances = await self._select_instances_to_remove(service_name, to_remove)
        
        for instance_id in instances:
            # Remove from load balancer first
            await self.load_balancer.remove_target(service_name, instance_id)
            
            # Wait for in-flight requests to complete
            await self._drain_instance(instance_id)
            
            # Gracefully shutdown
            await self.orchestrator.remove_instance(instance_id)
            
            logger.info(f"Removed instance {instance_id} from {service_name}")
```

### **13.2 Vertical Scaling - Making Servers Bigger**

Sometimes you need bigger servers, not more servers. This is like upgrading from a small car to a truck when you need to carry more:

```python
class VerticalScalingManager:
    """
    Manages vertical scaling (resizing) of services.
    
    Vertical scaling is like upgrading your computer's RAM or CPU -
    same machine, more power.
    """
    
    async def resize_instance(
        self,
        instance_id: str,
        new_size: InstanceSize
    ):
        """
        Resize an instance to a new size.
        
        This process is delicate because it usually requires downtime:
        1. Create replacement with new size
        2. Migrate traffic
        3. Decommission old instance
        """
        
        logger.info(f"Resizing instance {instance_id} to {new_size}")
        
        # Get current instance details
        current_instance = await self.get_instance(instance_id)
        
        # Create replacement instance with new size
        replacement_id = await self.create_instance(
            service=current_instance.service,
            size=new_size,
            config=current_instance.config
        )
        
        # Wait for replacement to be ready
        await self._wait_for_instance_ready(replacement_id)
        
        # Migrate traffic gradually (canary approach)
        await self._migrate_traffic(
            from_instance=instance_id,
            to_instance=replacement_id,
            strategy='gradual'
        )
        
        # Verify replacement is handling traffic well
        await self._verify_instance_health(replacement_id, duration=300)
        
        # Decommission old instance
        await self.decommission_instance(instance_id)
        
        logger.info(f"Successfully resized to {new_size}")
        
        return replacement_id
```

---

## **14. Testing & Evaluation: Quality Assurance**

Testing is how we ensure our system works correctly, not just in ideal conditions but in the messy reality of production. Think of testing as training for a marathon - you practice different scenarios so you're ready for race day.

### **14.1 Comprehensive Testing Strategy**

Testing isn't just about finding bugs; it's about building confidence that our system will work when it matters:

```python
class ComprehensiveTestingFramework:
    """
    Multi-layered testing framework ensuring system quality.
    
    Like a safety inspection for a car, we test every component
    individually and then test how they work together.
    """
    
    def __init__(self):
        self.test_suites = {
            'unit': UnitTestSuite(),
            'integration': IntegrationTestSuite(),
            'e2e': EndToEndTestSuite(),
            'performance': PerformanceTestSuite(),
            'security': SecurityTestSuite(),
            'chaos': ChaosTestSuite()
        }
        
    async def run_test_pipeline(self, stage: str = 'all') -> TestReport:
        """
        Run comprehensive test pipeline.
        
        Testing stages like layers of verification:
        1. Unit tests - Do individual parts work?
        2. Integration tests - Do parts work together?
        3. E2E tests - Does the whole system work?
        4. Performance tests - Is it fast enough?
        5. Security tests - Is it secure?
        6. Chaos tests - Can it handle failures?
        """
        
        report = TestReport()
        
        if stage in ['all', 'unit']:
            # Unit tests - Test individual functions
            logger.info("Running unit tests...")
            unit_results = await self._run_unit_tests()
            report.add_results('unit', unit_results)
            
            if not unit_results.passed:
                # Don't continue if unit tests fail
                logger.error("Unit tests failed, stopping pipeline")
                return report
        
        if stage in ['all', 'integration']:
            # Integration tests - Test service interactions
            logger.info("Running integration tests...")
            integration_results = await self._run_integration_tests()
            report.add_results('integration', integration_results)
        
        if stage in ['all', 'e2e']:
            # End-to-end tests - Test complete user flows
            logger.info("Running end-to-end tests...")
            e2e_results = await self._run_e2e_tests()
            report.add_results('e2e', e2e_results)
        
        if stage in ['all', 'performance']:
            # Performance tests - Test speed and scalability
            logger.info("Running performance tests...")
            perf_results = await self._run_performance_tests()
            report.add_results('performance', perf_results)
        
        if stage in ['all', 'security']:
            # Security tests - Test for vulnerabilities
            logger.info("Running security tests...")
            security_results = await self._run_security_tests()
            report.add_results('security', security_results)
        
        if stage in ['all', 'chaos']:
            # Chaos tests - Test failure handling
            logger.info("Running chaos tests...")
            chaos_results = await self._run_chaos_tests()
            report.add_results('chaos', chaos_results)
        
        return report
    
    async def _run_unit_tests(self) -> TestResults:
        """
        Run unit tests for individual components.
        
        Unit tests are like testing each ingredient before cooking -
        making sure each part is good before combining them.
        """
        
        results = TestResults()
        
        # Test authentication functions
        auth_tests = [
            TestCase(
                name="test_password_hashing",
                function=self._test_password_hashing,
                expected_outcome="Passwords hash correctly and verify"
            ),
            TestCase(
                name="test_jwt_generation",
                function=self._test_jwt_generation,
                expected_outcome="JWTs generate and validate correctly"
            ),
            TestCase(
                name="test_permission_checking",
                function=self._test_permission_checking,
                expected_outcome="Permissions correctly enforced"
            )
        ]
        
        for test in auth_tests:
            try:
                await test.function()
                results.add_pass(test.name)
            except AssertionError as e:
                results.add_failure(test.name, str(e))
            except Exception as e:
                results.add_error(test.name, str(e))
        
        # Test embedding functions
        embedding_tests = [
            TestCase(
                name="test_embedding_generation",
                function=self._test_embedding_generation,
                expected_outcome="Embeddings are correct dimension and normalized"
            ),
            TestCase(
                name="test_embedding_similarity",
                function=self._test_embedding_similarity,
                expected_outcome="Similar texts have high similarity scores"
            )
        ]
        
        for test in embedding_tests:
            try:
                await test.function()
                results.add_pass(test.name)
            except Exception as e:
                results.add_failure(test.name, str(e))
        
        return results
    
    async def _run_e2e_tests(self) -> TestResults:
        """
        Run end-to-end tests simulating real user flows.
        
        E2E tests are like a dress rehearsal - testing the complete
        experience from the user's perspective.
        """
        
        results = TestResults()
        
        # Test complete chat flow
        async def test_chat_flow():
            """Test a complete chat interaction from login to response."""
            
            # Create test user
            user = await create_test_user(
                email="test@example.com",
                password="Test123!",
                plan="pro"
            )
            
            # Login
            token = await login_user(user.email, "Test123!")
            assert token is not None, "Login failed"
            
            # Send chat message
            response = await send_chat_message(
                token=token,
                message="What is machine learning?",
                enable_rag=True
            )
            
            # Verify response
            assert response.status_code == 200, "Chat request failed"
            assert len(response.answer) > 0, "Empty response"
            assert response.sources is not None, "No sources provided"
            assert response.tokens_used > 0, "Token counting failed"
            
            # Verify usage was recorded
            usage = await get_user_usage(token)
            assert usage.messages_this_month > 0, "Usage not recorded"
            
            # Cleanup
            await delete_test_user(user.id)
        
        try:
            await test_chat_flow()
            results.add_pass("complete_chat_flow")
        except AssertionError as e:
            results.add_failure("complete_chat_flow", str(e))
        except Exception as e:
            results.add_error("complete_chat_flow", str(e))
        
        return results
    
    async def _run_performance_tests(self) -> TestResults:
        """
        Run performance tests to ensure the system meets speed requirements.
        
        Performance testing is like timing a runner - 
        we need to know how fast the system is under different conditions.
        """
        
        results = TestResults()
        
        # Test response time under load
        async def test_response_time_under_load():
            """Ensure response times stay acceptable under load."""
            
            # Generate load (100 concurrent users)
            concurrent_users = 100
            requests_per_user = 10
            
            async def user_session():
                """Simulate a single user session."""
                token = await get_test_token()
                
                response_times = []
                for _ in range(requests_per_user):
                    start = time.perf_counter()
                    
                    await send_chat_message(
                        token=token,
                        message="Test message",
                        enable_rag=True
                    )
                    
                    response_times.append(time.perf_counter() - start)
                
                return response_times
            
            # Run concurrent sessions
            tasks = [user_session() for _ in range(concurrent_users)]
            all_response_times = await asyncio.gather(*tasks)
            
            # Flatten response times
            response_times = [t for times in all_response_times for t in times]
            
            # Calculate percentiles
            p50 = np.percentile(response_times, 50)
            p95 = np.percentile(response_times, 95)
            p99 = np.percentile(response_times, 99)
            
            # Check against SLOs
            assert p50 < 1.0, f"P50 latency {p50}s exceeds 1s SLO"
            assert p95 < 3.0, f"P95 latency {p95}s exceeds 3s SLO"
            assert p99 < 5.0, f"P99 latency {p99}s exceeds 5s SLO"
            
            results.add_metric("response_time_p50", p50)
            results.add_metric("response_time_p95", p95)
            results.add_metric("response_time_p99", p99)
        
        try:
            await test_response_time_under_load()
            results.add_pass("response_time_under_load")
        except AssertionError as e:
            results.add_failure("response_time_under_load", str(e))
        
        return results
```

### **14.2 RAG Quality Evaluation**

Evaluating RAG quality is unique because we need to assess both retrieval accuracy and generation quality:

```python
class RAGQualityEvaluator:
    """
    Comprehensive RAG system evaluation framework.
    
    Evaluating RAG is like judging a chef who must both
    select the right ingredients (retrieval) and cook them well (generation).
    """
    
    def __init__(self):
        self.retrieval_evaluator = RetrievalEvaluator()
        self.generation_evaluator = GenerationEvaluator()
        self.end_to_end_evaluator = EndToEndEvaluator()
        
    async def evaluate_rag_system(
        self,
        test_dataset: TestDataset
    ) -> RAGEvaluation:
        """
        Comprehensive evaluation of RAG system quality.
        
        We evaluate:
        1. Retrieval quality - Are we finding the right documents?
        2. Generation quality - Are we producing good answers?
        3. End-to-end quality - Is the complete system working well?
        """
        
        evaluation = RAGEvaluation()
        
        # Evaluate retrieval
        logger.info("Evaluating retrieval quality...")
        retrieval_results = await self._evaluate_retrieval(test_dataset)
        evaluation.retrieval = retrieval_results
        
        # Evaluate generation
        logger.info("Evaluating generation quality...")
        generation_results = await self._evaluate_generation(test_dataset)
        evaluation.generation = generation_results
        
        # Evaluate end-to-end
        logger.info("Evaluating end-to-end quality...")
        e2e_results = await self._evaluate_end_to_end(test_dataset)
        evaluation.end_to_end = e2e_results
        
        # Calculate overall score
        evaluation.overall_score = self._calculate_overall_score(evaluation)
        
        return evaluation
    
    async def _evaluate_retrieval(self, dataset: TestDataset) -> RetrievalMetrics:
        """
        Evaluate retrieval quality.
        
        Retrieval metrics:
        - Precision: Of retrieved docs, how many are relevant?
        - Recall: Of relevant docs, how many did we retrieve?
        - MRR: How high is the first relevant result?
        - NDCG: How well are results ranked?
        """
        
        metrics = RetrievalMetrics()
        
        for query in dataset.queries:
            # Get retrieved documents
            retrieved = await self.retrieval_evaluator.retrieve(
                query.text,
                top_k=10
            )
            
            # Get ground truth relevant documents
            relevant = query.relevant_documents
            
            # Calculate precision
            retrieved_ids = [d.id for d in retrieved]
            relevant_ids = set(relevant)
            
            true_positives = len(set(retrieved_ids) & relevant_ids)
            precision = true_positives / len(retrieved_ids) if retrieved_ids else 0
            
            # Calculate recall
            recall = true_positives / len(relevant_ids) if relevant_ids else 0
            
            # Calculate MRR (Mean Reciprocal Rank)
            first_relevant_rank = None
            for i, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_ids:
                    first_relevant_rank = i
                    break
            
            mrr = 1 / first_relevant_rank if first_relevant_rank else 0
            
            # Store metrics
            metrics.add_query_metrics(
                query_id=query.id,
                precision=precision,
                recall=recall,
                mrr=mrr
            )
        
        # Calculate aggregates
        metrics.calculate_aggregates()
        
        return metrics
    
    async def _evaluate_generation(self, dataset: TestDataset) -> GenerationMetrics:
        """
        Evaluate generation quality.
        
        Generation metrics:
        - Faithfulness: Does the answer stick to the context?
        - Relevance: Does the answer address the question?
        - Fluency: Is the answer well-written?
        - Completeness: Does the answer cover all aspects?
        """
        
        metrics = GenerationMetrics()
        
        for test_case in dataset.test_cases:
            # Generate answer
            generated_answer = await self.generation_evaluator.generate(
                question=test_case.question,
                context=test_case.context
            )
            
            # Evaluate faithfulness (no hallucination)
            faithfulness = await self._evaluate_faithfulness(
                answer=generated_answer,
                context=test_case.context
            )
            
            # Evaluate relevance
            relevance = await self._evaluate_relevance(
                answer=generated_answer,
                question=test_case.question
            )
            
            # Evaluate fluency
            fluency = self._evaluate_fluency(generated_answer)
            
            # Compare with reference answer if available
            if test_case.reference_answer:
                similarity = self._calculate_answer_similarity(
                    generated_answer,
                    test_case.reference_answer
                )
            else:
                similarity = None
            
            metrics.add_generation_metrics(
                test_id=test_case.id,
                faithfulness=faithfulness,
                relevance=relevance,
                fluency=fluency,
                similarity=similarity
            )
        
        return metrics
```

---

## **15. Troubleshooting: When Things Go Wrong**

Even the best systems have problems. The key is being able to quickly diagnose and fix issues when they arise. Think of this section as your emergency medical guide for the system.

### **15.1 Common Issues and Solutions**

Let's explore the most common problems and their solutions, organized by symptom:

```python
class TroubleshootingGuide:
    """
    Comprehensive troubleshooting system for common issues.
    
    Like a diagnostic manual for a car, this helps you identify
    problems from symptoms and apply the right fix.
    """
    
    def __init__(self):
        self.diagnostics = DiagnosticEngine()
        self.solutions = SolutionRepository()
        self.history = TroubleshootingHistory()
        
    async def diagnose_issue(self, symptoms: List[str]) -> Diagnosis:
        """
        Diagnose system issues from symptoms.
        
        The diagnostic process:
        1. Gather symptoms
        2. Check system health
        3. Analyze patterns
        4. Identify root cause
        5. Recommend solution
        """
        
        diagnosis = Diagnosis()
        
        # Map symptoms to potential issues
        potential_issues = self._map_symptoms_to_issues(symptoms)
        
        # Run diagnostic checks
        for issue in potential_issues:
            check_result = await self._run_diagnostic_check(issue)
            
            if check_result.confirmed:
                diagnosis.add_confirmed_issue(issue, check_result.evidence)
        
        # Determine root cause
        root_cause = self._determine_root_cause(diagnosis.confirmed_issues)
        diagnosis.root_cause = root_cause
        
        # Get solution
        solution = self.solutions.get_solution(root_cause)
        diagnosis.recommended_solution = solution
        
        # Record in history for pattern analysis
        await self.history.record(diagnosis)
        
        return diagnosis
    
    def _map_symptoms_to_issues(self, symptoms: List[str]) -> List[Issue]:
        """
        Map observed symptoms to potential issues.
        
        Common symptom patterns:
        - Slow responses -> Performance issues
        - Errors -> Service failures
        - Inconsistent results -> Data issues
        """
        
        symptom_map = {
            'slow_response': [
                Issue('high_database_latency', probability=0.4),
                Issue('insufficient_caching', probability=0.3),
                Issue('model_loading_delay', probability=0.2),
                Issue('network_congestion', probability=0.1)
            ],
            'authentication_failures': [
                Issue('jwt_key_mismatch', probability=0.3),
                Issue('redis_connection_lost', probability=0.3),
                Issue('database_connection_issue', probability=0.2),
                Issue('time_sync_problem', probability=0.2)
            ],
            'empty_search_results': [
                Issue('vector_index_corrupted', probability=0.3),
                Issue('embedding_service_down', probability=0.3),
                Issue('mongodb_connection_lost', probability=0.2),
                Issue('query_malformed', probability=0.2)
            ],
            'high_error_rate': [
                Issue('service_overload', probability=0.3),
                Issue('dependency_failure', probability=0.3),
                Issue('memory_exhaustion', probability=0.2),
                Issue('code_regression', probability=0.2)
            ]
        }
        
        issues = []
        for symptom in symptoms:
            if symptom in symptom_map:
                issues.extend(symptom_map[symptom])
        
        # Sort by probability
        issues.sort(key=lambda x: x.probability, reverse=True)
        
        return issues
```

### **15.2 Debugging Strategies**

When troubleshooting doesn't solve the problem, we need to dig deeper with debugging:

```python
class DebugAssistant:
    """
    Advanced debugging assistant for complex issues.
    
    This is like having a detective's toolkit - various tools
    and techniques to investigate mysterious problems.
    """
    
    async def debug_request_flow(self, request_id: str) -> DebugReport:
        """
        Debug a specific request through the system.
        
        Following a request is like following a package through
        the postal system - we track it at every step.
        """
        
        report = DebugReport(request_id=request_id)
        
        # Gather all logs for this request
        logs = await self._gather_request_logs(request_id)
        report.add_logs(logs)
        
        # Gather all traces
        traces = await self._gather_request_traces(request_id)
        report.add_traces(traces)
        
        # Analyze the flow
        flow_analysis = self._analyze_request_flow(logs, traces)
        report.add_analysis(flow_analysis)
        
        # Identify anomalies
        anomalies = self._identify_anomalies(flow_analysis)
        report.add_anomalies(anomalies)
        
        # Generate timeline
        timeline = self._create_request_timeline(logs, traces)
        report.add_timeline(timeline)
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(timeline)
        report.add_bottlenecks(bottlenecks)
        
        return report
    
    def _analyze_request_flow(
        self,
        logs: List[LogEntry],
        traces: List[Trace]
    ) -> FlowAnalysis:
        """
        Analyze the request flow to understand what happened.
        
        Analysis includes:
        - Path taken through services
        - Time spent in each service
        - Data transformations
        - Error points
        """
        
        analysis = FlowAnalysis()
        
        # Build service call graph
        service_calls = []
        
        for trace in traces:
            service_calls.append({
                'service': trace.service_name,
                'operation': trace.operation,
                'duration': trace.duration_ms,
                'status': trace.status,
                'timestamp': trace.timestamp
            })
        
        analysis.service_flow = service_calls
        
        # Calculate time distribution
        total_time = sum(call['duration'] for call in service_calls)
        
        time_distribution = {}
        for call in service_calls:
            service = call['service']
            if service not in time_distribution:
                time_distribution[service] = 0
            time_distribution[service] += call['duration']
        
        # Convert to percentages
        for service in time_distribution:
            time_distribution[service] = (time_distribution[service] / total_time) * 100
        
        analysis.time_distribution = time_distribution
        
        # Identify errors
        errors = [call for call in service_calls if call['status'] != 'success']
        analysis.errors = errors
        
        return analysis
```

---

## **16. Cost Management: Efficiency & Budgeting**

Running an AI system can be expensive. This section is about making every dollar count, like a CFO for your system.

### **16.1 Cost Tracking and Analysis**

Understanding where money is being spent is the first step to controlling costs:

```python
class CostManagementSystem:
    """
    Comprehensive cost tracking and optimization system.
    
    This is like having a detailed expense report for your system,
    showing exactly where every penny goes.
    """
    
    def __init__(self, config: CostConfig):
        self.config = config
        self.cost_tracker = CostTracker()
        self.optimizer = CostOptimizer()
        self.forecaster = CostForecaster()
        
    async def analyze_costs(self, period: DateRange) -> CostAnalysis:
        """
        Analyze costs for a given period.
        
        Cost breakdown includes:
        - Infrastructure (servers, storage, network)
        - AI models (inference costs)
        - Databases (operations, storage)
        - External APIs
        - Human operations
        """
        
        analysis = CostAnalysis(period=period)
        
        # Infrastructure costs
        infrastructure_costs = await self._calculate_infrastructure_costs(period)
        analysis.add_category('infrastructure', infrastructure_costs)
        
        # AI model costs
        model_costs = await self._calculate_model_costs(period)
        analysis.add_category('ai_models', model_costs)
        
        # Database costs
        database_costs = await self._calculate_database_costs(period)
        analysis.add_category('databases', database_costs)
        
        # External API costs
        api_costs = await self._calculate_api_costs(period)
        analysis.add_category('external_apis', api_costs)
        
        # Calculate cost per transaction
        total_transactions = await self._count_transactions(period)
        analysis.cost_per_transaction = analysis.total_cost / total_transactions
        
        # Identify cost trends
        trends = await self._analyze_cost_trends(period)
        analysis.trends = trends
        
        # Generate recommendations
        recommendations = await self.optimizer.generate_recommendations(analysis)
        analysis.recommendations = recommendations
        
        return analysis
    
    async def _calculate_model_costs(self, period: DateRange) -> ModelCosts:
        """
        Calculate AI model inference costs.
        
        Model costs include:
        - Embedding generation (tokens processed)
        - Text generation (tokens generated)
        - Reranking (documents processed)
        - Fine-tuning (if applicable)
        """
        
        costs = ModelCosts()
        
        # Get usage metrics
        embedding_tokens = await self._get_embedding_token_count(period)
        generation_tokens = await self._get_generation_token_count(period)
        rerank_calls = await self._get_rerank_call_count(period)
        
        # Calculate costs based on pricing
        costs.embedding = embedding_tokens * self.config.embedding_cost_per_1k_tokens / 1000
        costs.generation = generation_tokens * self.config.generation_cost_per_1k_tokens / 1000
        costs.reranking = rerank_calls * self.config.rerank_cost_per_call
        
        # Add compute costs if self-hosted
        if self.config.self_hosted:
            compute_hours = await self._get_compute_hours(period)
            costs.compute = compute_hours * self.config.gpu_cost_per_hour
        
        return costs
    
    async def optimize_costs(self) -> OptimizationPlan:
        """
        Generate and execute cost optimization plan.
        
        Optimization strategies:
        - Right-sizing resources
        - Improving cache hit rates
        - Batch processing
        - Off-peak scheduling
        - Model quantization
        """
        
        plan = OptimizationPlan()
        
        # Analyze current usage patterns
        usage = await self._analyze_usage_patterns()
        
        # Right-sizing recommendations
        if usage.average_cpu < 40:
            plan.add_recommendation(
                "Downsize instances",
                estimated_savings=usage.compute_cost * 0.3,
                implementation="Reduce instance sizes by one tier"
            )
        
        # Cache optimization
        cache_hit_rate = await self._get_cache_hit_rate()
        if cache_hit_rate < 0.7:
            potential_reduction = (0.8 - cache_hit_rate) * usage.inference_cost
            plan.add_recommendation(
                "Improve caching",
                estimated_savings=potential_reduction,
                implementation="Increase cache TTL and size"
            )
        
        # Batch processing
        if usage.request_pattern == 'sporadic':
            plan.add_recommendation(
                "Implement batching",
                estimated_savings=usage.inference_cost * 0.2,
                implementation="Batch embed requests with 100ms window"
            )
        
        # Off-peak scheduling
        if usage.has_predictable_pattern:
            plan.add_recommendation(
                "Schedule heavy tasks off-peak",
                estimated_savings=usage.compute_cost * 0.15,
                implementation="Move batch jobs to 2-6 AM"
            )
        
        return plan
```

### **16.2 Budget Controls and Alerts**

Preventing cost overruns requires active monitoring and controls:

```python
class BudgetController:
    """
    Budget management and enforcement system.
    
    Like a financial controller who ensures you don't overspend,
    this system monitors and enforces budget limits.
    """
    
    async def enforce_budget_limits(self):
        """
        Enforce budget limits in real-time.
        
        Budget enforcement strategies:
        - Soft limits: Warnings and alerts
        - Hard limits: Service degradation
        - Emergency stops: Complete shutdown
        """
        
        while True:
            # Check current spend
            current_spend = await self._get_current_month_spend()
            budget_remaining = self.config.monthly_budget - current_spend
            days_remaining = self._days_remaining_in_month()
            
            # Calculate burn rate
            daily_burn_rate = current_spend / (30 - days_remaining)
            projected_spend = current_spend + (daily_burn_rate * days_remaining)
            
            # Check if we're on track
            if projected_spend > self.config.monthly_budget * 1.1:
                # Projected to exceed budget by >10%
                await self._trigger_emergency_measures()
                
            elif projected_spend > self.config.monthly_budget:
                # Projected to exceed budget
                await self._trigger_cost_reduction_measures()
                
            elif projected_spend > self.config.monthly_budget * 0.9:
                # Approaching budget limit
                await self._send_budget_warning()
            
            # Check every hour
            await asyncio.sleep(3600)
    
    async def _trigger_cost_reduction_measures(self):
        """
        Implement cost reduction measures.
        
        Gradual degradation to reduce costs:
        1. Reduce cache sizes
        2. Lower model precision
        3. Decrease rate limits
        4. Disable non-essential features
        """
        
        logger.warning("Implementing cost reduction measures")
        
        # Reduce cache sizes
        await self.cache_manager.reduce_cache_size(reduction=0.5)
        
        # Switch to smaller models
        await self.model_manager.switch_to_economy_mode()
        
        # Tighten rate limits
        await self.rate_limiter.reduce_limits(factor=0.7)
        
        # Disable non-essential features
        await self.feature_flags.disable(['analytics', 'advanced_search'])
        
        # Notify stakeholders
        await self.send_notification(
            level='WARNING',
            message="Cost reduction measures activated due to budget constraints"
        )
```

---

## **17. Roadmap & Evolution: What's Next**

The system is never truly "done" - it continues to evolve. This section outlines the future direction and how to prepare for it.

### **17.1 Future Enhancements**

Let's explore what's coming next and how to prepare for future capabilities:

```python
class SystemEvolutionRoadmap:
    """
    Roadmap for system evolution and enhancement.
    
    Like a city planner designing for future growth,
    we need to anticipate and prepare for what's coming.
    """
    
    def __init__(self):
        self.roadmap_items = []
        self.dependencies = {}
        self.timeline = Timeline()
        
    def define_roadmap(self) -> Roadmap:
        """
        Define the system evolution roadmap.
        
        Roadmap categories:
        - Near-term (0-3 months): Quick wins and fixes
        - Mid-term (3-9 months): Major features
        - Long-term (9-18 months): Architectural changes
        - Vision (18+ months): Revolutionary capabilities
        """
        
        roadmap = Roadmap()
        
        # Near-term enhancements
        roadmap.add_phase(
            "Near-term",
            timeline="0-3 months",
            items=[
                RoadmapItem(
                    title="Adaptive Retrieval",
                    description="""
                    Implement dynamic retrieval depth based on query complexity.
                    Simple queries get 3 documents, complex ones get 10+.
                    This optimizes both cost and quality.
                    """,
                    impact="20% cost reduction, 10% quality improvement",
                    effort="2 weeks",
                    dependencies=[]
                ),
                RoadmapItem(
                    title="Multi-language Support",
                    description="""
                    Add support for 10 major languages with language-specific
                    embeddings and generation models.
                    """,
                    impact="10x addressable market",
                    effort="1 month",
                    dependencies=["translation_api", "multilingual_models"]
                ),
                RoadmapItem(
                    title="Smart Caching 2.0",
                    description="""
                    Implement semantic caching that identifies similar queries
                    even when worded differently.
                    """,
                    impact="30% reduction in model calls",
                    effort="3 weeks",
                    dependencies=["similarity_index"]
                )
            ]
        )
        
        # Mid-term enhancements
        roadmap.add_phase(
            "Mid-term",
            timeline="3-9 months",
            items=[
                RoadmapItem(
                    title="Multimodal RAG",
                    description="""
                    Support images, PDFs, and videos in addition to text.
                    Process diagrams, charts, and visual information.
                    """,
                    impact="3x more use cases supported",
                    effort="3 months",
                    dependencies=["vision_models", "pdf_parser", "video_processor"]
                ),
                RoadmapItem(
                    title="Active Learning Pipeline",
                    description="""
                    Learn from user feedback to continuously improve.
                    Automatically identify poor responses and retrain.
                    """,
                    impact="Continuous quality improvement",
                    effort="2 months",
                    dependencies=["feedback_system", "training_pipeline"]
                ),
                RoadmapItem(
                    title="Federation Support",
                    description="""
                    Allow multiple organizations to share knowledge
                    while maintaining data privacy.
                    """,
                    impact="Enterprise market penetration",
                    effort="3 months",
                    dependencies=["privacy_preserving_ml", "federation_protocol"]
                )
            ]
        )
        
        # Long-term vision
        roadmap.add_phase(
            "Long-term Vision",
            timeline="9-18 months",
            items=[
                RoadmapItem(
                    title="Autonomous Agents",
                    description="""
                    Deploy agents that can perform multi-step tasks
                    autonomously, like research assistants.
                    """,
                    impact="10x productivity improvement",
                    effort="6 months",
                    dependencies=["agent_framework", "tool_integration"]
                ),
                RoadmapItem(
                    title="Real-time Knowledge Updates",
                    description="""
                    Continuously update knowledge base from streams,
                    websites, and APIs in real-time.
                    """,
                    impact="Always current information",
                    effort="4 months",
                    dependencies=["streaming_pipeline", "incremental_indexing"]
                ),
                RoadmapItem(
                    title="Explainable AI",
                    description="""
                    Provide detailed explanations for every decision,
                    making the system fully transparent.
                    """,
                    impact="Regulatory compliance, user trust",
                    effort="5 months",
                    dependencies=["interpretability_framework"]
                )
            ]
        )
        
        return roadmap
```

### **17.2 Preparing for Scale**

As the system grows, we need to prepare for massive scale:

```python
class ScalePreparationStrategy:
    """
    Strategy for preparing the system for massive scale.
    
    Like preparing a small town to become a major city,
    we need infrastructure that can grow.
    """
    
    def prepare_for_scale(self) -> ScaleStrategy:
        """
        Prepare system for 100x growth.
        
        Scale preparation areas:
        - Data architecture
        - Service architecture  
        - Operational capability
        - Cost efficiency
        """
        
        strategy = ScaleStrategy()
        
        # Data architecture evolution
        strategy.add_component(
            "Data Lake Formation",
            description="""
            Transition from databases to a unified data lake architecture.
            This provides unlimited storage and processing capability.
            """,
            implementation_steps=[
                "Implement data ingestion pipelines",
                "Set up data cataloging",
                "Implement data governance",
                "Build analytical capabilities"
            ]
        )
        
        # Service mesh adoption
        strategy.add_component(
            "Service Mesh",
            description="""
            Implement service mesh for better service-to-service communication,
            observability, and security at scale.
            """,
            implementation_steps=[
                "Deploy Istio or Linkerd",
                "Implement mutual TLS",
                "Set up distributed tracing",
                "Implement circuit breakers"
            ]
        )
        
        # Global distribution
        strategy.add_component(
            "Global Distribution",
            description="""
            Deploy across multiple regions for global low-latency access
            and disaster recovery.
            """,
            implementation_steps=[
                "Set up multi-region clusters",
                "Implement geo-routing",
                "Set up cross-region replication",
                "Implement conflict resolution"
            ]
        )
        
        return strategy
```

---

### **Key Principles to Remember**

The success of this system rests on several fundamental principles that should guide every decision:

**1. Reliability Over Features**: It's better to have a simple system that works reliably than a complex one that fails. Every feature we add increases complexity, so we must ensure each one truly adds value.

**2. Observable by Design**: You can't fix what you can't see. Every component emits metrics, logs, and traces. When something goes wrong, we have the data to understand why.

**3. Graceful Degradation**: When components fail, the system continues with reduced functionality rather than complete failure. Users might get slightly worse service, but they get service.

**4. Security in Depth**: Security isn't a single wall but multiple layers of defense. Even if one layer is breached, others remain to protect the system.

**5. Cost-Conscious Architecture**: Every architectural decision has cost implications. We optimize not just for performance but for cost-effectiveness.



---
