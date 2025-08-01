# app/utils/seed_data.py
"""
Enhanced data seeding utilities for the Redis-ScyllaDB hybrid knowledge base.
Provides functions to populate the knowledge base with comprehensive FAQ data.
"""

import logging
from typing import List, Dict, Any

from app.database.scylla_models import KnowledgeBase, KnowledgeEntry
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

def seed_knowledge_base(knowledge_base: KnowledgeBase) -> bool:
    """
    Populate the PROVIDED knowledge base with comprehensive FAQ data.

    Args:
        knowledge_base: The KnowledgeBase instance to populate

    Returns:
        bool: True if seeding was successful
    """

    # Enhanced FAQ data with more categories
    sample_data = [
        # Technology & Databases
        {
            "keyword": "redis",
            "question": "What is Redis and how is it used?",
            "answer": "Redis is an in-memory data structure store used as a database, cache, and message broker. It supports data structures like strings, hashes, lists, sets, and sorted sets with atomic operations. Redis is perfect for caching, real-time analytics, and session management."
        },
        {
            "keyword": "python",
            "question": "What is Python programming language?",
            "answer": "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data science, automation, artificial intelligence, and machine learning applications."
        },
        {
            "keyword": "database",
            "question": "What types of databases are there?",
            "answer": "There are several types of databases: Relational databases (SQL) like PostgreSQL and MySQL store structured data in tables. NoSQL databases like MongoDB handle unstructured data. In-memory databases like Redis provide ultra-fast access. Graph databases like Neo4j handle relationships."
        },
        {
            "keyword": "scylladb",
            "question": "What is ScyllaDB?",
            "answer": "ScyllaDB is a high-performance NoSQL database compatible with Apache Cassandra. It's written in C++ and offers dramatically better performance than traditional databases, making it ideal for applications requiring low latency and high throughput."
        },
        {
            "keyword": "programming",
            "question": "What is programming and how do I start?",
            "answer": "Programming is the process of creating instructions for computers using programming languages. To start: 1) Choose a beginner-friendly language like Python, 2) Learn basic concepts like variables and loops, 3) Practice with small projects, 4) Use online resources like tutorials and coding platforms."
        },

        # Chatbot & AI
        {
            "keyword": "chatbot",
            "question": "How do chatbots work?",
            "answer": "Chatbots use natural language processing (NLP) to understand user input and provide relevant responses. Modern chatbots can use rule-based systems, machine learning, or AI models like GPT to generate human-like conversations and help users with questions and tasks."
        },
        {
            "keyword": "artificial",
            "question": "What is artificial intelligence?",
            "answer": "Artificial Intelligence (AI) is the simulation of human intelligence in machines. AI systems can learn, reason, and make decisions. Common applications include chatbots, recommendation systems, image recognition, and autonomous vehicles."
        },
        {
            "keyword": "machine",
            "question": "What is machine learning?",
            "answer": "Machine Learning (ML) is a subset of AI where computers learn patterns from data without being explicitly programmed. ML algorithms can make predictions, classify data, and improve performance over time. Common types include supervised, unsupervised, and reinforcement learning."
        },

        # Web Development
        {
            "keyword": "web",
            "question": "How do websites work?",
            "answer": "Websites work through a client-server model. When you visit a website, your browser (client) sends a request to a web server. The server processes the request and sends back HTML, CSS, and JavaScript files that your browser renders as a webpage."
        },
        {
            "keyword": "api",
            "question": "What is an API?",
            "answer": "API stands for Application Programming Interface. It's a set of rules and protocols that allows different software applications to communicate with each other. APIs enable developers to access functionality or data from other services without knowing their internal implementation."
        },
        {
            "keyword": "html",
            "question": "What is HTML?",
            "answer": "HTML (HyperText Markup Language) is the standard markup language for creating web pages. It uses tags to structure content like headings, paragraphs, links, and images. HTML provides the basic structure of websites, which is then styled with CSS and made interactive with JavaScript."
        },

        # General Technology
        {
            "keyword": "cloud",
            "question": "What is cloud computing?",
            "answer": "Cloud computing delivers computing services (servers, storage, databases, networking) over the internet. Instead of owning physical hardware, you can access these resources on-demand from providers like AWS, Google Cloud, or Azure. Benefits include scalability, cost-efficiency, and accessibility."
        },
        {
            "keyword": "security",
            "question": "What is cybersecurity?",
            "answer": "Cybersecurity protects digital systems, networks, and data from digital attacks. It includes practices like using strong passwords, enabling two-factor authentication, keeping software updated, and being cautious with emails and downloads. Good security is essential for protecting personal and business information."
        },
        {
            "keyword": "backup",
            "question": "Why are data backups important?",
            "answer": "Data backups are copies of your important files stored separately from the original. They protect against data loss from hardware failure, cyber attacks, accidental deletion, or natural disasters. Follow the 3-2-1 rule: 3 copies of data, 2 different storage types, 1 offsite backup."
        },

        # Learning & Career
        {
            "keyword": "learning",
            "question": "How can I learn programming effectively?",
            "answer": "Effective programming learning includes: 1) Start with fundamentals and practice regularly, 2) Build real projects to apply knowledge, 3) Read others' code and contribute to open source, 4) Join coding communities and forums, 5) Don't be afraid to make mistakes - debugging teaches valuable skills."
        },
        {
            "keyword": "career",
            "question": "What career paths are available in technology?",
            "answer": "Technology offers diverse career paths: Software Developer (web, mobile, backend), Data Scientist/Analyst, DevOps Engineer, Cybersecurity Specialist, Product Manager, UX/UI Designer, Database Administrator, System Administrator, and emerging fields like AI/ML Engineer and Cloud Architect."
        },

        # Troubleshooting
        {
            "keyword": "troubleshooting",
            "question": "How do I troubleshoot technical problems?",
            "answer": "Effective troubleshooting steps: 1) Clearly identify the problem and when it started, 2) Check recent changes or updates, 3) Try basic fixes like restarting, 4) Search for error messages online, 5) Test with minimal configuration, 6) Document what you tried, 7) Ask for help with specific details."
        },
        {
            "keyword": "performance",
            "question": "How can I improve system performance?",
            "answer": "To improve system performance: 1) Close unnecessary programs and browser tabs, 2) Restart your system regularly, 3) Keep software updated, 4) Use antivirus software, 5) Clean up disk space, 6) Add more RAM if needed, 7) Consider SSD upgrade, 8) Monitor resource usage to identify bottlenecks."
        }
    ]

    success_count = 0
    total_count = len(sample_data)

    logger.info(f"Starting to seed knowledge base with {total_count} entries...")

    # Clear existing mock data and add new comprehensive data
    knowledge_base._mock_knowledge = []

    for entry in sample_data:
        try:
            knowledge_entry = KnowledgeEntry(
                keyword=entry["keyword"],
                question_id=uuid.uuid4(),
                question=entry["question"],
                answer=entry["answer"],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            knowledge_base._mock_knowledge.append(knowledge_entry)
            success_count += 1
            logger.debug(f"Added entry for keyword '{entry['keyword']}'")

        except Exception as e:
            logger.error(f"Error adding entry for keyword '{entry['keyword']}': {e}")

    logger.info(f"Seeding completed: {success_count}/{total_count} entries added successfully")
    logger.info(f"Knowledge base now contains {len(knowledge_base._mock_knowledge)} total entries")
    return success_count == total_count

def get_sample_questions() -> List[str]:
    """Get sample questions for testing the enhanced chatbot."""
    return [
        "What is Redis?",
        "How do I learn programming?",
        "What is artificial intelligence?",
        "How do websites work?",
        "What is machine learning?",
        "How do chatbots work?",
        "What is cloud computing?",
        "What career paths are available in technology?",
        "How can I improve system performance?",
        "What is cybersecurity?"
    ]