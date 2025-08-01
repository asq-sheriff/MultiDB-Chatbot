# app/database/scylla_models.py
"""
ScyllaDB models for the chatbot application.
These models handle persistent storage in ScyllaDB while Redis handles caching.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ConversationMessage:
    """Individual conversation message"""
    session_id: uuid.UUID
    actor: str  # 'user' or 'bot'
    message: str
    timestamp: datetime
    message_id: Optional[uuid.UUID] = None


@dataclass
class UserFeedback:
    """User feedback entry"""
    feedback_id: uuid.UUID
    user_id: str
    session_id: uuid.UUID
    feedback_message: str
    timestamp: datetime


class ConversationHistory:
    """
    Handles conversation history storage in ScyllaDB.
    This is a placeholder implementation for the Redis project.
    """

    def __init__(self):
        # In a real implementation, this would connect to ScyllaDB
        self._mock_storage = {}

    def save_message(self, session_id: uuid.UUID, actor: str, message: str) -> None:
        """Save a message to conversation history"""
        if str(session_id) not in self._mock_storage:
            self._mock_storage[str(session_id)] = []

        message_obj = ConversationMessage(
            session_id=session_id,
            actor=actor,
            message=message,
            timestamp=datetime.now(timezone.utc),
            message_id=uuid.uuid4()
        )

        self._mock_storage[str(session_id)].append(message_obj)

    def get_session_history(self, session_id: uuid.UUID, limit: int = 50) -> List[ConversationMessage]:
        """Get conversation history for a session"""
        messages = self._mock_storage.get(str(session_id), [])
        return messages[-limit:] if len(messages) > limit else messages


class UserFeedbackRepository:
    """
    Handles user feedback storage in ScyllaDB.
    This is a placeholder implementation for the Redis project.
    """

    def __init__(self):
        # In a real implementation, this would connect to ScyllaDB
        self._mock_feedback = {}

    def save_feedback(self, user_id: str, session_id: uuid.UUID, feedback_message: str) -> uuid.UUID:
        """Save user feedback"""
        feedback_id = uuid.uuid4()

        if user_id not in self._mock_feedback:
            self._mock_feedback[user_id] = []

        feedback = UserFeedback(
            feedback_id=feedback_id,
            user_id=user_id,
            session_id=session_id,
            feedback_message=feedback_message,
            timestamp=datetime.now(timezone.utc)
        )

        self._mock_feedback[user_id].append(feedback)
        return feedback_id

    def get_user_feedback(self, user_id: str, limit: int = 10) -> List[UserFeedback]:
        """Get user's feedback history"""
        feedback_list = self._mock_feedback.get(user_id, [])
        return feedback_list[-limit:] if len(feedback_list) > limit else feedback_list


@dataclass
class KnowledgeEntry:
    """Knowledge base entry"""
    keyword: str
    question_id: uuid.UUID
    question: str
    answer: str
    created_at: datetime
    updated_at: datetime


class KnowledgeBase:
    """
    Handles knowledge base storage in ScyllaDB.
    This is a placeholder implementation for the Redis project.
    """

    def __init__(self):
        # Mock knowledge base with some sample data
        self._mock_knowledge = [
            KnowledgeEntry(
                keyword="hello",
                question_id=uuid.uuid4(),
                question="How do I greet someone?",
                answer="You can say hello, hi, or good morning/afternoon/evening.",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            ),
            KnowledgeEntry(
                keyword="python",
                question_id=uuid.uuid4(),
                question="What is Python?",
                answer="Python is a high-level programming language known for its simplicity and readability.",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            ),
            KnowledgeEntry(
                keyword="redis",
                question_id=uuid.uuid4(),
                question="What is Redis?",
                answer="Redis is an in-memory data structure store used for caching, sessions, and real-time analytics.",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
        ]

    def search_keywords(self, keywords: List[str]) -> List[KnowledgeEntry]:
        """Search knowledge base by keywords - FIXED version"""
        results = []

        for entry in self._mock_knowledge:
            # Check if ANY keyword matches entry keyword, question, or answer
            for keyword in keywords:
                keyword_lower = keyword.lower()

                # Check direct keyword match
                if keyword_lower == entry.keyword.lower():
                    results.append(entry)
                    break

                # Check if keyword appears in question or answer
                elif (keyword_lower in entry.question.lower() or
                      keyword_lower in entry.answer.lower()):
                    results.append(entry)
                    break

        # Remove duplicates while preserving order
        seen = set()
        unique_results = []
        for entry in results:
            if entry.keyword not in seen:
                seen.add(entry.keyword)
                unique_results.append(entry)

        print(f"DEBUG: Searched for {keywords}, found {len(unique_results)} entries")  # Temporary debug
        return unique_results