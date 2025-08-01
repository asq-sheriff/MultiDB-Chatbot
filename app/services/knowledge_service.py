# app/services/knowledge_service.py - REPLACE entire file
"""
Knowledge service with Redis caching for improved performance.
Provides keyword-based matching and intelligent answer selection.
"""

import re
import logging
import json
from typing import List, Optional
from dataclasses import dataclass

from app.database.scylla_models import KnowledgeBase, KnowledgeEntry
from app.database.redis_connection import get_redis

logger = logging.getLogger(__name__)

# Get the Redis client from your connection manager
try:
    redis_client = get_redis()
    logger.info("KnowledgeService successfully connected to Redis.")
except Exception as e:
    logger.error(f"KnowledgeService failed to connect to Redis: {e}. Caching will be disabled.")
    redis_client = None

# Constants
CACHE_EXPIRATION_SECONDS = 300  # 5 minutes


@dataclass
class MatchResult:
    """
    Result of a knowledge search operation.
    Used by: services/enhanced_chatbot_service.py to handle search results
    """
    entry: KnowledgeEntry
    confidence: float
    matched_keywords: List[str]


class KnowledgeService:
    """
    Service for managing and searching the knowledge base with Redis caching.
    Provides keyword-based matching and intelligent answer selection.
    """

    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self._stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'how', 'when', 'where', 'why', 'who', 'which'
        }


    def seed_knowledge_base(self):
        """Seed the knowledge base with comprehensive data using THIS instance."""
        from app.utils.seed_data import seed_knowledge_base
        # Pass THIS knowledge base instance to the seeding function
        return seed_knowledge_base(self.knowledge_base)


    def get_best_answer(self, user_message: str) -> Optional[MatchResult]:
        """Get the best single answer for a user message, with caching."""
        print(f"🔍 DEBUG: get_best_answer called with: '{user_message}'")

        keywords = self.extract_keywords(user_message)
        print(f"🔍 DEBUG: Extracted keywords: {keywords}")

        if not keywords:
            print("🔍 DEBUG: No keywords extracted, returning None")
            return None

        # Check cache first
        if redis_client:
            cache_key = f"cache:best_answer:{':'.join(sorted(keywords))}"
            cached_data = redis_client.get(cache_key)
            if cached_data:
                print(f"✅ DEBUG: Cache Hit for key: '{cache_key}'")
                # ... rest of cache logic stays the same
            else:
                print(f"❌ DEBUG: Cache Miss for key: '{cache_key}'")

        print(f"🔍 DEBUG: Calling search_knowledge...")
        results = self.search_knowledge(user_message)
        print(f"🔍 DEBUG: search_knowledge returned {len(results)} results")

        if not results:
            print("🔍 DEBUG: No results from search_knowledge, returning None")
            return None

        best_result = results[0]
        confidence_threshold = 0.3
        print(f"🔍 DEBUG: Best result confidence: {best_result.confidence:.2f}, threshold: {confidence_threshold}")

        if best_result.confidence >= confidence_threshold:
            print(f"✅ DEBUG: Confidence above threshold, returning result")
            # ... rest of caching logic stays the same
            return best_result
        else:
            print(f"❌ DEBUG: Confidence {best_result.confidence:.2f} below threshold {confidence_threshold}")
            return None

    # app/services/knowledge_service.py - ADD debug to search_knowledge method

    def search_knowledge(self, user_message: str) -> List[MatchResult]:
        """Search the knowledge base for relevant answers."""
        print(f"🔍 DEBUG: search_knowledge called with: '{user_message}'")

        keywords = self.extract_keywords(user_message)
        print(f"🔍 DEBUG: search_knowledge extracted keywords: {keywords}")

        if not keywords:
            print("🔍 DEBUG: search_knowledge - No keywords extracted")
            return []

        # Search for matching entries
        print(f"🔍 DEBUG: Calling knowledge_base.search_keywords with: {keywords}")
        entries = self.knowledge_base.search_keywords(keywords)
        print(f"🔍 DEBUG: knowledge_base.search_keywords returned {len(entries)} entries")

        if not entries:
            print(f"🔍 DEBUG: No knowledge entries found for keywords: {keywords}")
            return []

        # Calculate match confidence for each entry
        results = []
        for i, entry in enumerate(entries):
            print(f"🔍 DEBUG: Processing entry {i+1}: keyword='{entry.keyword}', question='{entry.question[:50]}...'")
            confidence, matched_words = self._calculate_confidence(entry, keywords)
            print(f"🔍 DEBUG: Entry {i+1} confidence: {confidence:.2f}, matched_words: {matched_words}")

            if confidence > 0:
                results.append(MatchResult(
                    entry=entry,
                    confidence=confidence,
                    matched_keywords=matched_words
                ))

        # Sort by confidence (highest first)
        results.sort(key=lambda x: x.confidence, reverse=True)
        print(f"🔍 DEBUG: Final results count: {len(results)}")

        return results



    def extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from user input."""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text.lower())

        # Split into words
        words = text.split()

        # Filter out stop words and short words
        keywords = [
            word for word in words
            if len(word) > 2 and word not in self._stop_words
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)

        logger.debug(f"Extracted keywords: {unique_keywords}")
        return unique_keywords

    def _calculate_confidence(self, entry: KnowledgeEntry,
                              keywords: List[str]) -> tuple[float, List[str]]:
        """Calculate confidence score for a knowledge entry match."""
        confidence = 0.0
        matched_keywords = []

        # Check direct keyword match
        if entry.keyword in keywords:
            confidence += 0.5
            matched_keywords.append(entry.keyword)

        # Check if any keywords appear in the question or answer
        entry_text = f"{entry.question} {entry.answer}".lower()

        for keyword in keywords:
            if keyword in entry_text:
                confidence += 0.2
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)

        # Bonus for question similarity
        question_words = set(self.extract_keywords(entry.question))
        user_words = set(keywords)

        if question_words and user_words:
            overlap = len(question_words.intersection(user_words))
            total_unique = len(question_words.union(user_words))

            if total_unique > 0:
                similarity_bonus = (overlap / total_unique) * 0.3
                confidence += similarity_bonus

        # Normalize confidence to 0-1 range
        confidence = min(confidence, 1.0)

        return confidence, matched_keywords

    # app/services/knowledge_service.py - REPLACE the get_available_keywords method

    def get_available_keywords(self) -> List[str]:
        """Get list of available keywords from knowledge base."""
        try:
            # Get actual keywords from the knowledge base entries
            keywords = list(set([entry.keyword for entry in self.knowledge_base._mock_knowledge]))
            return sorted(keywords)
        except Exception as e:
            logger.error(f"Failed to get available keywords: {e}")
            return []

    def suggest_keywords(self, keyword: str) -> List[str]:
        """Suggest similar keywords."""
        # Simple implementation - in real case would use more sophisticated matching
        suggestions = []
        available = self.get_available_keywords()

        for available_keyword in available:
            if keyword.lower() in available_keyword.lower() or available_keyword.lower() in keyword.lower():
                suggestions.append(available_keyword)

        return suggestions[:3]  # Return top 3 suggestions