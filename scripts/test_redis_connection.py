# test_redis_connection.py - NEW FILE
"""
Quick test script to verify Redis and chatbot integration.
Run this to make sure everything is working before running main.py
"""

import asyncio
import logging
from app.config import config
from app.database.redis_connection import redis_manager
from app.services.chatbot_service import ChatbotService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_redis_integration():
    """Test Redis connection and basic chatbot functionality"""
    try:
        # Test Redis connection
        redis_manager.initialize()
        logger.info("✅ Redis connection successful")

        # Test chatbot service
        chatbot = ChatbotService()
        logger.info("✅ Chatbot service initialized")

        # Test session creation
        session_id = chatbot.create_session({"test": "user"})
        logger.info(f"✅ Session created: {str(session_id)[:8]}...")

        # Test message processing
        response = chatbot.process_message(session_id, "hello")
        logger.info(f"✅ Message processed: {response.message[:50]}...")
        logger.info(f"   Response time: {response.response_time_ms}ms")
        logger.info(f"   Cached: {response.cached}")

        # Test stats
        stats_response = chatbot.process_message(session_id, "/stats")
        logger.info("✅ Stats command working")

        # Test cache
        response2 = chatbot.process_message(session_id, "hello")
        logger.info(f"✅ Second message (should be faster): {response2.response_time_ms}ms")

        logger.info("🎉 All tests passed! Your Redis integration is working.")
        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    finally:
        redis_manager.close()

if __name__ == "__main__":
    asyncio.run(test_redis_integration())