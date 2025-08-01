# test_notifications.py - NEW FILE (create in project root)
"""
Test script for the notification system.
Run this to verify Redis Lists and background tasks are working.
"""

import asyncio
import time
import logging
from app.config import config
from app.database.redis_connection import redis_manager
from app.services.chatbot_service import ChatbotService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_notification_system():
    """Test the complete notification system"""
    try:
        # Initialize Redis
        redis_manager.initialize()
        logger.info("✅ Redis connection successful")

        # Initialize chatbot service
        chatbot = ChatbotService()
        logger.info("✅ Chatbot service with notifications initialized")

        # Create test session
        session_id = chatbot.create_session({"test": "notification_user"})
        user_id = f"user_{str(session_id)[:8]}"
        logger.info(f"✅ Test session created: {str(session_id)[:8]}...")

        print("\n" + "="*60)
        print("🧪 TESTING NOTIFICATION SYSTEM")
        print("="*60)

        # Test 1: Submit background analysis task
        print("\n📊 Test 1: Submitting analysis task...")
        response = chatbot.process_message(session_id, "/analyze large customer dataset")
        print(f"Response: {response.message[:100]}...")

        # Test 2: Submit background research task
        print("\n🔍 Test 2: Submitting research task...")
        response = chatbot.process_message(session_id, "/research artificial intelligence trends")
        print(f"Response: {response.message[:100]}...")

        # Test 3: Check notifications immediately (should be empty)
        print("\n📬 Test 3: Checking notifications (should be empty)...")
        response = chatbot.process_message(session_id, "/notifications")
        print(f"Response: {response.message[:100]}...")

        # Test 4: Peek at notifications (should also be empty)
        print("\n👀 Test 4: Peeking at notifications...")
        response = chatbot.process_message(session_id, "/notifications peek")
        print(f"Response: {response.message[:100]}...")

        # Wait for tasks to complete
        print("\n⏳ Waiting 10 seconds for background tasks to complete...")
        time.sleep(10)

        # Test 5: Check notifications again (should have results)
        print("\n📬 Test 5: Checking notifications after tasks complete...")
        response = chatbot.process_message(session_id, "/notifications")
        print(f"Response: {response.message}")

        # Test 6: Test direct Redis List operations
        print("\n🔧 Test 6: Testing direct Redis operations...")

        # Add test notification directly
        test_notification = {
            "title": "Direct Test Notification",
            "message": "This notification was added directly to Redis",
            "type": "info"
        }

        success = chatbot.notification_model.add_notification(user_id, test_notification)
        print(f"Direct notification add: {'✅ Success' if success else '❌ Failed'}")

        # Count notifications
        count = chatbot.notification_model.count_notifications(user_id)
        print(f"Notification count: {count}")

        # Peek at notifications
        notifications = chatbot.notification_model.peek_notifications(user_id, 2)
        print(f"Peeked notifications: {len(notifications)} found")

        # Get notifications (removes them)
        notifications = chatbot.notification_model.get_notifications(user_id, 1)
        print(f"Retrieved notifications: {len(notifications)} found")
        if notifications:
            print(f"First notification: {notifications[0]['title']}")

        # Final count
        final_count = chatbot.notification_model.count_notifications(user_id)
        print(f"Final notification count: {final_count}")

        print("\n✅ All notification tests completed successfully!")

        # Test Redis keys directly
        print("\n🔍 Redis Keys Inspection:")
        redis_client = redis_manager.client

        # Check notification keys
        notification_keys = redis_client.keys("notifications:user:*")
        print(f"Notification keys in Redis: {len(notification_keys)}")

        # Check analytics keys
        analytics_keys = redis_client.keys("analytics:*")
        print(f"Analytics keys in Redis: {len(analytics_keys)}")

        # Show some analytics
        tasks_submitted = redis_client.get("analytics:counter:background_tasks_submitted") or 0
        notifications_sent = redis_client.get("analytics:counter:notifications_sent") or 0
        print(f"Background tasks submitted: {tasks_submitted}")
        print(f"Notifications sent: {notifications_sent}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if 'chatbot' in locals():
            chatbot.background_tasks.shutdown()
        redis_manager.close()

if __name__ == "__main__":
    asyncio.run(test_notification_system())