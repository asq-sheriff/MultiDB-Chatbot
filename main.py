# main.py

import signal
import sys
import asyncio
import logging
import uuid
import os
from datetime import datetime, timezone
from typing import Optional

# Set compatibility environment variables
os.environ['SQLALCHEMY_WARN_20'] = '0'
os.environ['SQLALCHEMY_SILENCE_UBER_WARNING'] = '1'

from app.config import config
from app.database.redis_connection import redis_manager
from app.database.postgres_connection import postgres_manager
from app.services.chatbot_service import ChatbotService

# Import existing services (no changes needed)
from app.services.auth_service import auth_service

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class EnhancedChatbotApplication:
    """
    Enhanced chatbot application leveraging existing API infrastructure.
    MINIMAL changes - just adds CLI authentication to existing services.
    """

    def __init__(self):
        self.chatbot_service: Optional[ChatbotService] = None
        self.current_session: Optional[uuid.UUID] = None
        self.current_user = None
        self.current_user_token: Optional[str] = None

    async def initialize(self):
        """Initialize with existing infrastructure - NO setup script needed"""
        try:
            # Use existing initialization logic but make PostgreSQL mandatory

            # 1. Initialize Redis (existing code)
            redis_manager.initialize()
            logger.info("Redis connection initialized")

            # 2. Initialize PostgreSQL (make mandatory instead of optional)
            if not config.enable_postgresql:
                logger.error("PostgreSQL is now REQUIRED for unified architecture")
                return False

            try:
                await postgres_manager.initialize()
                logger.info("PostgreSQL connection initialized")
            except Exception as e:
                logger.error(f"PostgreSQL initialization failed: {e}")
                logger.error("PostgreSQL is now REQUIRED. Please ensure it's running.")
                return False

            # 3. Initialize chatbot service (existing code)
            self.chatbot_service = ChatbotService()
            logger.info("Enhanced chatbot service initialized")

            # 4. Seed knowledge base (existing code)
            logger.info("Seeding knowledge base...")
            if self.chatbot_service.knowledge_service.seed_knowledge_base():
                logger.info("✅ Knowledge base seeded successfully!")
            else:
                logger.warning("⚠️ Knowledge base seeding had some issues")

            return True
        except Exception as e:
            logger.error(f"Failed to initialize application: {e}")
            return False

    async def start_authenticated_session(self) -> bool:
        """Add authentication flow using existing auth_service"""
        print("\n" + "="*70)
        print("🔐 ENHANCED CHATBOT - Authentication Options")
        print("="*70)
        print("1. Login with existing account")
        print("2. Create new account")
        print("3. Continue as guest (limited features)")

        while True:
            try:
                choice = input("\nSelect option (1-3): ").strip()

                if choice == "1":
                    return await self._handle_login()
                elif choice == "2":
                    return await self._handle_registration()
                elif choice == "3":
                    return await self._handle_guest_mode()
                else:
                    print("❌ Please enter 1, 2, or 3")

            except KeyboardInterrupt:
                print("\n🛑 Authentication cancelled")
                return False

    async def _handle_login(self) -> bool:
        """Use existing auth_service for login"""
        try:
            print("\n📧 User Login")
            email = input("Email: ").strip()
            if not email:
                print("❌ Email is required")
                return False

            import getpass
            password = getpass.getpass("Password: ")
            if not password:
                print("❌ Password is required")
                return False

            # Use existing auth_service
            user = await auth_service.authenticate_user(email, password)

            if user:
                # Use existing JWT creation
                token_data = {"user_id": str(user.id), "email": user.email}
                self.current_user_token = auth_service.create_access_token(token_data)
                self.current_user = user

                print(f"\n✅ Login successful! Welcome back, {user.email}")
                print(f"📊 Subscription: {user.subscription_plan.title()}")

                # Create session with user data
                user_data = {
                    "user_id": str(user.id),
                    "email": user.email,
                    "subscription_plan": user.subscription_plan,
                    "authentication": "verified"
                }
                self.current_session = self.chatbot_service.create_session(user_data)
                return True
            else:
                print("❌ Invalid email or password")
                return False

        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False

    async def _handle_registration(self) -> bool:
        """Use existing auth_service for registration"""
        try:
            print("\n👤 Create New Account")
            email = input("Email: ").strip()
            if not email or "@" not in email:
                print("❌ Please enter a valid email address")
                return False

            import getpass
            password = getpass.getpass("Password: ")
            confirm_password = getpass.getpass("Confirm Password: ")

            if not password or len(password) < 6:
                print("❌ Password must be at least 6 characters")
                return False

            if password != confirm_password:
                print("❌ Passwords do not match")
                return False

            print("\n📊 Choose subscription plan:")
            print("1. Free (1000 messages/month, 10 background tasks)")
            print("2. Pro (10000 messages/month, 100 background tasks)")
            print("3. Enterprise (unlimited)")

            plan_choice = input("Select plan (1-3, default=1): ").strip() or "1"
            plan_map = {"1": "free", "2": "pro", "3": "enterprise"}
            subscription_plan = plan_map.get(plan_choice, "free")

            # Use existing auth_service
            user = await auth_service.create_user(
                email=email,
                password=password,
                subscription_plan=subscription_plan,
                is_active=True,
                is_verified=True
            )

            # Use existing JWT creation
            token_data = {"user_id": str(user.id), "email": user.email}
            self.current_user_token = auth_service.create_access_token(token_data)
            self.current_user = user

            print(f"\n✅ Account created successfully!")
            print(f"📧 Email: {user.email}")
            print(f"📊 Plan: {user.subscription_plan.title()}")

            # Create session
            user_data = {
                "user_id": str(user.id),
                "email": user.email,
                "subscription_plan": user.subscription_plan,
                "authentication": "verified"
            }
            self.current_session = self.chatbot_service.create_session(user_data)
            return True

        except ValueError as e:
            if "already exists" in str(e):
                print("❌ An account with this email already exists")
            else:
                print(f"❌ Registration failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False

    async def _handle_guest_mode(self) -> bool:
        """Handle guest mode with existing session creation"""
        print("\n👤 Continuing as Guest")
        print("📊 Limited features: Basic Q&A, no background tasks, no quota management")

        user_data = {"user_type": "guest", "authentication": "none"}
        self.current_session = self.chatbot_service.create_session(user_data)
        return True

    def process_user_input(self, user_input: str) -> str:
        """Process with authentication-aware routing"""
        if not self.current_session:
            return "❌ No active session."

        message_lower = user_input.lower().strip()

        # Handle authenticated commands BEFORE passing to chatbot_service
        if self.current_user and self.current_user_token:
            if message_lower == '/dashboard':
                return self._get_authenticated_dashboard()
            elif message_lower == '/profile':
                return self._get_authenticated_profile()

        # For everything else, use existing chatbot_service
        response = self.chatbot_service.process_message(self.current_session, user_input)

        # Add user context if authenticated
        if self.current_user:
            metadata = [f"👤 {self.current_user.email}", f"📊 {self.current_user.subscription_plan}"]
            return f"{response.message}\n\n🔧 {' | '.join(metadata)}"

        return response.message

    def _get_authenticated_dashboard(self) -> str:
        """Simple authenticated dashboard using existing API services"""
        try:
            user = self.current_user
            return f"""📊 **Personal Dashboard**

    👤 **Account**: {user.email} ({user.subscription_plan.title()})
    📅 **Member Since**: {user.created_at.strftime('%B %d, %Y')}
    ✅ **Status**: Active and Authenticated

    🎯 **Features Available**:
    - ✅ Background tasks with personal quotas
    - ✅ Persistent conversation history  
    - ✅ Priority processing
    - ✅ Personal analytics

    💡 All your data is saved across sessions!"""
        except Exception as e:
            return f"❌ Dashboard error: {e}"

    def _get_authenticated_profile(self) -> str:
        """Simple authenticated profile"""
        try:
            user = self.current_user
            return f"""👤 **Your Profile**

    📧 **Email**: {user.email}
    📊 **Plan**: {user.subscription_plan.title()}
    📅 **Member Since**: {user.created_at.strftime('%B %d, %Y')}
    🔐 **Authenticated**: ✅ Yes

    💡 Use `/dashboard` for usage statistics"""
        except Exception as e:
            return f"❌ Profile error: {e}"

    def get_session_stats(self) -> str:
        """Enhanced stats with user information"""
        if not self.current_session:
            return "No active session"

        # Use existing session summary
        summary = self.chatbot_service.get_session_summary(self.current_session)

        stats = f"""📊 Session Statistics:
- Session ID: {str(self.current_session)[:8]}...
- Messages: {summary.get('message_count', 0)}"""

        # Add user info if authenticated
        if self.current_user:
            stats += f"""
- User: {self.current_user.email}
- Plan: {self.current_user.subscription_plan.title()}
- Authentication: ✅ Verified"""
        else:
            stats += f"""
- User: Guest
- Authentication: ❌ None"""

        return stats

    def cleanup(self):
        """Use existing cleanup logic"""
        try:
            if self.chatbot_service:
                self.chatbot_service.background_tasks.shutdown()
                self.chatbot_service.timeout_processor.shutdown()

            redis_manager.close()

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(postgres_manager.close())
                else:
                    asyncio.run(postgres_manager.close())
            except Exception as e:
                logger.error(f"Error closing PostgreSQL: {e}")

            logger.info("Application cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

async def main():
    """Main entry point with minimal changes"""
    app = EnhancedChatbotApplication()

    # Signal handler
    def signal_handler(signum, frame):
        print("\n🛑 Received interrupt signal. Shutting down gracefully...")
        app.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize (NO setup script needed - uses existing infrastructure)
    if not await app.initialize():
        logger.error("Failed to initialize application")
        return

    # Start authentication flow
    if not await app.start_authenticated_session():
        print("👋 Goodbye!")
        return

    # Show startup message
    print(f"\n🚀 Enhanced FAQ Bot - Leveraging Existing API Infrastructure!")
    print("=" * 70)

    if app.current_user:
        print(f"👤 Welcome, {app.current_user.email}!")
        print(f"📊 Plan: {app.current_user.subscription_plan.title()}")
        print("🔐 Using existing API authentication system")
    else:
        print("👤 Guest mode - Basic features available")

    print("\n🎯 Available Commands:")
    print("   • Basic: /stats, /feedback <message>, /my-feedback")
    print("   • Background: /analyze <task>, /research <topic>")
    if app.current_user:
        print("   • Authenticated: Enhanced quotas and personal data")
    print("   • Help: help, quit")
    print("-" * 70)

    # Interactive session (existing logic)
    try:
        while True:
            try:
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    response = app.process_user_input(user_input)
                    print(f"🤖 Bot: {response}")
                    break

                if user_input.lower() == '/session-stats':
                    print(f"📊 {app.get_session_stats()}")
                    continue

                if not user_input:
                    continue

                response = app.process_user_input(user_input)
                print(f"🤖 Bot: {response}")

            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                break
            except EOFError:
                print("\n🛑 EOF received")
                break
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                print("🤖 Bot: Sorry, something went wrong. Please try again.")

    finally:
        print("\n🔄 Cleaning up...")
        app.cleanup()
        print("👋 Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())