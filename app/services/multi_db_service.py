import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import logging

from app.database.postgres_connection import postgres_manager
from app.database.postgres_models import User, UsageRecord, AuditLog
from app.database.redis_models import CacheModel, SessionModel, AnalyticsModel
from app.database.scylla_models import ConversationHistory
from app.services.auth_service import auth_service
from app.services.billing_service import billing_service

logger = logging.getLogger(__name__)

class MultiDatabaseService:
    """
    Coordinates operations across PostgreSQL, Redis, and ScyllaDB.

    Used by: chatbot_service.py as the main coordination layer
    Integration: This is the KEY integration point that connects all three databases
    """

    def __init__(self):
        # Database connections
        self.cache_model = CacheModel()
        self.session_model = SessionModel()
        self.analytics_model = AnalyticsModel()
        self.conversation_history = ConversationHistory()

    async def process_user_message_with_auth(self, session_id: str, user_message: str,
                                             user_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user message with full three-database integration.

        Called by: chatbot_service.py for authenticated message processing
        Flow: PostgreSQL (auth) → Redis (session/cache) → ScyllaDB (storage) → PostgreSQL (usage)
        """
        # 1. POSTGRESQL: Authenticate user if token provided
        user = None
        if user_token:
            token_payload = await auth_service.verify_token(user_token)
            if token_payload:
                user_id = token_payload.get("user_id")
                user = await auth_service.get_user_by_id(uuid.UUID(user_id))

                if user and not user.is_active:
                    raise PermissionError("User account is inactive")

        # 2. REDIS: Check rate limits if user is authenticated
        if user:
            await self._check_user_rate_limits(user)

        # 3. REDIS: Get/create session with user context
        session_data = self.session_model.get_session(session_id)
        if not session_data:
            user_context = {"user_id": str(user.id)} if user else {"user_type": "anonymous"}
            self.session_model.create_session(session_id, user_context)

        # 4. REDIS: Check cache for response
        cached_response = await self._check_message_cache(user_message)
        if cached_response:
            # Record cache hit analytics
            self.analytics_model.increment_counter("cache_hits")
            if user:
                await self._record_usage(user, "cached_message")
            return cached_response

        # 5. Generate response (this would call your existing chatbot logic)
        response = await self._generate_response(user_message)

        # 6. SCYLLADB: Store conversation for persistence
        session_uuid = uuid.UUID(session_id) if session_id else uuid.uuid4()
        self.conversation_history.save_message(session_uuid, 'user', user_message)
        self.conversation_history.save_message(session_uuid, 'bot', response['message'])

        # 7. REDIS: Cache the response
        await self._cache_response(user_message, response)

        # 8. POSTGRESQL: Record usage and audit if user is authenticated
        if user:
            await self._record_usage(user, "message_processed")
            await self._log_user_activity(user, "message_sent", {"message_length": len(user_message)})

        # 9. REDIS: Update session activity
        self.session_model.add_to_chat_history(session_id, {
            "actor": "user", "message": user_message
        })
        self.session_model.add_to_chat_history(session_id, {
            "actor": "bot", "message": response['message']
        })

        return response

    async def start_background_task_with_auth(self, user_token: str, task_type: str,
                                              task_data: Dict[str, Any]) -> str:
        """
        Start background task with full authentication and quota checking.

        Called by: chatbot_service.py for background task initiation
        Flow: PostgreSQL (auth/quota) → Redis (queue) → PostgreSQL (usage/audit)
        """
        # 1. POSTGRESQL: Authenticate and authorize user
        token_payload = await auth_service.verify_token(user_token)
        if not token_payload:
            raise PermissionError("Invalid authentication token")

        user = await auth_service.get_user_by_id(uuid.UUID(token_payload["user_id"]))
        if not user or not user.is_active:
            raise PermissionError("User not authorized for background tasks")

        # 2. POSTGRESQL: Check usage quotas
        await self._check_background_task_quota(user)

        # 3. REDIS: Queue the background task (integrate with your existing background_tasks service)
        task_id = str(uuid.uuid4())

        # 4. POSTGRESQL: Record task initiation
        await self._record_usage(user, "background_task_started")
        await self._log_user_activity(user, "background_task_initiated", {
            "task_id": task_id, "task_type": task_type
        })

        logger.info(f"Background task {task_id} started for user {user.email}")
        return task_id

    async def get_user_dashboard_data(self, user_token: str) -> Dict[str, Any]:
        """
        Get comprehensive user dashboard data from all three databases.

        Called by: API endpoints for user dashboard
        Flow: PostgreSQL (user/billing) → Redis (sessions) → ScyllaDB (history)
        """
        # 1. POSTGRESQL: Get user and subscription info
        token_payload = await auth_service.verify_token(user_token)
        if not token_payload:
            raise PermissionError("Invalid authentication token")

        user = await auth_service.get_user_by_id(uuid.UUID(token_payload["user_id"]))
        if not user:
            raise ValueError("User not found")

        # 2. POSTGRESQL: Get usage statistics
        usage_stats = await self._get_user_usage_stats(user)

        # 3. REDIS: Get session information
        session_stats = self._get_user_session_stats(str(user.id))

        # 4. SCYLLADB: Get conversation statistics (simplified for this example)
        conversation_stats = {"total_conversations": 0, "recent_activity": []}

        return {
            "user": {
                "id": str(user.id),
                "email": user.email,
                "subscription_plan": user.subscription_plan,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat()
            },
            "usage": usage_stats,
            "sessions": session_stats,
            "conversations": conversation_stats
        }

    # ===== PRIVATE HELPER METHODS =====

    async def _check_user_rate_limits(self, user: User) -> None:
        """Check if user has exceeded rate limits"""
        # Implementation would check Redis rate limiting
        pass

    async def _check_message_cache(self, message: str) -> Optional[Dict[str, Any]]:
        """Check Redis cache for message response"""
        import hashlib
        message_hash = hashlib.md5(message.lower().encode()).hexdigest()
        return self.cache_model.get_response(message_hash)

    async def _generate_response(self, message: str) -> Dict[str, Any]:
        """Generate chatbot response (integrate with your existing logic)"""
        # This would integrate with your existing chatbot_service response generation
        return {
            "message": f"Response to: {message}",
            "confidence": 0.8,
            "cached": False
        }

    async def _cache_response(self, message: str, response: Dict[str, Any]) -> None:
        """Cache response in Redis"""
        import hashlib
        message_hash = hashlib.md5(message.lower().encode()).hexdigest()
        self.cache_model.set_response(message_hash, response)

    async def _record_usage(self, user: User, resource_type: str) -> None:
        """Record usage in PostgreSQL for billing"""
        success = await billing_service.record_usage(
            user=user,
            resource_type=resource_type,
            quantity=1,
            extra_data={"timestamp": datetime.now(timezone.utc).isoformat()}
        )

        if not success:
            logger.warning(f"Failed to record usage for user {user.email}")

    async def _log_user_activity(self, user: User, action: str, metadata: Dict[str, Any]) -> None:
        """Log user activity for audit trail"""
        async with postgres_manager.get_session() as session:
            audit_log = AuditLog(
                user_id=user.id,
                action=action,
                resource_type="user_activity",
                new_values=metadata
            )
            session.add(audit_log)

    async def _check_background_task_quota(self, user: User) -> None:
        """Check if user can start background tasks"""
        quota_info = await billing_service.check_user_quota(user, "background_tasks")

        if not quota_info["has_quota"]:
            raise PermissionError(
                f"Background task quota exceeded. Used {quota_info['current_usage']}/{quota_info['max_allowed']} "
                f"for your {user.subscription_plan} plan. Upgrade for more tasks."
            )

    async def _get_user_usage_stats(self, user: User) -> Dict[str, Any]:
        """Get user usage statistics from PostgreSQL"""
        return await billing_service.get_usage_summary(user)


    def _get_user_session_stats(self, user_id: str) -> Dict[str, Any]:
        """Get user session statistics from Redis"""
        # Implementation would query Redis for session data
        return {
            "active_sessions": 1,
            "last_activity": datetime.now(timezone.utc).isoformat()
        }

# Global multi-database service instance
multi_db_service = MultiDatabaseService()