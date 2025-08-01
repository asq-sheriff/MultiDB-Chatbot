# app/services/chatbot_service.py

"""
Enhanced chatbot service with Redis and PostgreSQL integration for caching, sessions,
authentication, and analytics. This service extends the original chatbot functionality
with Redis-powered features and PostgreSQL business logic.
Used by: main.py as the primary interface for chatbot functionality
"""

import uuid
import hashlib
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

# ScyllaDB imports (from your existing project)
from app.database.scylla_models import ConversationHistory, ConversationMessage, UserFeedbackRepository
from app.services.knowledge_service import KnowledgeService, MatchResult

# Redis imports (new Redis functionality)
from app.database.redis_models import CacheModel, SessionModel, AnalyticsModel, PopularityTracker, NotificationModel
from app.services.background_tasks import BackgroundTaskService

from app.services.request_analyzer import RequestAnalyzer, TaskComplexity
from app.services.timeout_processor import TimeoutProcessor

# PostgreSQL imports (new PostgreSQL functionality)
from app.services.multi_db_service import multi_db_service
from app.services.auth_service import auth_service

logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """
    Enhanced response object for chatbot interactions with Redis metadata.
    Used by: main.py to handle chatbot responses
    """
    message: str
    confidence: Optional[float] = None
    matched_keywords: Optional[List[str]] = None
    has_context: bool = False
    session_info: Optional[Dict[str, Any]] = None
    # Redis-related fields
    cached: bool = False
    response_time_ms: Optional[int] = None
    analytics_recorded: bool = False

class ChatbotService:
    """
    Enhanced chatbot service with Redis and PostgreSQL integration.
    Provides caching, session management, analytics, authentication, and business logic
    on top of existing ScyllaDB functionality.
    """

    def __init__(self):
        # Existing ScyllaDB components
        self.conversation_history = ConversationHistory()
        self.knowledge_service = KnowledgeService()
        self.feedback_repository = UserFeedbackRepository()

        # Redis components
        self.cache_model = CacheModel()
        self.session_model = SessionModel()
        self.analytics_model = AnalyticsModel()
        self.notification_model = NotificationModel()
        self.popularity_tracker = PopularityTracker()

        # Background task service
        self.background_tasks = BackgroundTaskService()

        # Intelligent request processing components
        self.request_analyzer = RequestAnalyzer()
        self.timeout_processor = TimeoutProcessor(self.background_tasks)

        # PostgreSQL integration - Multi-database coordinator
        self.multi_db_service = multi_db_service

        # Keyword sets for message classification
        self._greeting_keywords = {'hello', 'hi', 'hey', 'good', 'morning', 'afternoon', 'evening'}
        self._farewell_keywords = {'bye', 'goodbye', 'exit', 'quit', 'thanks', 'thank'}
        self._help_keywords = {'help'}
        self._analysis_keywords = {'analyze', 'analysis', 'data', 'process'}
        self._research_keywords = {'research', 'investigate', 'study', 'find'}

    def create_session(self, user_data: Dict[str, Any] = None) -> uuid.UUID:
        """
        Create a new chat session with Redis session management.
        Called by: main.py when starting a new conversation
        """
        session_id = uuid.uuid4()

        # Create Redis session
        session_data = user_data or {"user_type": "anonymous"}
        self.session_model.create_session(str(session_id), session_data)

        # Record analytics
        self.analytics_model.increment_counter("sessions_created")
        self.analytics_model.record_event("session_created", {
            "session_id": str(session_id),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        logger.info("Created new enhanced chat session: %s", session_id)
        return session_id

    def process_message(self, session_id: uuid.UUID, user_message: str) -> ChatResponse:
        """
        Process a user message with Redis caching and analytics.
        Called by: main.py for each user input
        """
        start_time = datetime.now(timezone.utc)
        user_id = f"user_{str(session_id)[:8]}"

        # Record analytics for message processing
        self.analytics_model.increment_counter("messages_processed")
        self.analytics_model.record_event("message_received", {
            "session_id": str(session_id),
            "message_length": len(user_message),
            "timestamp": start_time.isoformat()
        })

        # Add message to Redis session history for fast access
        self.session_model.add_to_chat_history(str(session_id), {
            "actor": "user",
            "message": user_message
        })

        # Save to ScyllaDB for persistence (existing functionality)
        self.conversation_history.save_message(session_id, 'user', user_message)

        # Route to appropriate handler
        response = self._route_message(session_id, user_id, user_message)

        # Add bot response to session histories
        self.session_model.add_to_chat_history(str(session_id), {
            "actor": "bot",
            "message": response.message
        })
        self.conversation_history.save_message(session_id, 'bot', response.message)

        # Calculate response time and enhance response
        end_time = datetime.now(timezone.utc)
        response_time_ms = int((end_time - start_time).total_seconds() * 1000)

        # Add notification count if user has pending notifications
        notification_count = self.notification_model.count_notifications(user_id)
        if notification_count > 0:
            response.message += f"\n\n🔔 You have {notification_count} pending notifications. Type '/notifications' to check them."

        # Create enhanced response with timing data
        enhanced_response = ChatResponse(
            message=response.message,
            confidence=response.confidence,
            matched_keywords=response.matched_keywords,
            has_context=response.has_context,
            session_info=response.session_info,
            cached=getattr(response, 'cached', False),
            response_time_ms=response_time_ms,
            analytics_recorded=True
        )

        # Record response analytics
        self.analytics_model.record_event("response_generated", {
            "session_id": str(session_id),
            "response_time_ms": response_time_ms,
            "cached": enhanced_response.cached,
            "confidence": enhanced_response.confidence,
            "timestamp": end_time.isoformat()
        })

        return enhanced_response

    # =====================================
    # POSTGRESQL AUTHENTICATION METHODS
    # =====================================

    async def process_authenticated_message(self, session_id: uuid.UUID, user_message: str,
                                            user_token: Optional[str] = None) -> ChatResponse:
        """
        Process message with PostgreSQL authentication integration.

        Called by: main.py or API endpoints when user has authentication token
        Integration: Uses multi_db_service for three-database coordination
        """
        try:
            # Use multi-database service for authenticated processing
            result = await self.multi_db_service.process_user_message_with_auth(
                session_id=str(session_id),
                user_message=user_message,
                user_token=user_token
            )

            return ChatResponse(
                message=result["message"],
                confidence=result.get("confidence"),
                cached=result.get("cached", False),
                has_context=True,
                session_info={"authentication": "verified" if user_token else "anonymous"}
            )

        except PermissionError as e:
            return ChatResponse(
                message=f"Authentication error: {str(e)}",
                has_context=True,
                session_info={"error": "authentication_failed"}
            )
        except Exception as e:
            logger.error(f"Error in authenticated message processing: {e}")
            # Fallback to existing non-authenticated processing
            return self.process_message(session_id, user_message)

    async def start_authenticated_background_task(self, task_type: str, task_data: Dict[str, Any],
                                                  user_token: str) -> str:
        """
        Start background task with authentication and quota checking.

        Called by: API endpoints for authenticated background task requests
        Integration: Uses multi_db_service for PostgreSQL quota checking
        """
        try:
            return await self.multi_db_service.start_background_task_with_auth(
                user_token=user_token,
                task_type=task_type,
                task_data=task_data
            )
        except PermissionError as e:
            raise ValueError(f"Cannot start background task: {str(e)}")

    async def process_authenticated_command(self, session_id: uuid.UUID, user_message: str,
                                            user_token: Optional[str] = None) -> ChatResponse:
        """
        Enhanced command processing with authentication support.

        Called by: main.py for commands that benefit from authentication
        Integration: Provides user-specific data when authenticated
        """
        message_lower = user_message.lower().strip()

        # Get user info if authenticated
        user = None
        if user_token:
            try:
                token_payload = await auth_service.verify_token(user_token)
                if token_payload:
                    user_id = token_payload.get("user_id")
                    user = await auth_service.get_user_by_id(uuid.UUID(user_id))
            except Exception as e:
                logger.warning(f"Token verification failed: {e}")

        # Enhanced stats command with user data
        if message_lower == '/stats' and user:
            return await self._handle_authenticated_stats_command(session_id, user)

        # Enhanced background task commands with quota checking
        elif user_message.startswith('/analyze ') and user:
            return await self._handle_authenticated_analysis_command(session_id, user, user_message)

        elif user_message.startswith('/research ') and user:
            return await self._handle_authenticated_research_command(session_id, user, user_message)

        # Dashboard command for authenticated users
        elif message_lower == '/dashboard' and user:
            return await self._handle_dashboard_command(session_id, user, user_token)

        # Profile command for authenticated users
        elif message_lower == '/profile' and user:
            return await self._handle_profile_command(session_id, user)

        # Fallback to regular command processing
        else:
            return self._route_message(session_id, f"user_{str(session_id)[:8]}", user_message)

    # =====================================
    # AUTHENTICATED COMMAND HANDLERS
    # =====================================

    async def _handle_authenticated_stats_command(self, session_id: uuid.UUID, user) -> ChatResponse:
        """Enhanced stats with user-specific data"""
        try:
            # Get regular stats
            regular_stats = self._handle_stats_command(session_id)

            # Add user-specific stats
            dashboard_data = await self.multi_db_service.get_user_dashboard_data(
                await auth_service.create_access_token({"user_id": str(user.id)})
            )

            enhanced_message = regular_stats.message + f"""

👤 **Your Account Stats:**
- Email: {user.email}
- Plan: {user.subscription_plan.title()}
- Messages this month: {dashboard_data.get('usage', {}).get('messages_this_month', 0)}
- Background tasks: {dashboard_data.get('usage', {}).get('background_tasks_this_month', 0)}
- Member since: {user.created_at.strftime('%Y-%m-%d')}"""

            return ChatResponse(
                message=enhanced_message,
                has_context=True,
                session_info={'command_type': 'authenticated_stats_success', 'user_id': str(user.id)}
            )

        except Exception as e:
            logger.error(f"Failed to get authenticated stats: {e}")
            return self._handle_stats_command(session_id)  # Fallback

    async def _handle_authenticated_analysis_command(self, session_id: uuid.UUID, user, user_message: str) -> ChatResponse:
        """Enhanced analysis with quota checking"""
        analysis_description = user_message[9:].strip()

        if not analysis_description:
            return ChatResponse(
                message="📊 Please provide a description of what you'd like me to analyze.\n\nExample: /analyze large customer dataset for purchasing patterns",
                has_context=True,
                session_info={'command_type': 'analysis_invalid'}
            )

        try:
            # Check quotas using multi_db_service
            user_token = await auth_service.create_access_token({"user_id": str(user.id)})
            task_id = await self.multi_db_service.start_background_task_with_auth(
                user_token=user_token,
                task_type="analysis",
                task_data={"description": analysis_description}
            )

            response_message = f"""📊 **Authenticated Analysis Started!**

🎯 **Task**: {analysis_description}
🆔 **Task ID**: {task_id[:8]}...
👤 **User**: {user.email}
📊 **Plan**: {user.subscription_plan.title()}
⏱️ **Status**: Processing with priority based on your plan

I'll analyze your data and send you a notification when it's complete!

💡 Use '/notifications' to check for completion updates"""

            return ChatResponse(
                message=response_message,
                has_context=True,
                session_info={
                    'command_type': 'authenticated_analysis_submitted',
                    'task_id': task_id,
                    'user_id': str(user.id)
                }
            )

        except PermissionError as e:
            return ChatResponse(
                message=f"❌ **Quota Limit Reached**\n\n{str(e)}\n\nConsider upgrading your plan for more background tasks!",
                has_context=True,
                session_info={'command_type': 'analysis_quota_exceeded', 'user_id': str(user.id)}
            )
        except Exception as e:
            logger.error(f"Failed to submit authenticated analysis task: {e}")
            # Fallback to regular analysis
            return self._handle_analysis_command(session_id, f"user_{str(session_id)[:8]}", user_message)

    async def _handle_authenticated_research_command(self, session_id: uuid.UUID, user, user_message: str) -> ChatResponse:
        """Enhanced research with quota checking"""
        research_topic = user_message[10:].strip()

        if not research_topic:
            return ChatResponse(
                message="🔍 Please provide a topic you'd like me to research.\n\nExample: /research latest trends in artificial intelligence",
                has_context=True,
                session_info={'command_type': 'research_invalid'}
            )

        try:
            user_token = await auth_service.create_access_token({"user_id": str(user.id)})
            task_id = await self.multi_db_service.start_background_task_with_auth(
                user_token=user_token,
                task_type="research",
                task_data={"topic": research_topic}
            )

            response_message = f"""🔍 **Authenticated Research Started!**

🎯 **Topic**: {research_topic}
🆔 **Task ID**: {task_id[:8]}...
👤 **User**: {user.email}
📊 **Plan**: {user.subscription_plan.title()}
⏱️ **Status**: Researching with priority access

I'll gather comprehensive information and send you a notification when complete!

💡 Use '/notifications' to check for research updates"""

            return ChatResponse(
                message=response_message,
                has_context=True,
                session_info={
                    'command_type': 'authenticated_research_submitted',
                    'task_id': task_id,
                    'user_id': str(user.id)
                }
            )

        except PermissionError as e:
            return ChatResponse(
                message=f"❌ **Quota Limit Reached**\n\n{str(e)}\n\nConsider upgrading your plan for more research tasks!",
                has_context=True,
                session_info={'command_type': 'research_quota_exceeded', 'user_id': str(user.id)}
            )
        except Exception as e:
            logger.error(f"Failed to submit authenticated research task: {e}")
            # Fallback to regular research
            return self._handle_research_command(session_id, f"user_{str(session_id)[:8]}", user_message)

    async def _handle_dashboard_command(self, session_id: uuid.UUID, user, user_token: str) -> ChatResponse:
        """User dashboard with comprehensive data"""
        try:
            dashboard_data = await self.multi_db_service.get_user_dashboard_data(user_token)

            usage = dashboard_data.get('usage', {})
            sessions = dashboard_data.get('sessions', {})

            dashboard_message = f"""📊 **Personal Dashboard**

👤 **Account Information:**
- Email: {user.email}
- Subscription: {user.subscription_plan.title()}
- Status: {'✅ Active' if user.is_active else '❌ Inactive'}
- Member since: {user.created_at.strftime('%B %d, %Y')}

📈 **Usage This Month:**
- Messages: {usage.get('messages_this_month', 0)}
- Background Tasks: {usage.get('background_tasks_this_month', 0)}
- Quota Remaining: {usage.get('quota_remaining', 'Unlimited')}

🔄 **Session Activity:**
- Active Sessions: {sessions.get('active_sessions', 0)}
- Last Activity: {sessions.get('last_activity', 'Unknown')}

💡 **Available Commands:**
- `/profile` - View/edit profile
- `/stats` - Enhanced statistics
- `/analyze <task>` - Priority analysis
- `/research <topic>` - Priority research"""

            return ChatResponse(
                message=dashboard_message,
                has_context=True,
                session_info={
                    'command_type': 'dashboard_success',
                    'user_id': str(user.id),
                    'subscription_plan': user.subscription_plan
                }
            )

        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return ChatResponse(
                message=f"❌ Sorry, I couldn't load your dashboard right now. Your account: {user.email} ({user.subscription_plan})",
                has_context=True,
                session_info={'command_type': 'dashboard_error', 'user_id': str(user.id)}
            )

    async def _handle_profile_command(self, session_id: uuid.UUID, user) -> ChatResponse:
        """User profile information"""
        profile_message = f"""👤 **Your Profile**

📧 **Email:** {user.email}
📊 **Subscription:** {user.subscription_plan.title()}
📅 **Member Since:** {user.created_at.strftime('%B %d, %Y')}
✅ **Status:** {'Active' if user.is_active else 'Inactive'}
🔐 **Verified:** {'Yes' if user.is_verified else 'No'}

🎯 **Subscription Benefits:**
{'🌟 Premium support, Higher quotas, Priority processing' if user.subscription_plan == 'pro' else
        '🚀 All Premium features, Unlimited usage, Custom integrations' if user.subscription_plan == 'enterprise' else
        '📝 Basic features, Standard quotas, Community support'}

💡 Use `/dashboard` for usage statistics and `/stats` for detailed analytics."""

        return ChatResponse(
            message=profile_message,
            has_context=True,
            session_info={
                'command_type': 'profile_success',
                'user_id': str(user.id),
                'subscription_plan': user.subscription_plan
            }
        )

    # =====================================
    # MESSAGE ROUTING AND PROCESSING
    # =====================================

    def _route_message(self, session_id: uuid.UUID, user_id: str, user_message: str) -> ChatResponse:
        """
        Enhanced message routing with intelligent background task detection.
        Combines explicit commands, automatic detection, and timeout-based processing.
        """
        message_lower = user_message.lower().strip()

        # PHASE 1: Handle explicit commands first (highest priority) - FIXED
        if user_message.startswith('/feedback '):
            return self._handle_feedback_command(session_id, user_message)
        elif message_lower == '/my-feedback':
            return self._handle_view_feedback_command(session_id)
        elif message_lower == '/stats':
            return self._handle_stats_command(session_id)
        elif message_lower == '/trending':
            return self._handle_trending_command(session_id)
        elif message_lower == '/notifications':
            return self._handle_notifications_command(session_id, user_id)
        elif message_lower == '/notifications peek':
            return self._handle_notifications_peek_command(session_id, user_id)
        elif message_lower == '/notifications clear':
            return self._handle_notifications_clear_command(session_id, user_id)
        elif user_message.startswith('/analyze '):
            return self._handle_analysis_command(session_id, user_id, user_message)
        elif user_message.startswith('/research '):
            return self._handle_research_command(session_id, user_id, user_message)

        # PHASE 1.5: Handle PostgreSQL-dependent commands - NEW
        elif message_lower == '/dashboard':
            return self._handle_dashboard_command_fallback(session_id)
        elif message_lower == '/profile':
            return self._handle_profile_command_fallback(session_id)

        # PHASE 2: Handle simple conversational messages
        if self._is_greeting(message_lower):
            return self._handle_greeting(session_id)
        elif self._is_farewell(message_lower):
            return self._handle_farewell(session_id)
        elif self._is_help_request(message_lower):
            return self._handle_help_request()

        # PHASE 3: INTELLIGENT ANALYSIS - Continue with existing logic
        analysis = self.request_analyzer.analyze_request(user_message)

        logger.info(f"Request analysis for '{user_message[:50]}...': "
                    f"complexity={analysis.complexity.value}, "
                    f"duration={analysis.estimated_duration_seconds}s, "
                    f"background={analysis.should_background}, "
                    f"confidence={analysis.confidence:.2f}")

        # Record analytics for intelligent routing
        self.analytics_model.record_event("intelligent_analysis", {
            "session_id": str(session_id),
            "complexity": analysis.complexity.value,
            "estimated_duration": analysis.estimated_duration_seconds,
            "should_background": analysis.should_background,
            "confidence": analysis.confidence,
            "task_type": analysis.task_type,
            "detected_keywords": analysis.detected_keywords
        })

        # Continue with existing decision tree...
        # [Rest of the method stays the same]

        # Option A: High-confidence automatic background processing
        if (analysis.should_background and
                analysis.confidence >= 0.6 and
                analysis.complexity in [TaskComplexity.COMPLEX, TaskComplexity.HEAVY]):

            logger.info(f"AUTO-BACKGROUND: High confidence ({analysis.confidence:.2f}) "
                        f"complex task detected, routing to background immediately")

            return self._handle_automatic_background_task(
                session_id, user_id, user_message, analysis
            )

        # Option B: Medium-confidence timeout-based processing
        elif (analysis.should_background and
              analysis.confidence >= 0.4 and
              analysis.estimated_duration_seconds >= 5):

            logger.info(f"TIMEOUT-BASED: Medium confidence ({analysis.confidence:.2f}) "
                        f"task detected, using timeout processor")

            return self._handle_timeout_based_processing(
                session_id, user_id, user_message, analysis
            )

        # Option C: Suggest background processing to user
        elif (analysis.should_background and
              analysis.confidence >= 0.3):

            logger.info(f"SUGGEST-BACKGROUND: Lower confidence ({analysis.confidence:.2f}) "
                        f"but potential complex task, suggesting to user")

            # First provide a quick response, then suggest background option
            quick_response = self._generate_cached_response(session_id, user_message)

            # Enhance response with suggestion
            suggestion_text = self._create_background_suggestion(analysis, user_message)
            quick_response.message += f"\n\n{suggestion_text}"

            return quick_response

        # Option D: Normal processing (default path)
        else:
            logger.debug(f"NORMAL: Standard processing for message (confidence={analysis.confidence:.2f})")
            return self._generate_cached_response(session_id, user_message)

    def _handle_automatic_background_task(self, session_id: uuid.UUID, user_id: str,
                                          user_message: str, analysis) -> ChatResponse:
        """
        Handle requests that are automatically routed to background processing.
        NEW METHOD - called when high confidence complex task is detected.
        """
        try:
            # Determine task type and submit to background
            if analysis.task_type == "analysis" or "analy" in user_message.lower():
                description = user_message
                task_id = self.background_tasks.submit_data_analysis_task(
                    user_id=user_id,
                    data_description=description,
                    session_id=str(session_id)
                )
                task_type_display = "Analysis"
            else:
                topic = user_message
                task_id = self.background_tasks.submit_research_task(
                    user_id=user_id,
                    research_topic=topic,
                    session_id=str(session_id)
                )
                task_type_display = "Research"

            # Create user-friendly response
            response_message = f"""🤖 **Intelligent Processing Activated**

I've detected that your request requires complex processing ({analysis.complexity.value} task, ~{analysis.estimated_duration_seconds}s estimated).

**Auto-started**: {task_type_display} Task
🎯 **Request**: {user_message}
🆔 **Task ID**: {task_id[:8]}...
⏱️ **Status**: Processing in background
🔍 **Detected**: {', '.join(analysis.detected_keywords[:3]) if analysis.detected_keywords else 'Complex processing needed'}

You'll receive a notification when complete! Feel free to continue our conversation.

💡 Use '/notifications' to check progress"""

            return ChatResponse(
                message=response_message,
                has_context=True,
                session_info={
                    'command_type': 'automatic_background',
                    'task_id': task_id,
                    'analysis': {
                        'complexity': analysis.complexity.value,
                        'confidence': analysis.confidence,
                        'reason': analysis.reason
                    }
                }
            )

        except Exception as e:
            logger.error(f"Failed to auto-start background task: {e}")
            # Fallback to normal processing
            return self._generate_cached_response(session_id, user_message)

    def _handle_timeout_based_processing(self, session_id: uuid.UUID, user_id: str,
                                         user_message: str, analysis) -> ChatResponse:
        """
        Handle requests using timeout-based processing.
        NEW METHOD - starts normally but moves to background if it takes too long.
        """
        try:
            # Define the actual processing function
            def process_request():
                return self._generate_cached_response(session_id, user_message)

            # Determine task type for potential background processing
            task_type = "analysis" if analysis.task_type == "analysis" else "research"

            # Process with timeout monitoring
            result = self.timeout_processor.process_with_timeout(
                task_function=process_request,
                timeout_threshold=analysis.estimated_duration_seconds * 0.8,  # 80% of estimated time
                session_id=str(session_id),
                user_id=user_id,
                user_message=user_message,
                task_type=task_type
            )

            # Check if result is a timeout response (dict) or normal response (ChatResponse)
            if isinstance(result, dict) and result.get('timeout_transferred'):
                # Convert timeout response to ChatResponse
                return ChatResponse(
                    message=result['message'],
                    has_context=result['has_context'],
                    session_info=result['session_info']
                )
            else:
                # Normal completion within timeout
                # Add a note about intelligent processing
                result.message += f"\n\n⚡ *Completed with intelligent processing ({analysis.complexity.value} task)*"
                return result

        except Exception as e:
            logger.error(f"Failed in timeout-based processing: {e}")
            # Fallback to normal processing
            return self._generate_cached_response(session_id, user_message)

    def _create_background_suggestion(self, analysis, user_message: str) -> str:
        """
        Create a suggestion message for potential background processing.
        NEW METHOD - suggests background processing to users.
        """
        task_type = analysis.task_type or "processing"

        suggestion = f"""💡 **Smart Processing Suggestion**

Based on your request, this might be a good candidate for background processing:
- **Complexity**: {analysis.complexity.value.title()} task
- **Estimated time**: ~{analysis.estimated_duration_seconds} seconds
- **Type**: {task_type.title()}

**Want faster results?** Try:
• `/analyze {user_message}` for detailed background analysis
• `/research {user_message}` for comprehensive background research

Background tasks send notifications when complete! 🔔"""

        return suggestion

    def _generate_cached_response(self, session_id: uuid.UUID, user_message: str) -> ChatResponse:
        """
        SINGLE method for generating cached responses with enhanced features.
        Combines caching, popularity tracking, and analytics.
        """
        # Create cache key from message hash
        message_hash = hashlib.md5(user_message.lower().strip().encode()).hexdigest()

        # Try cache first
        cached_response = self.cache_model.get_response(message_hash)
        if cached_response:
            self.analytics_model.increment_counter("cache_hits")
            # Track popularity of cached questions
            self.popularity_tracker.increment_question_popularity(message_hash)

            response_data = cached_response.get("response", {})
            return ChatResponse(
                message=response_data.get("message", ""),
                confidence=response_data.get("confidence"),
                matched_keywords=response_data.get("matched_keywords", []),
                has_context=response_data.get("has_context", False),
                session_info=response_data.get("session_info"),
                cached=True
            )

        # Cache miss - generate new response
        self.analytics_model.increment_counter("cache_misses")
        response = self._generate_knowledge_response(session_id, user_message)

        # Cache high-quality responses with enhanced metadata
        if response.has_context and response.confidence and response.confidence > 0.3:
            # Generate smart tags for cache invalidation
            tags = []
            if response.matched_keywords:
                tags.extend([f"keyword:{kw}" for kw in response.matched_keywords[:3]])
            if response.confidence:
                confidence_tier = "high" if response.confidence > 0.8 else "medium"
                tags.append(f"confidence:{confidence_tier}")

            cache_data = {
                "message": response.message,
                "confidence": response.confidence,
                "matched_keywords": response.matched_keywords,
                "has_context": response.has_context,
                "session_info": response.session_info
            }

            # Use enhanced caching with metadata
            self.cache_model.cache_with_metadata(message_hash, cache_data, tags=tags)
            # Track popularity of new questions
            self.popularity_tracker.increment_question_popularity(message_hash)

            logger.debug("Cached response for message: %s", user_message[:50])

        return response

    def _generate_knowledge_response(self, session_id: uuid.UUID, user_message: str) -> ChatResponse:
        """
        Generate response using knowledge service or handle unknown queries.
        This is the core response generation without caching logic.
        """
        # Try to find knowledge match
        match_result = self.knowledge_service.get_best_answer(user_message)

        if match_result:
            return ChatResponse(
                message=match_result.entry.answer,
                confidence=match_result.confidence,
                matched_keywords=match_result.matched_keywords,
                has_context=True
            )

        # No knowledge match - handle as unknown query
        return self._handle_unknown_query(user_message)

    # =====================================
    # COMMAND HANDLERS
    # =====================================

    def _handle_notifications_peek_command(self, session_id: uuid.UUID, user_id: str) -> ChatResponse:
        """Handle '/notifications peek' command - preview without removing"""
        try:
            notifications = self.notification_model.peek_notifications(user_id, count=3)
            total_count = self.notification_model.count_notifications(user_id)

            if not notifications:
                return ChatResponse(
                    message="👀 **No notifications to preview!**\n\nYou're all caught up. Background tasks will notify you when they complete.\n\n💡 Use '/notifications' to read and mark notifications as read",
                    has_context=True,
                    session_info={'command_type': 'notifications_peek_empty'}
                )

            message = f"👀 **Preview of {total_count} notification{'s' if total_count != 1 else ''}** (not marked as read):\n\n"

            for i, notification in enumerate(notifications, 1):
                timestamp = datetime.fromisoformat(notification['timestamp'].replace('Z', '+00:00'))
                time_str = timestamp.strftime("%m/%d %H:%M")

                icon = "✅" if notification['type'] == 'success' else "❌" if notification['type'] == 'error' else "ℹ️"

                message += f"**{i}.** {icon} {notification['title']}\n"
                message += f"📅 *{time_str}*\n\n"

            if total_count > len(notifications):
                message += f"... and {total_count - len(notifications)} more\n\n"

            message += "💡 Use `/notifications` to read full details and mark as read"

            return ChatResponse(
                message=message,
                has_context=True,
                session_info={
                    'command_type': 'notifications_peek_success',
                    'preview_count': len(notifications),
                    'total_count': total_count
                }
            )

        except Exception as e:
            logger.error(f"Failed to peek notifications for user {user_id}: {e}")
            return ChatResponse(
                message="❌ Sorry, I couldn't preview your notifications right now.",
                has_context=True,
                session_info={'command_type': 'notifications_peek_error'}
            )

    def _handle_notifications_command(self, session_id: uuid.UUID, user_id: str) -> ChatResponse:
        """Handle '/notifications' command - get and mark notifications as read"""
        try:
            notifications = self.notification_model.get_notifications(user_id, count=5)

            if not notifications:
                return ChatResponse(
                    message="📬 No new notifications!\n\nYou're all caught up. New notifications will appear here when background tasks complete.",
                    has_context=True,
                    session_info={'command_type': 'notifications_empty'}
                )

            # Format notifications for display with better organization
            message = f"📬 **{len(notifications)} Notification{'s' if len(notifications) > 1 else ''}** (marked as read):\n\n"

            for i, notification in enumerate(notifications, 1):
                timestamp = datetime.fromisoformat(notification['timestamp'].replace('Z', '+00:00'))
                time_str = timestamp.strftime("%m/%d %H:%M")

                message += f"**{i}.** {notification['message']}\n"
                message += f"📅 *Completed: {time_str}*\n\n"

            message += "💡 **Tips**:\n"
            message += "• Use `/notifications peek` to preview without marking as read\n"
            message += "• Background tasks automatically notify you when complete"

            self.analytics_model.increment_counter("notifications_checked")

            return ChatResponse(
                message=message,
                has_context=True,
                session_info={
                    'command_type': 'notifications_success',
                    'notification_count': len(notifications)
                }
            )

        except Exception as e:
            logger.error(f"Failed to get notifications for user {user_id}: {e}")
            return ChatResponse(
                message="❌ Sorry, I couldn't retrieve your notifications right now. Please try again later.",
                has_context=True,
                session_info={'command_type': 'notifications_error'}
            )

    def _handle_notifications_clear_command(self, session_id: uuid.UUID, user_id: str) -> ChatResponse:
        """Handle '/notifications clear' command - clear all notifications"""
        try:
            count = self.notification_model.clear_notifications(user_id)

            if count == 0:
                message = "📬 No notifications to clear!\n\nYou're all caught up."
            else:
                message = f"🗑️ **Cleared {count} notification{'s' if count != 1 else ''}**\n\nYour notification queue is now empty."

            return ChatResponse(
                message=message,
                has_context=True,
                session_info={
                    'command_type': 'notifications_clear_success',
                    'cleared_count': count
                }
            )

        except Exception as e:
            logger.error(f"Failed to clear notifications for user {user_id}: {e}")
            return ChatResponse(
                message="❌ Sorry, I couldn't clear your notifications right now.",
                has_context=True,
                session_info={'command_type': 'notifications_clear_error'}
            )

    def _handle_analysis_command(self, session_id: uuid.UUID, user_id: str, user_message: str) -> ChatResponse:
        """Handle '/analyze <description>' command - submit analysis task"""
        analysis_description = user_message[9:].strip()  # Remove '/analyze '

        if not analysis_description:
            return ChatResponse(
                message="📊 Please provide a description of what you'd like me to analyze.\n\nExample: /analyze large customer dataset for purchasing patterns",
                has_context=True,
                session_info={'command_type': 'analysis_invalid'}
            )

        try:
            task_id = self.background_tasks.submit_data_analysis_task(
                user_id=user_id,
                data_description=analysis_description,
                session_id=str(session_id)
            )

            response_message = f"""📊 Data Analysis Started!

🎯 **Task**: {analysis_description}
🆔 **Task ID**: {task_id[:8]}...
⏱️ **Status**: Processing in background

I'll analyze your data and send you a notification when it's complete. You can continue chatting normally - I'll let you know when the results are ready!

💡 Use '/notifications' to check for completion updates"""

            return ChatResponse(
                message=response_message,
                has_context=True,
                session_info={
                    'command_type': 'analysis_submitted',
                    'task_id': task_id
                }
            )

        except Exception as e:
            logger.error(f"Failed to submit analysis task: {e}")
            return ChatResponse(
                message="❌ Sorry, I couldn't start the analysis task right now. Please try again later.",
                has_context=True,
                session_info={'command_type': 'analysis_error'}
            )

    def _handle_research_command(self, session_id: uuid.UUID, user_id: str, user_message: str) -> ChatResponse:
        """Handle '/research <topic>' command - submit research task"""
        research_topic = user_message[10:].strip()  # Remove '/research '

        if not research_topic:
            return ChatResponse(
                message="🔍 Please provide a topic you'd like me to research.\n\nExample: /research latest trends in artificial intelligence",
                has_context=True,
                session_info={'command_type': 'research_invalid'}
            )

        try:
            task_id = self.background_tasks.submit_research_task(
                user_id=user_id,
                research_topic=research_topic,
                session_id=str(session_id)
            )

            response_message = f"""🔍 Research Task Started!

🎯 **Topic**: {research_topic}
🆔 **Task ID**: {task_id[:8]}...
⏱️ **Status**: Researching in background

I'll gather comprehensive information on this topic and send you a notification when the research is complete. Feel free to ask other questions while I work!

💡 Use '/notifications' to check for research updates"""

            return ChatResponse(
                message=response_message,
                has_context=True,
                session_info={
                    'command_type': 'research_submitted',
                    'task_id': task_id
                }
            )

        except Exception as e:
            logger.error(f"Failed to submit research task: {e}")
            return ChatResponse(
                message="❌ Sorry, I couldn't start the research task right now. Please try again later.",
                has_context=True,
                session_info={'command_type': 'research_error'}
            )

    def _handle_stats_command(self, session_id: uuid.UUID) -> ChatResponse:
        """Handle stats command to show Redis analytics"""
        try:
            # Get analytics from Redis
            sessions_created = self.analytics_model.redis.get("analytics:counter:sessions_created") or 0
            messages_processed = self.analytics_model.redis.get("analytics:counter:messages_processed") or 0
            cache_hits = self.analytics_model.redis.get("analytics:counter:cache_hits") or 0
            cache_misses = self.analytics_model.redis.get("analytics:counter:cache_misses") or 0

            # Calculate cache hit rate
            total_cache_requests = int(cache_hits) + int(cache_misses)
            hit_rate = (int(cache_hits) / total_cache_requests * 100) if total_cache_requests > 0 else 0

            # Get current session info
            session_data = self.session_model.get_session(str(session_id))
            session_messages = len(session_data.get("chat_history", [])) if session_data else 0

            stats_message = f"""📊 Chatbot Statistics:

🔢 Global Stats:
- Total Sessions: {sessions_created}
- Total Messages: {messages_processed}
- Cache Hit Rate: {hit_rate:.1f}% ({cache_hits}/{total_cache_requests} requests)

👤 Your Session:
- Messages in this session: {session_messages}
- Session ID: {str(session_id)[:8]}...

💾 Cache Performance:
- Cache Hits: {cache_hits}
- Cache Misses: {cache_misses}"""

            return ChatResponse(
                message=stats_message,
                has_context=True,
                session_info={'command_type': 'stats_success'}
            )

        except Exception as e:
            logger.error("Failed to retrieve stats: %s", str(e))
            return ChatResponse(
                message="Sorry, I couldn't retrieve statistics right now. Please try again later.",
                has_context=True,
                session_info={'command_type': 'stats_error'}
            )

    def _handle_trending_command(self, session_id: uuid.UUID) -> ChatResponse:
        """Show trending questions"""
        try:
            trending = self.popularity_tracker.get_trending_questions(5)

            if not trending:
                return ChatResponse(
                    message="No trending questions yet. Keep asking to see what's popular!",
                    has_context=True,
                    session_info={'command_type': 'trending_empty'}
                )

            message = "🔥 Trending Questions Today:\n\n"
            for i, (question_hash, score) in enumerate(trending, 1):
                message += f"{i}. Question #{question_hash[:8]}... (🔥 {int(score)} views)\n"

            message += "\n💡 These are the most popular questions today!"

            return ChatResponse(
                message=message,
                has_context=True,
                session_info={'command_type': 'trending_success', 'count': len(trending)}
            )
        except Exception as e:
            logger.error("Failed to get trending questions: %s", str(e))
            return ChatResponse(
                message="Sorry, I couldn't get trending questions right now.",
                has_context=True,
                session_info={'command_type': 'trending_error'}
            )

    def _handle_feedback_command(self, session_id: uuid.UUID, user_message: str) -> ChatResponse:
        """Handle feedback submission command"""
        feedback_message = user_message[10:].strip()

        if not feedback_message:
            return ChatResponse(
                message="Please provide feedback after the command. Example: /feedback This chatbot is helpful!",
                has_context=True,
                session_info={'command_type': 'feedback_invalid'}
            )

        try:
            user_id = f"user_{str(session_id)[:8]}"
            feedback_id = self.feedback_repository.save_feedback(
                user_id=user_id,
                session_id=session_id,
                feedback_message=feedback_message
            )

            self.analytics_model.increment_counter("feedback_submitted")
            self.analytics_model.record_event("feedback_submitted", {
                "session_id": str(session_id),
                "feedback_length": len(feedback_message),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

            return ChatResponse(
                message=f"Thank you for your feedback! Your feedback has been recorded (ID: {str(feedback_id)[:8]}). "
                        f"Use '/my-feedback' to view your previous feedback or '/stats' to see chatbot statistics.",
                has_context=True,
                session_info={
                    'command_type': 'feedback_success',
                    'feedback_id': feedback_id,
                    'user_id': user_id
                }
            )

        except Exception as e:
            logger.error("Failed to save feedback: %s", str(e))
            return ChatResponse(
                message="Sorry, I couldn't save your feedback right now. Please try again later.",
                has_context=True,
                session_info={'command_type': 'feedback_error'}
            )

    def _handle_view_feedback_command(self, session_id: uuid.UUID) -> ChatResponse:
        """Handle viewing user's previous feedback"""
        try:
            user_id = f"user_{str(session_id)[:8]}"
            feedback_list = self.feedback_repository.get_user_feedback(user_id, limit=10)

            if not feedback_list:
                return ChatResponse(
                    message="You haven't submitted any feedback yet. Use '/feedback <your message>' to share your thoughts!",
                    has_context=True,
                    session_info={'command_type': 'feedback_view_empty'}
                )

            feedback_text = f"Your Recent Feedback ({len(feedback_list)} entries):\n"
            feedback_text += "-" * 40 + "\n"

            for i, feedback in enumerate(feedback_list, 1):
                timestamp_str = feedback.timestamp.strftime("%Y-%m-%d %H:%M")
                feedback_preview = feedback.feedback_message[:50]
                if len(feedback.feedback_message) > 50:
                    feedback_preview += "..."

                feedback_text += f"{i}. [{timestamp_str}] {feedback_preview}\n"

                if i >= 5:
                    remaining = len(feedback_list) - 5
                    if remaining > 0:
                        feedback_text += f"... and {remaining} more entries\n"
                    break

            return ChatResponse(
                message=feedback_text,
                has_context=True,
                session_info={
                    'command_type': 'feedback_view_success',
                    'feedback_count': len(feedback_list)
                }
            )

        except Exception as e:
            logger.error("Failed to retrieve feedback: %s", str(e))
            return ChatResponse(
                message="Sorry, I couldn't retrieve your feedback right now. Please try again later.",
                has_context=True,
                session_info={'command_type': 'feedback_view_error'}
            )

    # =====================================
    # CONTENT-BASED HANDLERS
    # =====================================

    def _handle_greeting(self, session_id: uuid.UUID) -> ChatResponse:
        """Handle greeting messages"""
        session_data = self.session_model.get_session(str(session_id))
        if session_data:
            chat_history = session_data.get("chat_history", [])
            is_first_interaction = len(chat_history) <= 1
        else:
            history = self.get_session_context(session_id, 2)
            is_first_interaction = len(history) <= 1

        if is_first_interaction:
            message = ("Hello! Welcome to the Enhanced FAQ Bot with Redis acceleration and PostgreSQL business logic. "
                       "I can help answer questions about various topics with lightning-fast responses. You can ask me anything, "
                       "or type 'help' to see what I can do for you.\n\n"
                       "💡 Commands: /feedback <message>, /my-feedback, /stats, /notifications, /dashboard, /profile")
        else:
            message = "Hello again! How can I help you today?"

        return ChatResponse(
            message=message,
            has_context=True,
            session_info={'greeting_type': 'first' if is_first_interaction else 'repeat'}
        )

    def _handle_farewell(self, session_id: uuid.UUID) -> ChatResponse:
        """Handle farewell messages"""
        session_summary = self.get_session_summary(session_id)
        message_count = session_summary.get('user_messages', 0)

        self.analytics_model.record_event("session_ended", {
            "session_id": str(session_id),
            "total_messages": message_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        if message_count <= 1:
            message = "Goodbye! Thanks for trying out the Enhanced FAQ Bot."
        else:
            message = (f"Goodbye! We covered {message_count} questions in this session. "
                       "Feel free to come back anytime!")

        return ChatResponse(
            message=message,
            has_context=True,
            session_info={'farewell_type': 'short' if message_count <= 1 else 'full'}
        )

    def _handle_help_request(self) -> ChatResponse:
        """Handle help requests with enhanced commands"""
        keywords = self.knowledge_service.get_available_keywords()

        base_message = ("I'm an Enhanced FAQ bot with Redis acceleration and PostgreSQL business logic that can answer "
                        "questions on various topics. You can ask me about things naturally, and I'll try to find "
                        "the best answer for you.")

        if keywords:
            example_keywords = keywords[:5]
            keyword_text = ", ".join(example_keywords)
            message = (f"{base_message}\n\n"
                       f"Some topics I can help with include: {keyword_text}")

            if len(keywords) > 5:
                message += f" and {len(keywords) - 5} more topics."
        else:
            message = f"{base_message} The knowledge base is currently being set up."

        message += "\n\n💡 **Commands:**\n"
        message += "• `/feedback <message>` - Share your feedback\n"
        message += "• `/my-feedback` - View your previous feedback\n"
        message += "• `/stats` - View chatbot usage statistics\n"
        message += "• `/trending` - See trending questions\n\n"

        message += "🔔 **Notifications & Background Tasks:**\n"
        message += "• `/notifications` - Check your notifications\n"
        message += "• `/notifications peek` - Preview notifications\n"
        message += "• `/notifications clear` - Clear all notifications\n"
        message += "• `/analyze <description>` - Start background data analysis\n"
        message += "• `/research <topic>` - Start background research task\n\n"

        message += "👤 **User Account (when authenticated):**\n"
        message += "• `/dashboard` - Personal dashboard with usage stats\n"
        message += "• `/profile` - View your profile information\n\n"

        message += "⚡ **Background tasks** run while you continue chatting and notify you when complete!"

        return ChatResponse(
            message=message,
            has_context=True,
            session_info={'available_keywords': len(keywords)}
        )

    def _handle_unknown_query(self, user_message: str) -> ChatResponse:
        """Handle unknown queries with suggestions"""
        keywords = self.knowledge_service.extract_keywords(user_message)

        self.analytics_model.increment_counter("unknown_queries")
        self.analytics_model.record_event("unknown_query", {
            "message": user_message[:100],
            "keywords": keywords,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        if keywords:
            suggestions = []
            for keyword in keywords[:3]:
                similar = self.knowledge_service.suggest_keywords(keyword)
                suggestions.extend(similar)

            if suggestions:
                unique_suggestions = list(set(suggestions))[:5]
                suggestion_text = ", ".join(unique_suggestions)
                message = (f"I don't have a specific answer for that question. "
                           f"However, I can help with topics like: {suggestion_text}. "
                           f"Try asking about one of these areas!")
            else:
                message = ("I don't have information about that topic yet. "
                           "Try asking about something else, or type 'help' to see "
                           "what I can assist with.")
        else:
            message = ("I'm not sure how to help with that. Could you try rephrasing "
                       "your question or type 'help' to see what I can do?")

        message += "\n\n💡 Use '/feedback <your message>' to suggest improvements or '/stats' to see usage statistics"

        return ChatResponse(
            message=message,
            has_context=False,
            session_info={'query_type': 'unknown', 'extracted_keywords': keywords}
        )

    # =====================================
    # HELPER METHODS
    # =====================================

    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        words = set(message.split())
        return bool(words.intersection(self._greeting_keywords))

    def _is_farewell(self, message: str) -> bool:
        """Check if message is a farewell"""
        words = set(message.split())
        return bool(words.intersection(self._farewell_keywords))

    def _is_help_request(self, message: str) -> bool:
        """Check if message is a help request"""
        words = set(message.split())
        if 'help' in words:
            return True

        help_phrases = [
            'what can you do',
            'what are your commands',
            'show me commands'
        ]

        for phrase in help_phrases:
            if phrase in message:
                return True

        return False

    # =====================================
    # UTILITY METHODS
    # =====================================

    def get_session_context_fast(self, session_id: uuid.UUID, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation context from Redis (fast access)"""
        try:
            session_data = self.session_model.get_session(str(session_id))
            if not session_data:
                return []

            chat_history = session_data.get("chat_history", [])
            return chat_history[-limit:] if len(chat_history) > limit else chat_history

        except Exception as e:
            logger.error("Failed to get fast session context: %s", str(e))
            return self.get_session_context(session_id, limit)

    def get_session_context(self, session_id: uuid.UUID, limit: int = 10) -> List[ConversationMessage]:
        """Get recent conversation context from ScyllaDB (fallback/persistent access)"""
        try:
            return self.conversation_history.get_session_history(session_id, limit)
        except Exception as e:
            logger.error("Failed to get session context: %s", str(e))
            return []

    def get_session_summary(self, session_id: uuid.UUID) -> Dict[str, Any]:
        """Get session summary with Redis data"""
        try:
            # Try Redis first for fast access
            session_data = self.session_model.get_session(str(session_id))
            if session_data:
                chat_history = session_data.get("chat_history", [])
                user_messages = sum(1 for msg in chat_history if msg.get("actor") == "user")
                bot_messages = sum(1 for msg in chat_history if msg.get("actor") == "bot")

                return {
                    'session_id': str(session_id),
                    'message_count': len(chat_history),
                    'start_time': session_data.get("created_at"),
                    'last_activity': session_data.get("last_activity"),
                    'user_messages': user_messages,
                    'bot_messages': bot_messages,
                    'source': 'redis'
                }

            # Fallback to ScyllaDB
            history = self.conversation_history.get_session_history(session_id, 50)
            if not history:
                return {
                    'session_id': str(session_id),
                    'message_count': 0,
                    'start_time': None,
                    'last_activity': None,
                    'user_messages': 0,
                    'bot_messages': 0,
                    'source': 'scylladb_empty'
                }

            user_messages = sum(1 for msg in history if msg.actor == 'user')
            bot_messages = sum(1 for msg in history if msg.actor == 'bot')
            history.reverse()

            return {
                'session_id': str(session_id),
                'message_count': len(history),
                'start_time': history[0].timestamp,
                'last_activity': history[-1].timestamp,
                'user_messages': user_messages,
                'bot_messages': bot_messages,
                'source': 'scylladb'
            }

        except Exception as e:
            logger.error("Failed to get session summary: %s", str(e))
            return {'session_id': str(session_id), 'error': str(e)}

    def invalidate_cache(self, pattern: str = None) -> int:
        """Invalidate cached responses"""
        try:
            count = self.cache_model.invalidate_cache(pattern)
            self.analytics_model.record_event("cache_invalidated", {
                "pattern": pattern,
                "count": count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            logger.info("Invalidated %d cache entries with pattern: %s", count, pattern)
            return count
        except Exception as e:
            logger.error("Failed to invalidate cache: %s", str(e))
            return 0

    def _handle_dashboard_command_fallback(self, session_id: uuid.UUID) -> ChatResponse:
        """Handle dashboard command - works in both console and authenticated modes"""

        # For console application, always show Redis-only dashboard
        # In a web application, this would check for actual authentication

        summary = self.get_session_summary(session_id)

        # Get analytics data
        analytics_counters = {
            'sessions': self.analytics_model.redis.get("analytics:counter:sessions_created") or 0,
            'messages': self.analytics_model.redis.get("analytics:counter:messages_processed") or 0,
            'cache_hits': self.analytics_model.redis.get("analytics:counter:cache_hits") or 0,
            'cache_misses': self.analytics_model.redis.get("analytics:counter:cache_misses") or 0,
        }

        # Calculate cache hit rate
        total_cache_requests = int(analytics_counters['cache_hits']) + int(analytics_counters['cache_misses'])
        hit_rate = (int(analytics_counters['cache_hits']) / total_cache_requests * 100) if total_cache_requests > 0 else 0

        # Get session data
        session_data = self.session_model.get_session(str(session_id))
        chat_messages = len(session_data.get("chat_history", [])) if session_data else 0

        dashboard_message = f"""📊 **Personal Dashboard**

    👤 **Current Session:**
    - Session ID: {str(session_id)[:8]}...
    - Messages in session: {chat_messages}
    - Session started: {session_data.get('created_at', 'Unknown') if session_data else 'Unknown'}
    - Session source: {summary.get('source', 'redis')}

    📈 **Global Application Stats:**
    - Total sessions created: {analytics_counters['sessions']}
    - Total messages processed: {analytics_counters['messages']}
    - Cache hit rate: {hit_rate:.1f}% ({analytics_counters['cache_hits']}/{total_cache_requests})

    💾 **Performance Metrics:**
    - Cache hits: {analytics_counters['cache_hits']}
    - Cache misses: {analytics_counters['cache_misses']}
    - Redis acceleration: {'✅ Active' if hit_rate > 0 else '⏳ Building cache'}

    🎯 **Available Features:**
    - ✅ Smart knowledge base with caching
    - ✅ Background task processing
    - ✅ Real-time notifications
    - ✅ Session management
    - ✅ Analytics and feedback

    💡 **Commands You Can Use:**
    - `/stats` - Detailed statistics
    - `/research <topic>` - Background research
    - `/analyze <description>` - Data analysis
    - `/feedback <message>` - Send feedback
    - `/notifications` - Check notifications

    🚀 **System Status:** All systems operational in console mode"""

        if hasattr(self, 'multi_db_service') and self.multi_db_service:
            dashboard_message += f"\n\n🔐 **Authentication:** PostgreSQL available for web API features"
        else:
            dashboard_message += f"\n\n📊 **Mode:** Redis-powered console application"

        return ChatResponse(
            message=dashboard_message,
            has_context=True,
            session_info={'command_type': 'dashboard_console_success'}
        )

    def _handle_profile_command_fallback(self, session_id: uuid.UUID) -> ChatResponse:
        """Handle profile command - works in console mode"""

        session_data = self.session_model.get_session(str(session_id))
        summary = self.get_session_summary(session_id)

        # Calculate some session stats
        chat_history = session_data.get('chat_history', []) if session_data else []
        user_messages = sum(1 for msg in chat_history if msg.get('actor') == 'user')
        bot_messages = sum(1 for msg in chat_history if msg.get('actor') == 'bot')

        profile_message = f"""👤 **Session Profile**

    🔍 **Session Details:**
    - Session ID: {str(session_id)[:8]}...
    - User Type: Console User (Anonymous)
    - Created: {session_data.get('created_at', 'Unknown') if session_data else 'Unknown'}
    - Last Activity: {session_data.get('last_activity', 'Unknown') if session_data else 'Unknown'}

    📊 **Your Activity:**
    - Messages sent: {user_messages}
    - Responses received: {bot_messages}
    - Total interactions: {len(chat_history)}
    - Average response time: Fast (Redis cached)

    🎯 **Features You've Used:**
    - Knowledge base queries: ✅
    - Command interface: ✅
    - Session persistence: ✅
    - Background tasks: {'✅' if any('/research' in str(msg) or '/analyze' in str(msg) for msg in chat_history) else '❌'}

    💻 **Console Application Features:**
    - Smart Q&A with caching
    - Background task processing  
    - Real-time notifications
    - Session analytics
    - Feedback system

    💡 **Available Actions:**
    - Continue chatting with the knowledge base
    - Use `/research <topic>` for background research
    - Use `/analyze <task>` for data analysis
    - Use `/feedback <message>` to improve the system
    - Use `/stats` for detailed analytics

    🔐 **For Authentication Features:**
    - User accounts and profiles require web API access
    - Run the FastAPI server: `python app/api/main.py`
    - Access via web interface for full authentication"""

        return ChatResponse(
            message=profile_message,
            has_context=True,
            session_info={'command_type': 'profile_console_success'}
        )