# app/api/main.py

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from app.config import config
from app.database.redis_connection import redis_manager
from app.database.postgres_connection import postgres_manager
from app.api.endpoints import auth
from app.api.dependencies import get_current_user, get_optional_current_user, get_chatbot_service
from app.database.postgres_models import User

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for API startup/shutdown"""

    logger.info("🚀 Starting FastAPI application...")

    # Track which services are initialized
    services_status = {
        'redis': False,
        'postgresql': False,
        'scylladb': False
    }

    try:
        # 1. Initialize Redis (Critical for API)
        try:
            redis_manager.initialize()
            if redis_manager.test_connection():
                services_status['redis'] = True
                logger.info("✅ Redis connected and ready")
            else:
                raise ConnectionError("Redis connection test failed")
        except Exception as e:
            logger.error(f"❌ Redis initialization failed: {e}")
            raise

        # 2. Initialize PostgreSQL (Critical for API auth)
        try:
            await postgres_manager.initialize()
            if await postgres_manager.test_connection():
                services_status['postgresql'] = True
                logger.info("✅ PostgreSQL connected and ready")
            else:
                raise ConnectionError("PostgreSQL connection test failed")
        except Exception as e:
            logger.error(f"❌ PostgreSQL initialization failed: {e}")
            raise

        # 3. Initialize ScyllaDB (Optional for API - can use mock)
        try:
            import socket

            # First check if ScyllaDB is even listening
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            scylla_available = sock.connect_ex(('127.0.0.1', 9042)) == 0
            sock.close()

            if scylla_available:
                logger.info("🔍 ScyllaDB detected on port 9042, attempting connection...")

                from app.database.scylla_connection import ScyllaDBConnection

                # Reset to ensure clean state for API
                ScyllaDBConnection.reset_singleton()

                # Small delay to let reset complete
                import time
                time.sleep(1)

                scylla_conn = ScyllaDBConnection()
                scylla_conn.connect(force_reconnect=True)

                if scylla_conn.is_connected():
                    scylla_conn.ensure_keyspace("chatbot_ks")
                    services_status['scylladb'] = True
                    logger.info("✅ ScyllaDB cluster connected")
                else:
                    logger.warning("⚠️ ScyllaDB connection failed, using mock mode")
            else:
                logger.info("ℹ️ ScyllaDB not detected on port 9042, using mock mode")

        except Exception as e:
            logger.warning(f"⚠️ ScyllaDB initialization error: {e}")
            logger.info("🔄 API will use mock ScyllaDB implementation")
            # Don't let ScyllaDB issues stop the API from starting

        # Report status
        active_dbs = sum(services_status.values())
        logger.info(f"🎉 API ready with {active_dbs}/3 databases active")

    except Exception as e:
        logger.error(f"❌ Critical service initialization failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down FastAPI application...")

    try:
        # Close Redis
        redis_manager.close()
        logger.info("✅ Redis closed")

        # Close PostgreSQL
        await postgres_manager.close()
        logger.info("✅ PostgreSQL closed")

        # Close ScyllaDB (if connected)
        if services_status.get('scylladb'):
            try:
                from app.database.scylla_connection import ScyllaDBConnection
                conn = ScyllaDBConnection()
                conn.disconnect()
                logger.info("✅ ScyllaDB closed")
            except:
                pass

        logger.info("✅ API shutdown complete")

    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="MultiDB Chatbot API",
    description="Enhanced chatbot with Redis, PostgreSQL, and ScyllaDB integration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(auth.router, prefix="/api/v1")

# Chat endpoint for sending messages
@app.post("/api/v1/chat/message")
async def send_message(
        message_data: dict,
        user_token: str = Depends(get_optional_current_user),
        chatbot_service = Depends(get_chatbot_service)
):
    """
    Send message to chatbot with comprehensive error handling.

    Integration: Uses dependency injection for chatbot service
    Location: Main chat endpoint for API
    """
    try:
        from uuid import uuid4

        # Validate input
        message = message_data.get('message', '').strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        if len(message) > 1000:  # Reasonable limit
            raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")

        # Get or create session ID
        session_id = message_data.get('session_id', str(uuid4()))

        # Process message based on authentication status
        if user_token:
            # Authenticated user processing
            response = await chatbot_service.process_authenticated_message(
                session_id=session_id,
                user_message=message,
                user_token=user_token
            )
        else:
            # Anonymous user processing
            response = chatbot_service.process_message(
                session_id=session_id,
                user_message=message
            )

        return {
            "session_id": session_id,
            "message": response.message,
            "confidence": response.confidence,
            "cached": getattr(response, 'cached', False),
            "response_time_ms": getattr(response, 'response_time_ms', None),
            "has_context": response.has_context,
            "session_info": response.session_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

# Notifications endpoint
@app.get("/api/v1/chat/notifications")
async def get_notifications(
        current_user: User = Depends(get_current_user)
):
    """
    Get user notifications with proper error handling.

    Integration: Uses Redis notification system
    Location: Notifications endpoint for authenticated users
    """
    try:
        from app.database.redis_models import NotificationModel

        notification_model = NotificationModel()
        user_id = f"user_{str(current_user.id)[:8]}"

        notifications = notification_model.get_notifications(user_id, count=10)
        count = notification_model.count_notifications(user_id)

        return {
            "notifications": notifications,
            "total_count": count,
            "user_id": user_id
        }

    except Exception as e:
        logger.error(f"Error getting notifications for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to get notifications")

# Background task endpoint
@app.post("/api/v1/chat/background-task")
async def start_background_task(
        task_data: dict,
        current_user: User = Depends(get_current_user),
        chatbot_service = Depends(get_chatbot_service)
):
    """
    Start authenticated background task with validation.

    Integration: Uses multi_db_service for quota checking
    Location: Background task endpoint for authenticated users
    """
    try:
        # Validate task data
        task_type = task_data.get("task_type", "analysis")
        if task_type not in ["analysis", "research"]:
            raise HTTPException(status_code=400, detail="Invalid task_type. Must be 'analysis' or 'research'")

        # Validate required fields based on task type
        if task_type == "analysis":
            if not task_data.get("description"):
                raise HTTPException(status_code=400, detail="Description required for analysis tasks")
        elif task_type == "research":
            if not task_data.get("topic"):
                raise HTTPException(status_code=400, detail="Topic required for research tasks")

        # Create token for the user
        from app.services.auth_service import auth_service
        token_data = {"user_id": str(current_user.id)}
        user_token = auth_service.create_access_token(token_data)

        # Start the background task
        task_id = await chatbot_service.start_authenticated_background_task(
            task_type=task_type,
            task_data=task_data,
            user_token=user_token
        )

        return {
            "task_id": task_id,
            "status": "started",
            "task_type": task_type,
            "user_id": str(current_user.id)
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting background task for user {current_user.id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start background task: {str(e)}")

# Root endpoint with database status
@app.get("/")
async def root():
    """Enhanced root endpoint with database status checking"""
    # Check database connections
    redis_status = "❌ Disconnected"
    postgres_status = "❌ Disconnected"

    try:
        if redis_manager.test_connection():
            redis_status = "✅ Connected"
    except:
        pass

    try:
        if await postgres_manager.test_connection():
            postgres_status = "✅ Connected"
    except:
        pass

    return {
        "message": "🚀 MultiDB Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "redis": redis_status,
            "postgresql": postgres_status,
            "scylladb": "⚡ Available (when enabled)"
        },
        "endpoints": {
            "authentication": {
                "register": "POST /api/v1/auth/register",
                "login": "POST /api/v1/auth/login",
                "profile": "GET /api/v1/auth/me",
                "dashboard": "GET /api/v1/auth/dashboard"
            },
            "chat": {
                "send_message": "POST /api/v1/chat/message",
                "notifications": "GET /api/v1/chat/notifications",
                "background_task": "POST /api/v1/chat/background-task"
            }
        }
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    status = {}

    # Check Redis
    try:
        status["redis"] = "healthy" if redis_manager.test_connection() else "unhealthy"
    except:
        status["redis"] = "unhealthy"

    # Check PostgreSQL
    try:
        status["postgresql"] = "healthy" if await postgres_manager.test_connection() else "unhealthy"
    except:
        status["postgresql"] = "unhealthy"

    # Check if we can create a chatbot service
    try:
        from app.services.chatbot_service import ChatbotService
        test_service = ChatbotService()
        status["chatbot"] = "healthy"
    except:
        status["chatbot"] = "unhealthy"

    overall_status = "healthy" if all(s == "healthy" for s in status.values()) else "degraded"

    return {
        "status": overall_status,
        "services": status,
        "timestamp": datetime.now().isoformat()
    }