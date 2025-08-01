# app/api/dependencies.py

from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.postgres_connection import postgres_manager
from app.database.redis_connection import get_redis
from app.services.auth_service import auth_service
from app.database.postgres_models import User

# Security scheme for JWT tokens
security = HTTPBearer()

# FIXED: Better dependency injection for ChatbotService
def get_chatbot_service():
    """
    Dependency to get the ChatbotService instance.

    Integration: Creates a new instance or reuses existing one
    Usage: chatbot_service = Depends(get_chatbot_service)

    Location: Use this in any endpoint that needs chatbot functionality
    """
    try:
        # Import here to avoid circular imports
        from app.services.chatbot_service import ChatbotService

        # Create a new service instance for this request
        # This is safe because ChatbotService is stateless for individual requests
        service = ChatbotService()

        # Ensure knowledge base is seeded
        if not hasattr(service.knowledge_service, '_seeded'):
            service.knowledge_service.seed_knowledge_base()
            service.knowledge_service._seeded = True

        return service

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ChatbotService not available: {str(e)}"
        )

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for PostgreSQL database session.

    Used by: All API endpoints that need database access
    Integration: Provides database session to endpoint functions
    """
    async with postgres_manager.get_session() as session:
        yield session

def get_redis_client():
    """
    Dependency for Redis client.

    Used by: API endpoints that need Redis access
    Integration: Provides Redis client to endpoint functions
    """
    return get_redis()

async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        session: AsyncSession = Depends(get_db_session)
) -> User:
    """
    Dependency to get current authenticated user.

    Used by: Protected API endpoints
    Integration: Validates JWT token and returns User object

    Args:
        credentials: JWT token from Authorization header
        session: Database session

    Returns:
        User: Current authenticated user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    try:
        # Verify JWT token
        token_payload = await auth_service.verify_token(credentials.credentials)
        if not token_payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )

        # Get user from database
        user_id = token_payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )

        user = await auth_service.get_user_by_id(user_id)
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )

        return user

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def get_current_active_user(
        current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to ensure user is active.

    Used by: API endpoints that require active users
    Integration: Additional check on top of get_current_user
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    return current_user

def get_optional_current_user(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[str]:
    """
    Dependency for optional authentication.

    Used by: Endpoints that work with or without authentication
    Integration: Returns token if provided, None otherwise

    Returns:
        Optional[str]: JWT token if provided, None otherwise
    """
    if credentials:
        return credentials.credentials
    return None