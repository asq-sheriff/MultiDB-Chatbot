import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import logging

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.postgres_connection import postgres_manager
from app.database.postgres_models import User, Organization, AuditLog
from app.services.auth_service import auth_service

logger = logging.getLogger(__name__)

class UserService:
    """
    User management and profile service.

    Used by: API endpoints in app/api/endpoints/users.py, chatbot_service.py
    Integration: Works with auth_service.py and provides user management
    """

    async def update_user_profile(self, user_id: uuid.UUID,
                                  profile_data: Dict[str, Any]) -> Optional[User]:
        """
        Update user profile information.

        Called by: API endpoints for profile updates
        Location: Will be used in app/api/endpoints/users.py

        Args:
            user_id: User UUID
            profile_data: Dict with profile updates

        Returns:
            Updated User object or None if failed
        """
        try:
            async with postgres_manager.get_session() as session:
                # Get current user
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()

                if not user:
                    return None

                # Store old values for audit
                old_values = {
                    "email": user.email,
                    "preferences": user.preferences,
                    "subscription_plan": user.subscription_plan
                }

                # Update allowed fields
                updatable_fields = ["preferences"]
                update_data = {}

                for field in updatable_fields:
                    if field in profile_data:
                        update_data[field] = profile_data[field]
                        setattr(user, field, profile_data[field])

                if update_data:
                    await session.commit()
                    await session.refresh(user)

                    # Log audit trail
                    await self._log_audit(
                        session, user_id, "profile_updated", "user",
                        resource_id=str(user_id),
                        old_values=old_values,
                        new_values=update_data
                    )

                    logger.info(f"Updated profile for user {user.email}")

                return user

        except Exception as e:
            logger.error(f"Failed to update user profile {user_id}: {e}")
            return None

    async def change_subscription_plan(self, user_id: uuid.UUID,
                                       new_plan: str) -> bool:
        """
        Change user subscription plan.

        Called by: billing_service.py, API endpoints
        Location: Used in billing flows and admin operations

        Args:
            user_id: User UUID
            new_plan: New plan type ("free", "pro", "enterprise")

        Returns:
            bool: Success status
        """
        try:
            valid_plans = ["free", "pro", "enterprise"]
            if new_plan not in valid_plans:
                raise ValueError(f"Invalid plan: {new_plan}")

            async with postgres_manager.get_session() as session:
                # Update subscription plan
                result = await session.execute(
                    update(User)
                    .where(User.id == user_id)
                    .values(subscription_plan=new_plan)
                    .returning(User)
                )

                updated_user = result.scalar_one_or_none()
                if not updated_user:
                    return False

                # Log audit trail
                await self._log_audit(
                    session, user_id, "subscription_changed", "user",
                    resource_id=str(user_id),
                    new_values={"subscription_plan": new_plan}
                )

                logger.info(f"Changed subscription for user {updated_user.email} to {new_plan}")
                return True

        except Exception as e:
            logger.error(f"Failed to change subscription for user {user_id}: {e}")
            return False

    async def deactivate_user(self, user_id: uuid.UUID,
                              deactivated_by: Optional[uuid.UUID] = None) -> bool:
        """
        Deactivate user account.

        Called by: Admin operations, user self-deletion
        Location: Used in admin panels and account deletion flows

        Args:
            user_id: User UUID to deactivate
            deactivated_by: Admin user ID (if admin action)

        Returns:
            bool: Success status
        """
        try:
            async with postgres_manager.get_session() as session:
                # Update user status
                result = await session.execute(
                    update(User)
                    .where(User.id == user_id)
                    .values(is_active=False)
                    .returning(User)
                )

                deactivated_user = result.scalar_one_or_none()
                if not deactivated_user:
                    return False

                # Log audit trail
                await self._log_audit(
                    session, deactivated_by or user_id, "user_deactivated", "user",
                    resource_id=str(user_id),
                    new_values={"is_active": False, "deactivated_by": str(deactivated_by) if deactivated_by else "self"}
                )

                logger.info(f"Deactivated user {deactivated_user.email}")
                return True

        except Exception as e:
            logger.error(f"Failed to deactivate user {user_id}: {e}")
            return False

    async def get_user_statistics(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get comprehensive user statistics.

        Called by: Dashboard endpoints, analytics
        Location: Used in user dashboard and admin panels

        Args:
            user_id: User UUID

        Returns:
            Dict with user statistics
        """
        try:
            async with postgres_manager.get_session() as session:
                # Get user basic info
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalar_one_or_none()

                if not user:
                    return {}

                # Get audit activity count
                audit_count_result = await session.execute(
                    select(func.count(AuditLog.id)).where(AuditLog.user_id == user_id)
                )
                audit_count = audit_count_result.scalar() or 0

                return {
                    "user_id": str(user_id),
                    "email": user.email,
                    "subscription_plan": user.subscription_plan,
                    "is_active": user.is_active,
                    "is_verified": user.is_verified,
                    "created_at": user.created_at.isoformat(),
                    "total_audit_events": audit_count,
                    "preferences": user.preferences or {}
                }

        except Exception as e:
            logger.error(f"Failed to get user statistics {user_id}: {e}")
            return {"error": str(e)}

    async def search_users(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search users by email or other criteria.

        Called by: Admin panels, user management interfaces
        Location: Used in admin operations

        Args:
            query: Search query (email pattern)
            limit: Maximum results to return

        Returns:
            List of user summaries
        """
        try:
            async with postgres_manager.get_session() as session:
                # Search by email pattern
                result = await session.execute(
                    select(User)
                    .where(User.email.ilike(f"%{query}%"))
                    .limit(limit)
                )

                users = result.scalars().all()

                return [
                    {
                        "id": str(user.id),
                        "email": user.email,
                        "subscription_plan": user.subscription_plan,
                        "is_active": user.is_active,
                        "created_at": user.created_at.isoformat()
                    }
                    for user in users
                ]

        except Exception as e:
            logger.error(f"Failed to search users with query '{query}': {e}")
            return []

    async def _log_audit(self, session: AsyncSession, user_id: Optional[uuid.UUID],
                         action: str, resource_type: str, resource_id: Optional[str] = None,
                         old_values: Optional[dict] = None, new_values: Optional[dict] = None):
        """
        Internal method to log audit events.

        Used by: All methods in this service that modify data
        Location: Called internally within user_service methods

        Args:
            session: Database session
            user_id: User performing the action
            action: Action description
            resource_type: Type of resource modified
            resource_id: ID of the resource
            old_values: Previous values
            new_values: New values
        """
        audit_log = AuditLog(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            old_values=old_values,
            new_values=new_values
        )
        session.add(audit_log)

# Global user service instance
user_service = UserService()