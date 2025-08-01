import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import logging

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.postgres_connection import postgres_manager
from app.database.postgres_models import User, Subscription, UsageRecord

logger = logging.getLogger(__name__)

class BillingService:
    """
    Billing and subscription management service.

    Used by: multi_db_service.py for quota checking, API endpoints for billing
    Integration: Called from multi_db_service._check_background_task_quota()
    """

    async def check_user_quota(self, user: User, resource_type: str) -> Dict[str, Any]:
        """
        Check if user has quota available for the resource.

        Called by: multi_db_service._check_background_task_quota()
        Location: Used in multi_db_service.py line 95-100

        Args:
            user: User object from PostgreSQL
            resource_type: Type of resource ("background_tasks", "messages", "api_calls")

        Returns:
            Dict with quota information and availability
        """
        try:
            async with postgres_manager.get_session() as session:
                # Get current billing period
                now = datetime.now(timezone.utc)
                period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)

                # Get current usage for this period
                usage_query = select(func.sum(UsageRecord.quantity)).where(
                    UsageRecord.user_id == user.id,
                    UsageRecord.resource_type == resource_type,
                    UsageRecord.billing_period_start >= period_start,
                    UsageRecord.billing_period_end <= period_end
                )

                result = await session.execute(usage_query)
                current_usage = result.scalar() or 0

                # Get subscription limits
                limits = self._get_plan_limits(user.subscription_plan)
                max_allowed = limits.get(resource_type, 0)

                return {
                    "has_quota": current_usage < max_allowed,
                    "current_usage": current_usage,
                    "max_allowed": max_allowed,
                    "remaining": max(0, max_allowed - current_usage),
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to check quota for user {user.id}: {e}")
            # Default to allowing the action if check fails
            return {
                "has_quota": True,
                "current_usage": 0,
                "max_allowed": 1000,
                "remaining": 1000,
                "error": str(e)
            }

    async def record_usage(self, user: User, resource_type: str, quantity: int = 1,
                           extra_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record resource usage for billing.

        Called by: multi_db_service._record_usage()
        Location: Used in multi_db_service.py line 125-135

        Args:
            user: User object
            resource_type: Type of resource used
            quantity: Amount of resource used
            extra_data: Additional usage metadata

        Returns:
            bool: Success status
        """
        try:
            async with postgres_manager.get_session() as session:
                now = datetime.now(timezone.utc)
                period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)

                usage_record = UsageRecord(
                    user_id=user.id,
                    resource_type=resource_type,
                    quantity=quantity,
                    billing_period_start=period_start,
                    billing_period_end=period_end,
                    extra_data=extra_data or {}
                )

                session.add(usage_record)
                await session.commit()

                logger.debug(f"Recorded usage: {quantity} {resource_type} for user {user.email}")
                return True

        except Exception as e:
            logger.error(f"Failed to record usage for user {user.id}: {e}")
            return False

    def _get_plan_limits(self, plan_type: str) -> Dict[str, int]:
        """
        Get resource limits for subscription plan.

        Used by: check_user_quota() method above
        Location: Called internally within this service

        Args:
            plan_type: Subscription plan ("free", "pro", "enterprise")

        Returns:
            Dict with resource limits
        """
        plan_limits = {
            "free": {
                "messages": 1000,
                "background_tasks": 10,
                "api_calls": 100
            },
            "pro": {
                "messages": 10000,
                "background_tasks": 100,
                "api_calls": 1000
            },
            "enterprise": {
                "messages": 100000,
                "background_tasks": 1000,
                "api_calls": 10000
            }
        }

        return plan_limits.get(plan_type, plan_limits["free"])

    async def get_usage_summary(self, user: User) -> Dict[str, Any]:
        """
        Get usage summary for user dashboard.

        Called by: multi_db_service._get_user_usage_stats()
        Location: Used in multi_db_service.py line 190-200

        Args:
            user: User object

        Returns:
            Dict with usage summary
        """
        try:
            async with postgres_manager.get_session() as session:
                now = datetime.now(timezone.utc)
                period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)

                # Get all usage for current period
                usage_query = select(
                    UsageRecord.resource_type,
                    func.sum(UsageRecord.quantity).label('total')
                ).where(
                    UsageRecord.user_id == user.id,
                    UsageRecord.billing_period_start >= period_start,
                    UsageRecord.billing_period_end <= period_end
                ).group_by(UsageRecord.resource_type)

                result = await session.execute(usage_query)
                usage_data = {row.resource_type: row.total for row in result}

                # Get plan limits
                limits = self._get_plan_limits(user.subscription_plan)

                return {
                    "messages_this_month": usage_data.get("messages", 0),
                    "background_tasks_this_month": usage_data.get("background_tasks", 0),
                    "api_calls_this_month": usage_data.get("api_calls", 0),
                    "quota_remaining": limits.get("messages", 0) - usage_data.get("messages", 0),
                    "limits": limits,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat()
                }

        except Exception as e:
            logger.error(f"Failed to get usage summary for user {user.id}: {e}")
            return {
                "messages_this_month": 0,
                "background_tasks_this_month": 0,
                "api_calls_this_month": 0,
                "quota_remaining": 1000,
                "error": str(e)
            }

# Global billing service instance
billing_service = BillingService()