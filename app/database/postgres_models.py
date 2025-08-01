import uuid
from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import (
    Column, String, Boolean, Integer, DateTime, Text,
    ForeignKey, UUID, ARRAY, JSON, Numeric, Index
)
from sqlalchemy.orm import relationship, Mapped, mapped_column, DeclarativeBase
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID, JSONB

class DatabaseBase(DeclarativeBase):
    """Base class for all PostgreSQL models"""
    pass

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

class Organization(DatabaseBase, TimestampMixin):
    """
    Organization management for multi-tenant support.

    Used by: user_service.py for enterprise features
    Integration: Users belong to organizations
    """
    __tablename__ = "organizations"

    id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[Optional[str]] = mapped_column(String(255), unique=True)
    settings: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Relationships
    users: Mapped[List["User"]] = relationship("User", back_populates="organization")

class User(DatabaseBase, TimestampMixin):
    """
    User management and authentication.

    Used by: auth_service.py, user_service.py, chatbot_service.py
    Integration: Links to sessions via user_id, billing via subscriptions
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)

    # User status and preferences
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    subscription_plan: Mapped[str] = mapped_column(String(50), default="free")

    # Organization (for future multi-tenant support)
    organization_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("organizations.id"),
        nullable=True
    )

    # User preferences stored as JSONB
    preferences: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Relationships - FIXED: Added missing organization relationship
    organization: Mapped[Optional["Organization"]] = relationship(
        "Organization",
        back_populates="users"
    )
    subscriptions: Mapped[List["Subscription"]] = relationship(
        "Subscription",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    usage_records: Mapped[List["UsageRecord"]] = relationship(
        "UsageRecord",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    audit_logs: Mapped[List["AuditLog"]] = relationship(
        "AuditLog",
        back_populates="user"
    )

class Subscription(DatabaseBase, TimestampMixin):
    """
    User subscription and billing information.

    Used by: billing_service.py for subscription management
    Integration: Links users to their billing plans and limits
    """
    __tablename__ = "subscriptions"

    id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )

    # Subscription details
    plan_type: Mapped[str] = mapped_column(String(50), nullable=False)  # free, pro, enterprise
    status: Mapped[str] = mapped_column(String(20), nullable=False)     # active, cancelled, expired
    billing_cycle: Mapped[str] = mapped_column(String(20), default="monthly")  # monthly, yearly

    # Pricing
    amount_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    # Subscription period
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=func.now)
    ends_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    auto_renew: Mapped[bool] = mapped_column(Boolean, default=True)

    # Usage limits (stored as JSONB for flexibility)
    limits: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="subscriptions")

class UsageRecord(DatabaseBase, TimestampMixin):
    """
    Track resource usage for billing and quotas.

    Used by: billing_service.py, chatbot_service.py for quota enforcement
    Integration: Tracks user activity from Redis/ScyllaDB for billing
    """
    __tablename__ = "usage_records"

    id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id"),
        nullable=False
    )

    # Usage details
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)  # messages, background_tasks, api_calls
    quantity: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    # Billing period for aggregation
    billing_period_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    billing_period_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Additional metadata
    extra_data: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="usage_records")

    # Indexes for efficient billing queries
    __table_args__ = (
        Index('idx_usage_user_period', 'user_id', 'billing_period_start', 'billing_period_end'),
        Index('idx_usage_resource_type', 'resource_type'),
    )

class AuditLog(DatabaseBase, TimestampMixin):
    """
    Audit trail for compliance and security monitoring.

    Used by: All services for audit logging, compliance reporting
    Integration: Tracks all significant user actions across the system
    """
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id")
    )

    # Action details
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    resource_id: Mapped[Optional[str]] = mapped_column(String(255))

    # Change tracking
    old_values: Mapped[Optional[dict]] = mapped_column(JSONB)
    new_values: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Request context
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # Support IPv6
    user_agent: Mapped[Optional[str]] = mapped_column(Text)

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="audit_logs")

    # Indexes for efficient audit queries
    __table_args__ = (
        Index('idx_audit_user_action', 'user_id', 'action'),
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_timestamp', 'created_at'),
    )

class FeatureFlag(DatabaseBase, TimestampMixin):
    """
    Feature flags for gradual rollouts and A/B testing.

    Used by: All services to check feature availability
    Integration: Controls which features are available to which users
    """
    __tablename__ = "feature_flags"

    id: Mapped[uuid.UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4
    )

    # Flag details
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)

    # Rollout configuration
    rollout_percentage: Mapped[int] = mapped_column(Integer, default=0)  # 0-100
    target_user_segments: Mapped[Optional[List[str]]] = mapped_column(ARRAY(String))

    # Conditions for enabling (JSONB for flexibility)
    conditions: Mapped[Optional[dict]] = mapped_column(JSONB, default=dict)

class SystemSetting(DatabaseBase, TimestampMixin):
    """
    System-wide configuration and settings.

    Used by: Configuration management, system administration
    Integration: Stores dynamic configuration that can be changed without redeployment
    """
    __tablename__ = "system_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONB, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)

    # Change tracking
    updated_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("users.id")
    )