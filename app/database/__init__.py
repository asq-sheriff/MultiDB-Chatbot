# app/database/__init__.py - Fixed for Redis 6.2.0
"""Database package with ScyllaDB models and Redis integration."""

# ScyllaDB imports (always available)
from .scylla_models import (
    ConversationHistory,
    ConversationMessage,
    UserFeedbackRepository,
    KnowledgeBase,
    KnowledgeEntry
)

# Redis imports with error handling
try:
    from .redis_connection import redis_manager, get_redis
    REDIS_AVAILABLE = True

    # Try async redis import
    try:
        from .redis_connection import get_async_redis
    except ImportError:
        get_async_redis = None

    from .redis_models import CacheModel, SessionModel, AnalyticsModel, NotificationModel
except ImportError as e:
    print(f"⚠️ Redis imports failed: {e}")
    REDIS_AVAILABLE = False
    redis_manager = None
    get_redis = None
    get_async_redis = None
    CacheModel = None
    SessionModel = None
    AnalyticsModel = None
    NotificationModel = None

# PostgreSQL imports with error handling
try:
    from .postgres_connection import postgres_manager, get_postgres_session, DatabaseBase
    from .postgres_models import (
        User, Organization, Subscription, UsageRecord,
        AuditLog, FeatureFlag, SystemSetting
    )
    POSTGRES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PostgreSQL imports failed: {e}")
    POSTGRES_AVAILABLE = False
    postgres_manager = None
    get_postgres_session = None
    DatabaseBase = None
    User = None
    Organization = None
    Subscription = None
    UsageRecord = None
    AuditLog = None
    FeatureFlag = None
    SystemSetting = None

__all__ = [
    # ScyllaDB (always available)
    'ConversationHistory',
    'ConversationMessage',
    'UserFeedbackRepository',
    'KnowledgeBase',
    'KnowledgeEntry',

    # Redis (conditional)
    'redis_manager',
    'get_redis',
    'get_async_redis',
    'CacheModel',
    'SessionModel',
    'AnalyticsModel',
    'NotificationModel',

    # PostgreSQL (conditional)
    'postgres_manager',
    'get_postgres_session',
    'DatabaseBase',
    'User',
    'Organization',
    'Subscription',
    'UsageRecord',
    'AuditLog',
    'FeatureFlag',
    'SystemSetting'
]