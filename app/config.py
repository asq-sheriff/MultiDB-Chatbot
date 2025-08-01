# app/config.py
import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import secrets

load_dotenv()


@dataclass
class ScyllaConfig:
    """ScyllaDB configuration with cluster support"""
    hosts: list = None
    port: int = 9042
    keyspace: str = "chatbot_ks"
    datacenter: str = "datacenter1"

    # Cluster-specific settings
    connect_timeout: int = 10
    control_connection_timeout: int = 10
    protocol_version: int = 4

    # Connection retry settings
    max_retries: int = 3
    retry_delay: float = 2.0

    def __post_init__(self):
        if self.hosts is None:
            # Default to single primary node for simplicity
            self.hosts = ["127.0.0.1"]

    @classmethod
    def get_scylla_config(cls) -> dict:
        """Get ScyllaDB configuration optimized for cluster"""
        instance = cls()
        return {
            'hosts': instance.hosts,
            'port': instance.port,
            'keyspace': instance.keyspace,
            'datacenter': instance.datacenter,
            'connect_timeout': instance.connect_timeout,
            'control_connection_timeout': instance.control_connection_timeout,
            'protocol_version': instance.protocol_version,
            'max_retries': instance.max_retries,
            'retry_delay': instance.retry_delay
        }

    @classmethod
    def get_cluster_config(cls) -> dict:
        """Get configuration for multi-node cluster"""
        return {
            'hosts': ["127.0.0.1"],  # Primary node only for now
            'port': 9043,
            'keyspace': 'chatbot_ks',
            'datacenter': 'datacenter1',
            'connect_timeout': 15,  # Longer timeout for cluster
            'control_connection_timeout': 15,
            'protocol_version': 4
        }

    @classmethod
    def has_auth_credentials(cls) -> bool:
        """Check if authentication credentials are provided"""
        return False  # No auth for local development

    @classmethod
    def get_auth_credentials(cls) -> tuple:
        """Get authentication credentials"""
        return None, None

@dataclass
class RedisConfig:
    """Redis-specific configuration"""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: Optional[str] = os.getenv("REDIS_PASSWORD")
    max_connections: int = 20
    socket_timeout: int = 5
    socket_connect_timeout: int = 5

    # Cache settings
    default_cache_ttl: int = 3600  # 1 hour
    session_ttl: int = 86400       # 24 hours
    analytics_ttl: int = 604800    # 7 days

@dataclass
class PostgreSQLConfig:
    """PostgreSQL configuration for business logic"""
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    database: str = os.getenv("POSTGRES_DB", "chatbot_app")
    username: str = os.getenv("POSTGRES_USER", "chatbot_user")
    password: str = os.getenv("POSTGRES_PASSWORD", "secure_password")

    # Connection pool settings (compatible with SQLAlchemy 2.0.41 + asyncpg 0.30.0)
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600

    # Security settings
    secret_key: str = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  # 24 hours

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

@dataclass
class ApplicationConfig:
    """Main application configuration"""
    scylla: ScyllaConfig
    redis: RedisConfig
    postgresql: PostgreSQLConfig

    # Application settings
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_rate_limit: int = int(os.getenv("API_RATE_LIMIT", "100"))
    max_chat_history: int = 50

    # Feature flags - FIXED: Added enable_postgresql
    enable_caching: bool = True
    enable_analytics: bool = True
    enable_notifications: bool = True
    enable_postgresql: bool = os.getenv("ENABLE_POSTGRESQL", "true").lower() == "true"

    # Intelligent Processing Configuration
    enable_intelligent_routing: bool = True
    enable_timeout_processing: bool = True
    enable_auto_background: bool = True

    # Thresholds for automatic background processing
    auto_background_threshold_seconds: int = 8    # Tasks longer than this go to background
    timeout_check_interval_seconds: float = 1.0  # How often to check for timeouts
    min_confidence_for_auto_background: float = 0.6  # Confidence threshold for immediate background
    min_confidence_for_timeout: float = 0.4       # Confidence threshold for timeout processing
    min_confidence_for_suggestion: float = 0.3    # Confidence threshold for suggesting background

# Global config instance
config = ApplicationConfig(
    scylla=ScyllaConfig(),
    redis=RedisConfig(),
    postgresql=PostgreSQLConfig()
)