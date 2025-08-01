"""
ScyllaDB connection management module optimized for learning.
This module uses ScyllaDB-specific features for the best learning experience.
Used by: models.py, services/, and main.py
"""

import os
import sys
import logging
from typing import Optional, List
import traceback

# FORCE pure Python mode for Python 3.12+ BEFORE any cassandra imports
if sys.version_info >= (3, 12):
    os.environ['CASS_DRIVER_NO_EXTENSIONS'] = '1'
    os.environ['CASS_DRIVER_NO_CYTHON'] = '1'
    os.environ['CASS_DRIVER_NO_MURMUR3'] = '1'
    os.environ['CASS_DRIVER_EVENT_LOOP_IMPL'] = 'asyncio'

logger = logging.getLogger(__name__)

# Suppress noisy cluster reconnection warnings for development
logging.getLogger('cassandra.cluster').setLevel(logging.ERROR)
logging.getLogger('cassandra.pool').setLevel(logging.ERROR)

# Global variable to track which connection implementation to use
_use_fallback = False

def _try_import_driver():
    """Try to import the ScyllaDB driver with compatibility settings."""
    global _use_fallback

    try:
        # Import ScyllaDB driver components
        from cassandra.cluster import Cluster, Session
        from cassandra.auth import PlainTextAuthProvider
        from cassandra.policies import DCAwareRoundRobinPolicy

        # Check if we're using ScyllaDB driver specifically
        python_version = sys.version_info
        if python_version >= (3, 12):
            _use_fallback = True
            logger.info("Using scylla-driver in pure Python mode (Python 3.12+)")
        else:
            logger.info("Using scylla-driver with optimizations")

        return True, (Cluster, Session, PlainTextAuthProvider, DCAwareRoundRobinPolicy)

    except Exception as e:
        logger.error("Failed to load ScyllaDB driver: %s", str(e))
        logger.error("Please ensure scylla-driver is installed: pip install scylla-driver")
        raise ImportError(f"Could not load ScyllaDB driver: {e}")

# Try to import the driver at module load time
try:
    _import_success, (Cluster, Session, PlainTextAuthProvider, DCAwareRoundRobinPolicy) = _try_import_driver()
except ImportError as e:
    logger.error("ScyllaDB driver import failed: %s", str(e))
    raise

from app.config import ScyllaConfig


class ScyllaDBConnection:
    """
    Singleton class for managing ScyllaDB connections.
    This ensures we maintain a single connection pool throughout the application.
    """

    _instance: Optional['ScyllaDBConnection'] = None
    _cluster: Optional[Cluster] = None
    _session: Optional[Session] = None

    def __new__(cls) -> 'ScyllaDBConnection':
        """Ensure only one instance exists (Singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def connect(self) -> None:
        """
        Establish connection to ScyllaDB cluster.
        Called by: main.py during application startup
        """
        if self._cluster is not None:
            logger.info("Already connected to ScyllaDB")
            return

        try:
            config = ScyllaConfig.get_scylla_config()

            # Setup load balancing policy
            load_balancing_policy = DCAwareRoundRobinPolicy(local_dc=config['datacenter'])

            # Create cluster configuration with multi-port support
            contact_points = config['hosts'] if isinstance(config['hosts'], list) else [config['hosts']]

            cluster_kwargs = {
                'contact_points': contact_points,
                'port': config['port'],  # Default port, driver will discover actual ports
                'load_balancing_policy': load_balancing_policy,
                'protocol_version': 4,
                'control_connection_timeout': 10,  # Faster control connection
                'connect_timeout': 10,             # Faster individual connections
            }

            request_timeout_val = 30
            if not (_use_fallback or sys.version_info >= (3, 12)):
                cluster_kwargs['compression'] = True
                request_timeout_val = 12
                logger.info("Using ScyllaDB driver with full optimizations")
            else:
                logger.info("Using ScyllaDB driver in pure Python mode")

            if ScyllaConfig.has_auth_credentials():
                username, password = ScyllaConfig.get_auth_credentials()
                cluster_kwargs['auth_provider'] = PlainTextAuthProvider(
                    username=username,
                    password=password
                )

            self._cluster = Cluster(**cluster_kwargs)

            # Connect WITHOUT specifying a keyspace initially
            self._session = self._cluster.connect()
            self._session.default_timeout = request_timeout_val

            logger.info("Connected to ScyllaDB cluster: %s", config['hosts'])

        except Exception as e:
            error_msg = f"Failed to connect to ScyllaDB: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error("Config attempted: hosts=%s, port=%s", config['hosts'], config['port'])
            logger.debug("Full traceback: %s", traceback.format_exc())
            raise ConnectionError(f"Cannot connect to ScyllaDB cluster: {str(e)}") from e

    def ensure_keyspace(self, keyspace: str) -> None:
        """
        Ensure we're connected to the correct keyspace.
        Called by: models.py before executing queries

        Args:
            keyspace (str): Keyspace name to use
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to ScyllaDB")

        try:
            # Simply set the keyspace without checking existence
            self.set_keyspace(keyspace)
            logger.debug("Switched to keyspace: %s", keyspace)

        except Exception as e:
            logger.error("Failed to set keyspace '%s': %s", keyspace, str(e))
            raise

    def get_session(self) -> Session:
        """
        Get the current database session.
        Called by: models.py and services/ modules

        Returns:
            Session: Active ScyllaDB session

        Raises:
            RuntimeError: If not connected to database
        """
        if self._session is None:
            raise RuntimeError("Not connected to ScyllaDB. Call connect() first.")
        return self._session

    def disconnect(self) -> None:
        """
        Close connection to ScyllaDB cluster.
        Called by: main.py during application shutdown
        """
        if self._cluster is not None:
            self._cluster.shutdown()
            self._cluster = None
            self._session = None
            logger.info("Disconnected from ScyllaDB")

    def is_connected(self) -> bool:
        """
        Check if connected to ScyllaDB.
        Called by: Health check utilities and services

        Returns:
            bool: True if connected, False otherwise
        """
        return self._session is not None and not self._session.is_shutdown

    def set_keyspace(self, keyspace: str) -> None:
        """
        Set the session to a specific keyspace.
        Called by: main.py after ensuring keyspace exists.
        """
        if self._session:
            try:
                self._session.set_keyspace(keyspace)
                logger.info("Switched to keyspace: %s", keyspace)
            except Exception as e:
                logger.error("Failed to set keyspace to '%s': %s", keyspace, str(e))
                raise

    def execute_cql_file(self, file_path: str) -> None:
        """
        Execute CQL commands from a file.
        Called by: Setup scripts and migrations

        Args:
            file_path (str): Path to the CQL file
        """
        if not self.is_connected():
            raise RuntimeError("Not connected to ScyllaDB")

        try:
            with open(file_path, 'r') as file:
                cql_content = file.read()

            # Split by semicolon and execute each statement
            statements = [stmt.strip() for stmt in cql_content.split(';') if stmt.strip()]

            for statement in statements:
                logger.debug("Executing CQL: %s...", statement[:100])
                self._session.execute(statement)

            logger.info("Successfully executed CQL file: %s", file_path)

        except Exception as e:
            logger.error("Failed to execute CQL file %s: %s", file_path, str(e))
            raise