# app/__init__.py - UPDATE content
"""
Enhanced FAQ Chatbot with Redis Integration

This package provides a chatbot with Redis caching, session management,
and real-time analytics built on top of ScyllaDB persistence.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make key classes easily importable
from .services.chatbot_service import ChatbotService, ChatResponse
from .config import config

__all__ = [
    'ChatbotService',
    'ChatResponse',
    'config'
]