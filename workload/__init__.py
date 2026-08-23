"""Workload implementations."""

from .RAGWorkload import RAGInput, RAGWorkload
from .MultiAgentFullConnectionWorkload import (
    AgentSpec, MultiAgentFullConnectionInput, MultiAgentFullConnectionWorkload,
)

__all__ = [
    "RAGInput",
    "RAGWorkload",
    "AgentSpec",
    "MultiAgentFullConnectionInput",
    "MultiAgentFullConnectionWorkload",
]
