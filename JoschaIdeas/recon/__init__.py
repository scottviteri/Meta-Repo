"""
Request Confirmation Networks (ReCoNs)

A Python implementation of neuro-symbolic script execution networks
based on Bach & Herger (2015).

ReCoNs are spreading activation networks with stateful units connected
by typed links that can perform both neural computations and controlled
script execution.
"""

from .core import (
    LinkType,
    ScriptState,
    TerminalState,
    MessageType,
    Message,
    Link,
    Unit,
    ScriptUnit,
    TerminalUnit,
    ReCoNNetwork,
)
from .engine import ReCoNEngine
from .scripts import ScriptBuilder
from .env import Environment, GridWorldEnvironment

__version__ = "0.1.0"
__author__ = "Based on Bach & Herger (2015)"

__all__ = [
    "LinkType",
    "ScriptState",
    "TerminalState",
    "MessageType",
    "Message",
    "Link",
    "Unit",
    "ScriptUnit",
    "TerminalUnit",
    "ReCoNNetwork",
    "ReCoNEngine",
    "ScriptBuilder",
    "Environment",
    "GridWorldEnvironment",
]
