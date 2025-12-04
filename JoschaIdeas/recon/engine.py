"""
Simulation engine for Request Confirmation Networks.

This module implements the synchronous stepping loop and message routing
that drives ReCoN execution.
"""

from typing import Dict, List, Set, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

from .core import (
    ReCoNNetwork,
    Unit,
    ScriptUnit,
    TerminalUnit,
    Link,
    LinkType,
    ScriptState,
    TerminalState,
    Message,
    MessageType,
)


class ExecutionStatus(Enum):
    """Status of script execution."""
    RUNNING = auto()
    CONFIRMED = auto()
    FAILED = auto()
    IDLE = auto()


@dataclass
class StepResult:
    """Result of a single simulation step."""
    step_number: int
    status: ExecutionStatus
    active_units: List[str]
    confirmed_units: List[str]
    failed_units: List[str]
    messages_passed: int


class ReCoNEngine:
    """
    Simulation engine for executing ReCoN networks.

    The engine handles:
    - Synchronous stepping of all units
    - Message propagation along links
    - Tracking execution status
    - Numerical activation propagation (optional)
    """

    def __init__(self, network: ReCoNNetwork,
                 use_numerical_activation: bool = False):
        """
        Initialize the engine.

        Args:
            network: The ReCoN network to simulate
            use_numerical_activation: Whether to use numerical spreading activation
                                     in addition to discrete messages
        """
        self.network = network
        self.use_numerical_activation = use_numerical_activation
        self.step_count = 0
        self._active_requests: Set[str] = set()
        self._execution_history: List[StepResult] = []

        # Numerical activation matrices (if enabled)
        if use_numerical_activation:
            self._init_activation_matrices()

        # Callbacks for monitoring
        self._step_callbacks: List[Callable[[StepResult], None]] = []

    def _init_activation_matrices(self):
        """Initialize weight matrices for numerical spreading activation."""
        n = len(self.network.units)
        unit_ids = list(self.network.units.keys())
        self._unit_index = {uid: i for i, uid in enumerate(unit_ids)}

        # Create sparse weight matrices for each link type
        self.W: Dict[LinkType, np.ndarray] = {}
        for lt in LinkType:
            self.W[lt] = np.zeros((n, n), dtype=float)

        for link in self.network.links:
            i = self._unit_index[link.source_id]
            j = self._unit_index[link.target_id]
            # Use mean of weight vector as scalar weight
            w = np.mean(link.weight) if len(link.weight) > 0 else 1.0
            self.W[link.link_type][i, j] = w

        # Activation vectors per link type
        self.a: Dict[LinkType, np.ndarray] = {}
        for lt in LinkType:
            self.a[lt] = np.zeros(n, dtype=float)

    def add_step_callback(self, callback: Callable[[StepResult], None]):
        """Add a callback to be called after each step."""
        self._step_callbacks.append(callback)

    def request(self, root_id: str) -> bool:
        """
        Send an external request to a root unit.

        Args:
            root_id: ID of the unit to request

        Returns:
            True if the request was accepted
        """
        if root_id not in self.network.units:
            return False

        unit = self.network.units[root_id]
        if isinstance(unit, ScriptUnit) and unit.state == ScriptState.INACTIVE:
            self._active_requests.add(root_id)
            return True

        return False

    def cancel(self, root_id: str):
        """
        Cancel an active request.

        Args:
            root_id: ID of the root unit to cancel
        """
        self._active_requests.discard(root_id)

    def step(self) -> StepResult:
        """
        Perform one simulation step.

        This consists of two phases:
        1. Propagation: Route messages from previous step to target units
        2. Calculation: Each unit updates its state and emits new messages

        Returns:
            StepResult with information about the step
        """
        self.step_count += 1
        messages_passed = 0

        # Phase 1: Collect messages from previous step and route to targets
        messages_to_deliver: Dict[str, Dict[LinkType, List[Message]]] = {
            uid: {lt: [] for lt in LinkType}
            for uid in self.network.units
        }

        for unit in self.network.units.values():
            outgoing = unit.get_outgoing_messages()
            for link_type, messages in outgoing.items():
                for msg in messages:
                    for link in unit.out_links[link_type]:
                        target_id = link.target_id
                        messages_to_deliver[target_id][link_type].append(msg)
                        messages_passed += 1

        # Deliver messages to target units
        for uid, msg_dict in messages_to_deliver.items():
            unit = self.network.units[uid]
            for link_type, messages in msg_dict.items():
                for msg in messages:
                    unit.receive_message(link_type, msg)

        # Phase 2: Each unit computes its update
        active_units = []
        confirmed_units = []
        failed_units = []

        for uid, unit in self.network.units.items():
            # Check if this unit has an external request
            external_request = uid in self._active_requests
            if external_request:
                self._active_requests.discard(uid)  # Clear after processing

            # Step the unit
            unit.step(external_request=external_request)

            # Track unit states for result
            if isinstance(unit, ScriptUnit):
                if unit.state in (ScriptState.ACTIVE, ScriptState.WAITING,
                                  ScriptState.REQUESTED):
                    active_units.append(uid)
                elif unit.state == ScriptState.CONFIRMED:
                    confirmed_units.append(uid)
                elif unit.state == ScriptState.FAILED:
                    failed_units.append(uid)
            elif isinstance(unit, TerminalUnit):
                if unit.state == TerminalState.ACTIVE:
                    active_units.append(uid)
                elif unit.state == TerminalState.CONFIRMED:
                    confirmed_units.append(uid)

        # Numerical activation update (if enabled)
        if self.use_numerical_activation:
            self._numerical_step()

        # Determine overall execution status
        roots = self.network.get_roots()
        if not roots:
            # Use all script units without incoming SUB links as potential roots
            roots = {uid for uid, u in self.network.units.items()
                     if isinstance(u, ScriptUnit) and
                     not u.has_links_of_type(LinkType.SUB, "in")}

        root_states = [self.network.units[r].state for r in roots
                       if isinstance(self.network.units[r], ScriptUnit)]

        if all(s == ScriptState.CONFIRMED for s in root_states):
            status = ExecutionStatus.CONFIRMED
        elif all(s in (ScriptState.FAILED, ScriptState.CONFIRMED)
                 for s in root_states) and any(s == ScriptState.FAILED
                                                for s in root_states):
            status = ExecutionStatus.FAILED
        elif all(s == ScriptState.INACTIVE for s in root_states):
            status = ExecutionStatus.IDLE
        else:
            status = ExecutionStatus.RUNNING

        result = StepResult(
            step_number=self.step_count,
            status=status,
            active_units=active_units,
            confirmed_units=confirmed_units,
            failed_units=failed_units,
            messages_passed=messages_passed,
        )

        self._execution_history.append(result)

        # Call registered callbacks
        for callback in self._step_callbacks:
            callback(result)

        return result

    def _numerical_step(self):
        """
        Perform numerical spreading activation update.

        For each link type τ: z^τ = (W^τ)^T * a^τ
        Then update each unit's activation based on incoming z values.
        """
        n = len(self.network.units)
        unit_ids = list(self.network.units.keys())

        # Compute incoming activation for each link type
        z: Dict[LinkType, np.ndarray] = {}
        for lt in LinkType:
            z[lt] = self.W[lt].T @ self.a[lt]

        # Update each unit's outgoing activation based on state
        for lt in LinkType:
            self.a[lt] = np.zeros(n, dtype=float)

        for uid, unit in self.network.units.items():
            i = self._unit_index[uid]

            if isinstance(unit, ScriptUnit):
                # Encode state as numerical activation
                state = unit.state

                if state in (ScriptState.REQUESTED, ScriptState.ACTIVE,
                            ScriptState.SUPPRESSED, ScriptState.WAITING,
                            ScriptState.FAILED):
                    # Emit inhibit_request on POR
                    self.a[LinkType.POR][i] = -1.0
                    # Emit inhibit_confirm on RET
                    self.a[LinkType.RET][i] = -1.0

                if state in (ScriptState.ACTIVE, ScriptState.WAITING):
                    # Emit request on SUB
                    self.a[LinkType.SUB][i] = 1.0
                    # Emit wait on SUR
                    self.a[LinkType.SUR][i] = 0.5

                if state == ScriptState.CONFIRMED:
                    # Emit confirm on SUR
                    self.a[LinkType.SUR][i] = 1.0
                    # Still emit inhibit_confirm on RET
                    self.a[LinkType.RET][i] = -1.0

                if state == ScriptState.TRUE:
                    # Emit inhibit_confirm on RET
                    self.a[LinkType.RET][i] = -1.0

            elif isinstance(unit, TerminalUnit):
                if unit.state == TerminalState.CONFIRMED:
                    # Emit confirm on SUR
                    self.a[LinkType.SUR][i] = 1.0

    def run(self, max_steps: int = 100) -> ExecutionStatus:
        """
        Run the simulation until completion or max steps.

        Args:
            max_steps: Maximum number of steps to run

        Returns:
            Final execution status
        """
        for _ in range(max_steps):
            result = self.step()
            if result.status in (ExecutionStatus.CONFIRMED,
                                 ExecutionStatus.FAILED,
                                 ExecutionStatus.IDLE):
                return result.status

        return ExecutionStatus.RUNNING  # Timed out

    def run_with_request(self, root_id: str, max_steps: int = 100) -> ExecutionStatus:
        """
        Send a request to a root unit and run until completion.

        Args:
            root_id: ID of the root unit to request
            max_steps: Maximum number of steps to run

        Returns:
            Final execution status
        """
        self.request(root_id)
        return self.run(max_steps)

    def get_history(self) -> List[StepResult]:
        """Get the execution history."""
        return self._execution_history.copy()

    def get_unit_states(self) -> Dict[str, Any]:
        """Get current state of all units."""
        states = {}
        for uid, unit in self.network.units.items():
            if isinstance(unit, ScriptUnit):
                states[uid] = {
                    "type": "script",
                    "state": unit.state.name,
                    "activation": unit.activation.copy(),
                }
            elif isinstance(unit, TerminalUnit):
                states[uid] = {
                    "type": "terminal",
                    "state": unit.state.name,
                    "activation": unit.activation.copy(),
                }
        return states

    def reset(self):
        """Reset the engine and network to initial state."""
        self.step_count = 0
        self._active_requests.clear()
        self._execution_history.clear()
        self.network.reset()

        if self.use_numerical_activation:
            for lt in LinkType:
                self.a[lt] = np.zeros(len(self.network.units), dtype=float)

    def print_state(self):
        """Print the current state of all units for debugging."""
        print(f"\n=== Step {self.step_count} ===")
        for uid, unit in self.network.units.items():
            if isinstance(unit, ScriptUnit):
                print(f"  {uid}: {unit.state.name}")
            elif isinstance(unit, TerminalUnit):
                print(f"  {uid}: {unit.state.name} "
                      f"(activation={unit.activation})")
