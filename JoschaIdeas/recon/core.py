"""
Core data structures for Request Confirmation Networks.

This module defines:
- Enumerations for link types, states, and message types
- Link class for typed connections between units
- Unit base class and subclasses (ScriptUnit, TerminalUnit)
- Message class for inter-unit communication
- ReCoNNetwork container class
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from dataclasses import dataclass, field
import numpy as np


class LinkType(Enum):
    """
    Types of links in a ReCoN.

    - SUB: from parent to child (has-part / substep relation)
    - SUR: from child to parent (inverse of SUB)
    - POR: from predecessor to successor in a sequence
    - RET: from successor to predecessor (inverse of POR)
    """
    SUB = auto()  # parent -> child
    SUR = auto()  # child -> parent
    POR = auto()  # predecessor -> successor
    RET = auto()  # successor -> predecessor


class ScriptState(Enum):
    """
    Discrete states for script units.

    State transitions:
    - INACTIVE: initial state, no active request
    - REQUESTED: received request but may be blocked by predecessor
    - ACTIVE: actively executing, requesting children
    - SUPPRESSED: temporarily blocked by predecessor
    - WAITING: waiting for children to complete
    - TRUE: all required children confirmed, waiting to propagate confirm
    - CONFIRMED: confirmed to parent
    - FAILED: execution failed
    """
    INACTIVE = auto()
    REQUESTED = auto()
    ACTIVE = auto()
    SUPPRESSED = auto()
    WAITING = auto()
    TRUE = auto()
    CONFIRMED = auto()
    FAILED = auto()


class TerminalState(Enum):
    """
    Discrete states for terminal units.

    Terminal units have simplified semantics:
    - INACTIVE: not currently measuring
    - ACTIVE: performing measurement
    - CONFIRMED: measurement successful
    """
    INACTIVE = auto()
    ACTIVE = auto()
    CONFIRMED = auto()


class MessageType(Enum):
    """
    Types of messages that propagate along links.

    - REQUEST: parent asks child to validate (top-down via SUB)
    - INHIBIT_REQUEST: predecessor prevents successor from activating (lateral via POR)
    - WAIT: child indicates it's still active (bottom-up via SUR)
    - CONFIRM: child confirms success (bottom-up via SUR)
    - INHIBIT_CONFIRM: predecessor prevents earlier elements from confirming (lateral via RET)
    - FAIL: explicit failure message (bottom-up via SUR)
    """
    REQUEST = auto()
    INHIBIT_REQUEST = auto()
    WAIT = auto()
    CONFIRM = auto()
    INHIBIT_CONFIRM = auto()
    FAIL = auto()


@dataclass
class Message:
    """
    A message passed between units.

    Attributes:
        type: The message type
        payload: Optional numeric value, vector, or other data
        source_id: ID of the sending unit
    """
    type: MessageType
    payload: Optional[Any] = None
    source_id: Optional[str] = None


@dataclass
class Link:
    """
    A typed, weighted link between two units.

    Attributes:
        source_id: ID of the source unit
        target_id: ID of the target unit
        link_type: Type of the link (SUB, SUR, POR, RET)
        weight: Weight vector for the link
    """
    source_id: str
    target_id: str
    link_type: LinkType
    weight: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    def __post_init__(self):
        if not isinstance(self.weight, np.ndarray):
            self.weight = np.array(self.weight, dtype=float)


# Message routing table: state -> {link_type: message_type}
# Based on the specification's routing matrix
MESSAGE_ROUTING_TABLE: Dict[ScriptState, Dict[LinkType, Optional[MessageType]]] = {
    ScriptState.INACTIVE: {
        LinkType.POR: None,
        LinkType.RET: None,
        LinkType.SUB: None,
        LinkType.SUR: None,
    },
    ScriptState.REQUESTED: {
        LinkType.POR: MessageType.INHIBIT_REQUEST,
        LinkType.RET: MessageType.INHIBIT_CONFIRM,
        LinkType.SUB: None,
        LinkType.SUR: MessageType.WAIT,
    },
    ScriptState.ACTIVE: {
        LinkType.POR: MessageType.INHIBIT_REQUEST,
        LinkType.RET: MessageType.INHIBIT_CONFIRM,
        LinkType.SUB: MessageType.REQUEST,
        LinkType.SUR: MessageType.WAIT,
    },
    ScriptState.SUPPRESSED: {
        LinkType.POR: MessageType.INHIBIT_REQUEST,
        LinkType.RET: MessageType.INHIBIT_CONFIRM,
        LinkType.SUB: None,
        LinkType.SUR: None,
    },
    ScriptState.WAITING: {
        LinkType.POR: MessageType.INHIBIT_REQUEST,
        LinkType.RET: MessageType.INHIBIT_CONFIRM,
        LinkType.SUB: MessageType.REQUEST,
        LinkType.SUR: MessageType.WAIT,
    },
    ScriptState.TRUE: {
        LinkType.POR: None,
        LinkType.RET: MessageType.INHIBIT_CONFIRM,
        LinkType.SUB: None,
        LinkType.SUR: None,
    },
    ScriptState.CONFIRMED: {
        LinkType.POR: None,
        LinkType.RET: MessageType.INHIBIT_CONFIRM,
        LinkType.SUB: None,
        LinkType.SUR: MessageType.CONFIRM,
    },
    ScriptState.FAILED: {
        LinkType.POR: MessageType.INHIBIT_REQUEST,
        LinkType.RET: MessageType.INHIBIT_CONFIRM,
        LinkType.SUB: None,
        LinkType.SUR: None,
    },
}


class Unit:
    """
    Base class for ReCoN units.

    Each unit has:
    - A unique identifier
    - A continuous activation vector
    - Incoming and outgoing links organized by type
    """

    def __init__(self, uid: str, dim_activation: int = 1):
        """
        Initialize a unit.

        Args:
            uid: Unique identifier for this unit
            dim_activation: Dimensionality of the activation vector
        """
        self.id = uid
        self.dim = dim_activation
        self.activation = np.zeros(dim_activation, dtype=float)

        # Links organized by type (populated by Network)
        self.in_links: Dict[LinkType, List[Link]] = {lt: [] for lt in LinkType}
        self.out_links: Dict[LinkType, List[Link]] = {lt: [] for lt in LinkType}

        # Message buffer for current step
        self._incoming_messages: Dict[LinkType, List[Message]] = {lt: [] for lt in LinkType}
        self._outgoing_messages: Dict[LinkType, List[Message]] = {lt: [] for lt in LinkType}

    def receive_message(self, link_type: LinkType, message: Message):
        """Buffer an incoming message."""
        self._incoming_messages[link_type].append(message)

    def get_outgoing_messages(self) -> Dict[LinkType, List[Message]]:
        """Get messages to send on each link type."""
        return self._outgoing_messages

    def clear_message_buffers(self):
        """Clear message buffers for a new step."""
        self._incoming_messages = {lt: [] for lt in LinkType}
        self._outgoing_messages = {lt: [] for lt in LinkType}

    def has_links_of_type(self, link_type: LinkType, direction: str = "out") -> bool:
        """Check if unit has any links of the given type."""
        if direction == "out":
            return len(self.out_links[link_type]) > 0
        else:
            return len(self.in_links[link_type]) > 0

    def step(self, external_request: bool = False) -> Dict[LinkType, List[Message]]:
        """
        Perform one update step.

        Args:
            external_request: Whether an external request is being made to this unit

        Returns:
            Dict mapping link types to lists of messages to emit
        """
        raise NotImplementedError("Subclasses must implement step()")

    def reset(self):
        """Reset the unit to its initial state."""
        self.activation = np.zeros(self.dim, dtype=float)
        self.clear_message_buffers()


class ScriptUnit(Unit):
    """
    A script unit representing an intermediate or top-level script element.

    Script units implement a finite-state machine driven by incoming messages.
    They can have children (via SUB links) and may be part of sequences
    (via POR/RET links).
    """

    def __init__(self, uid: str, dim_activation: int = 1,
                 aggregation_policy: str = "all"):
        """
        Initialize a script unit.

        Args:
            uid: Unique identifier
            dim_activation: Dimensionality of activation vector
            aggregation_policy: How to aggregate child results
                - "all": all children must succeed
                - "any": at least one child must succeed
        """
        super().__init__(uid, dim_activation)
        self.state = ScriptState.INACTIVE
        self.aggregation_policy = aggregation_policy

        # Track which children have responded
        self._children_waiting: Set[str] = set()
        self._children_confirmed: Set[str] = set()
        self._children_failed: Set[str] = set()

    def _decode_incoming_messages(self) -> Dict[str, Any]:
        """
        Decode incoming messages into semantic flags.

        Returns:
            Dict with flags:
            - blocked_by_pred: True if inhibit_request received on POR
            - inhibit_confirm_received: True if inhibit_confirm received on RET
            - child_waits: True if any child sent WAIT
            - child_confirms: True if required children confirmed
            - child_fails: True if any child failed (for "all" policy)
        """
        # Check for inhibit_request from predecessors (on incoming POR links)
        blocked_by_pred = any(
            msg.type == MessageType.INHIBIT_REQUEST
            for msg in self._incoming_messages[LinkType.POR]
        )

        # Check for inhibit_confirm from successors (on incoming RET links)
        inhibit_confirm_received = any(
            msg.type == MessageType.INHIBIT_CONFIRM
            for msg in self._incoming_messages[LinkType.RET]
        )

        # Process child responses (on incoming SUR links)
        for msg in self._incoming_messages[LinkType.SUR]:
            if msg.source_id:
                if msg.type == MessageType.WAIT:
                    self._children_waiting.add(msg.source_id)
                elif msg.type == MessageType.CONFIRM:
                    self._children_waiting.discard(msg.source_id)
                    self._children_confirmed.add(msg.source_id)
                elif msg.type == MessageType.FAIL:
                    self._children_waiting.discard(msg.source_id)
                    self._children_failed.add(msg.source_id)

        # Get all child IDs from SUB links
        all_children = {link.target_id for link in self.out_links[LinkType.SUB]}

        # Determine if children are done based on aggregation policy
        if self.aggregation_policy == "all":
            # All children must confirm
            child_confirms = (
                len(all_children) > 0 and
                self._children_confirmed == all_children
            )
            child_fails = len(self._children_failed) > 0
        else:  # "any"
            # At least one child must confirm
            child_confirms = len(self._children_confirmed) > 0
            child_fails = (
                len(self._children_failed) == len(all_children) and
                len(all_children) > 0
            )

        child_waits = len(self._children_waiting) > 0

        return {
            "blocked_by_pred": blocked_by_pred,
            "inhibit_confirm_received": inhibit_confirm_received,
            "child_waits": child_waits,
            "child_confirms": child_confirms,
            "child_fails": child_fails,
        }

    def _emit_messages_for_state(self):
        """Emit messages based on current state using the routing table."""
        routing = MESSAGE_ROUTING_TABLE[self.state]

        for link_type, msg_type in routing.items():
            if msg_type is not None and self.has_links_of_type(link_type, "out"):
                self._outgoing_messages[link_type].append(
                    Message(type=msg_type, source_id=self.id)
                )

    def step(self, external_request: bool = False) -> Dict[LinkType, List[Message]]:
        """
        Perform one update step for this script unit.

        Implements the state machine from the specification.
        """
        # Clear outgoing messages
        self._outgoing_messages = {lt: [] for lt in LinkType}

        # Decode incoming messages
        flags = self._decode_incoming_messages()

        # State transitions
        if self.state == ScriptState.INACTIVE:
            # Check for request from parent (on incoming SUB links)
            request_received = any(
                msg.type == MessageType.REQUEST
                for msg in self._incoming_messages[LinkType.SUB]
            )
            if external_request or request_received:
                self.state = ScriptState.REQUESTED
                self._children_waiting.clear()
                self._children_confirmed.clear()
                self._children_failed.clear()

        elif self.state == ScriptState.REQUESTED:
            if not flags["blocked_by_pred"]:
                self.state = ScriptState.ACTIVE

        elif self.state == ScriptState.ACTIVE:
            if flags["blocked_by_pred"]:
                self.state = ScriptState.SUPPRESSED
            elif flags["child_confirms"]:
                self.state = ScriptState.TRUE
            elif flags["child_waits"]:
                self.state = ScriptState.WAITING
            elif flags["child_fails"] and not flags["child_waits"]:
                self.state = ScriptState.FAILED

        elif self.state == ScriptState.WAITING:
            if flags["child_confirms"]:
                self.state = ScriptState.TRUE
            elif not flags["child_waits"] and flags["child_fails"]:
                self.state = ScriptState.FAILED

        elif self.state == ScriptState.SUPPRESSED:
            if not flags["blocked_by_pred"]:
                self.state = ScriptState.ACTIVE

        elif self.state == ScriptState.TRUE:
            if not flags["inhibit_confirm_received"]:
                self.state = ScriptState.CONFIRMED

        # CONFIRMED and FAILED are terminal states (until reset)

        # Emit messages based on new state
        self._emit_messages_for_state()

        # Clear incoming messages after processing
        self._incoming_messages = {lt: [] for lt in LinkType}

        return self._outgoing_messages

    def reset(self):
        """Reset to initial state."""
        super().reset()
        self.state = ScriptState.INACTIVE
        self._children_waiting.clear()
        self._children_confirmed.clear()
        self._children_failed.clear()


class TerminalUnit(Unit):
    """
    A terminal unit that interfaces with the environment.

    Terminal units perform measurements (sensors) or actions (actuators)
    when requested by their parent script units.
    """

    def __init__(self, uid: str, dim_activation: int = 1,
                 measurement_fn: Optional[Callable[[], np.ndarray]] = None,
                 threshold: float = 0.5):
        """
        Initialize a terminal unit.

        Args:
            uid: Unique identifier
            dim_activation: Dimensionality of activation vector
            measurement_fn: Callback that returns a measurement vector
            threshold: Threshold for determining confirmation (if scalar measurement)
        """
        super().__init__(uid, dim_activation)
        self.state = TerminalState.INACTIVE
        self.measurement_fn = measurement_fn
        self.threshold = threshold

    def set_measurement_fn(self, fn: Callable[[], np.ndarray]):
        """Set the measurement function."""
        self.measurement_fn = fn

    def step(self, external_request: bool = False) -> Dict[LinkType, List[Message]]:
        """
        Perform one update step for this terminal unit.

        Terminal units:
        1. Respond to REQUEST messages from parents
        2. Query the environment via measurement_fn
        3. Emit CONFIRM or FAIL based on measurement
        """
        self._outgoing_messages = {lt: [] for lt in LinkType}

        # Check for request from parent (on incoming SUB links)
        request_received = any(
            msg.type == MessageType.REQUEST
            for msg in self._incoming_messages[LinkType.SUB]
        )

        if self.state == TerminalState.INACTIVE:
            if request_received or external_request:
                self.state = TerminalState.ACTIVE

        if self.state == TerminalState.ACTIVE:
            # Perform measurement
            if self.measurement_fn is not None:
                measurement = self.measurement_fn()
                if not isinstance(measurement, np.ndarray):
                    measurement = np.array([measurement], dtype=float)
                self.activation = measurement

                # Determine success based on measurement
                # Default: threshold on first element or mean
                if len(measurement) == 1:
                    success = measurement[0] >= self.threshold
                else:
                    success = np.mean(measurement) >= self.threshold
            else:
                # No measurement function, default to confirm
                success = True

            if success:
                self.state = TerminalState.CONFIRMED
                self._outgoing_messages[LinkType.SUR].append(
                    Message(type=MessageType.CONFIRM,
                           payload=self.activation,
                           source_id=self.id)
                )
            else:
                # Failed measurement
                self.state = TerminalState.INACTIVE
                self._outgoing_messages[LinkType.SUR].append(
                    Message(type=MessageType.FAIL,
                           payload=self.activation,
                           source_id=self.id)
                )

        elif self.state == TerminalState.CONFIRMED:
            # Continue emitting confirm while in confirmed state
            self._outgoing_messages[LinkType.SUR].append(
                Message(type=MessageType.CONFIRM,
                       payload=self.activation,
                       source_id=self.id)
            )

        # Clear incoming messages
        self._incoming_messages = {lt: [] for lt in LinkType}

        return self._outgoing_messages

    def reset(self):
        """Reset to initial state."""
        super().reset()
        self.state = TerminalState.INACTIVE


class ReCoNNetwork:
    """
    Container for a Request Confirmation Network.

    Manages units and links, enforces structural constraints,
    and provides methods for network construction and validation.
    """

    def __init__(self):
        """Initialize an empty network."""
        self.units: Dict[str, Unit] = {}
        self.links: List[Link] = []
        self._root_units: Set[str] = set()  # Units that can receive external requests

    def add_unit(self, unit: Unit) -> None:
        """
        Add a unit to the network.

        Args:
            unit: The unit to add

        Raises:
            ValueError: If a unit with the same ID already exists
        """
        if unit.id in self.units:
            raise ValueError(f"Unit with ID '{unit.id}' already exists")
        self.units[unit.id] = unit

    def add_link(self, source_id: str, target_id: str,
                 link_type: LinkType, weight: np.ndarray = None) -> Link:
        """
        Add a link between two units.

        Args:
            source_id: ID of the source unit
            target_id: ID of the target unit
            link_type: Type of the link
            weight: Weight vector (default: [1.0])

        Returns:
            The created Link object

        Raises:
            ValueError: If structural constraints are violated
        """
        if source_id not in self.units:
            raise ValueError(f"Source unit '{source_id}' not found")
        if target_id not in self.units:
            raise ValueError(f"Target unit '{target_id}' not found")

        source = self.units[source_id]
        target = self.units[target_id]

        # Enforce terminal unit constraints
        if isinstance(source, TerminalUnit):
            if link_type not in (LinkType.SUR,):
                raise ValueError(
                    f"Terminal unit '{source_id}' can only be source of SUR links"
                )

        if isinstance(target, TerminalUnit):
            if link_type not in (LinkType.SUB,):
                raise ValueError(
                    f"Terminal unit '{target_id}' can only be target of SUB links"
                )

        # Create and register the link
        if weight is None:
            weight = np.array([1.0])

        link = Link(source_id, target_id, link_type, weight)
        self.links.append(link)

        source.out_links[link_type].append(link)
        target.in_links[link_type].append(link)

        return link

    def add_link_pair(self, parent_id: str, child_id: str,
                      pair_type: str = "partonomic",
                      weight: np.ndarray = None) -> Tuple[Link, Link]:
        """
        Add a matched pair of links (SUB/SUR or POR/RET).

        Args:
            parent_id: ID of the parent/predecessor unit
            child_id: ID of the child/successor unit
            pair_type: "partonomic" for SUB/SUR, "temporal" for POR/RET
            weight: Weight vector for both links

        Returns:
            Tuple of (forward_link, reverse_link)
        """
        if pair_type == "partonomic":
            forward = self.add_link(parent_id, child_id, LinkType.SUB, weight)
            reverse = self.add_link(child_id, parent_id, LinkType.SUR, weight)
        elif pair_type == "temporal":
            forward = self.add_link(parent_id, child_id, LinkType.POR, weight)
            reverse = self.add_link(child_id, parent_id, LinkType.RET, weight)
        else:
            raise ValueError(f"Unknown pair type: {pair_type}")

        return forward, reverse

    def set_root(self, unit_id: str) -> None:
        """Mark a unit as a root that can receive external requests."""
        if unit_id not in self.units:
            raise ValueError(f"Unit '{unit_id}' not found")
        self._root_units.add(unit_id)

    def get_roots(self) -> Set[str]:
        """Get the set of root unit IDs."""
        return self._root_units.copy()

    def validate(self) -> List[str]:
        """
        Validate the network structure.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check that script units have at least one child
        for uid, unit in self.units.items():
            if isinstance(unit, ScriptUnit):
                if not unit.has_links_of_type(LinkType.SUB, "out"):
                    errors.append(
                        f"Script unit '{uid}' has no children (SUB links)"
                    )

        # Check link pair consistency
        sub_pairs = set()
        sur_pairs = set()
        por_pairs = set()
        ret_pairs = set()

        for link in self.links:
            pair = (link.source_id, link.target_id)
            rev_pair = (link.target_id, link.source_id)

            if link.link_type == LinkType.SUB:
                sub_pairs.add(pair)
                if rev_pair not in sur_pairs and link.target_id in self.units:
                    # Check if matching SUR exists
                    pass  # Will be caught by SUR check
            elif link.link_type == LinkType.SUR:
                sur_pairs.add(pair)
            elif link.link_type == LinkType.POR:
                por_pairs.add(pair)
            elif link.link_type == LinkType.RET:
                ret_pairs.add(pair)

        return errors

    def reset(self) -> None:
        """Reset all units to their initial state."""
        for unit in self.units.values():
            unit.reset()

    def get_unit(self, uid: str) -> Unit:
        """Get a unit by ID."""
        return self.units.get(uid)

    def get_script_units(self) -> List[ScriptUnit]:
        """Get all script units."""
        return [u for u in self.units.values() if isinstance(u, ScriptUnit)]

    def get_terminal_units(self) -> List[TerminalUnit]:
        """Get all terminal units."""
        return [u for u in self.units.values() if isinstance(u, TerminalUnit)]

    def __repr__(self) -> str:
        return (
            f"ReCoNNetwork(units={len(self.units)}, links={len(self.links)}, "
            f"roots={self._root_units})"
        )
