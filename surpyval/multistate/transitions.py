"""The states of a multi-state model and the moves allowed between them."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def _label(value: Any, what: str) -> Any:
    """A state label: any hashable value but ``None``."""
    if value is None:
        raise ValueError(f"{what} cannot be None: a state needs a label.")
    try:
        hash(value)
    except TypeError:
        raise ValueError(
            f"{what} {value!r} is not hashable: use a number, a string or "
            "a tuple as a state label."
        ) from None
    return value


class TransitionMatrix:
    """
    The states of a multi-state model and the transitions allowed between
    them.

    A multi-state model follows each unit through a set of *states* (for
    example "healthy", "ill" and "dead"). A *transition* is a move from
    one state straight into another; a state with no way out is
    *absorbing* (death, scrapped), the others are *transient*. The
    transitions are numbered in the order given, which is the order of
    ``transitions`` and of every per-transition result.

    Build one from a list of ``(from, to)`` pairs, or with one of the
    common shapes: :meth:`illness_death`, :meth:`competing_risks` and
    :meth:`progressive`.

    Parameters
    ----------
    transitions : iterable of (from, to) pairs
        The allowed transitions, each a pair of state labels (any hashable
        values but ``None``). A transition from a state into itself, and a
        pair given twice, are refused.
    states : sequence, optional
        Every state, in the order results report them. Defaults to the
        states of ``transitions`` in the order they first appear there. It
        may add states no transition reaches or leaves.

    Attributes
    ----------
    states : tuple
        The states, in order.
    transitions : tuple of (from, to)
        The allowed transitions, in order.
    matrix : numpy array
        ``(K, K)`` integers: ``matrix[i, j]`` is the number of the
        transition from state ``i`` to state ``j``, or -1 where there is
        none (mstate's ``tmat``, numbered from 0).

    Examples
    --------
    >>> from surpyval.multistate import TransitionMatrix
    >>> tm = TransitionMatrix.illness_death()
    >>> tm.states
    ('healthy', 'ill', 'dead')
    >>> tm.transitions
    (('healthy', 'ill'), ('healthy', 'dead'), ('ill', 'dead'))
    >>> tm.absorbing
    ('dead',)
    >>> tm.matrix
    array([[-1,  0,  1],
           [-1, -1,  2],
           [-1, -1, -1]])
    """

    states: tuple
    transitions: tuple

    def __init__(
        self,
        transitions: Iterable[Sequence[Any]],
        states: Sequence[Any] | None = None,
    ) -> None:
        pairs = []
        for pair in transitions:
            pair = tuple(pair)
            if len(pair) != 2:
                raise ValueError(
                    f"Each transition is a (from, to) pair; got {pair!r}."
                )
            a = _label(pair[0], "A transition's 'from' state")
            b = _label(pair[1], "A transition's 'to' state")
            if a == b:
                raise ValueError(
                    f"The transition {pair!r} goes from a state into "
                    "itself; a transition is a move into another state."
                )
            if (a, b) in pairs:
                raise ValueError(f"The transition {pair!r} is given twice.")
            pairs.append((a, b))
        if not pairs:
            raise ValueError(
                "A multi-state model needs at least one transition."
            )
        seen: list = []
        for a, b in pairs:
            for s in (a, b):
                if s not in seen:
                    seen.append(s)
        if states is None:
            order = seen
        else:
            order = [_label(s, "A state") for s in states]
            if len(set(order)) != len(order):
                raise ValueError(f"A state is listed twice in {states!r}.")
            missing = [s for s in seen if s not in order]
            if missing:
                raise ValueError(
                    f"The transitions use states {missing!r} that are not "
                    f"in states={list(states)!r}: list every state."
                )
        self.states = tuple(order)
        self.transitions = tuple(pairs)
        self._index = {s: i for i, s in enumerate(self.states)}

    # -- common shapes ------------------------------------------------------

    @classmethod
    def illness_death(
        cls,
        states: Sequence[Any] = ("healthy", "ill", "dead"),
        recovery: bool = False,
    ) -> TransitionMatrix:
        """The illness-death model: from the first state to the second
        (illness) or the third (death), and from the second to the third;
        with ``recovery=True`` also back from the second to the first.

        Examples
        --------
        >>> from surpyval.multistate import TransitionMatrix
        >>> TransitionMatrix.illness_death(
        ...     ("working", "degraded", "failed"), recovery=True
        ... ).transitions[-1]
        ('degraded', 'working')
        """
        if len(states) != 3:
            raise ValueError(
                "The illness-death model has three states; got "
                f"{list(states)!r}."
            )
        a, b, c = states
        pairs = [(a, b), (a, c), (b, c)]
        if recovery:
            pairs.append((b, a))
        return cls(pairs, states=states)

    @classmethod
    def competing_risks(
        cls, causes: Sequence[Any], initial: Any = "event-free"
    ) -> TransitionMatrix:
        """Competing risks as a multi-state model: one transient state,
        ``initial``, and an absorbing state for each cause.

        Examples
        --------
        >>> from surpyval.multistate import TransitionMatrix
        >>> TransitionMatrix.competing_risks(["relapse", "death"]).matrix
        array([[-1,  0,  1],
               [-1, -1, -1],
               [-1, -1, -1]])
        """
        return cls([(initial, cause) for cause in causes])

    @classmethod
    def progressive(cls, states: Sequence[Any]) -> TransitionMatrix:
        """States in a chain, each leading only to the next (a unit
        that wears through stages, or a disease through its stages).

        Examples
        --------
        >>> from surpyval.multistate import TransitionMatrix
        >>> TransitionMatrix.progressive([1, 2, 3, 4]).transitions
        ((1, 2), (2, 3), (3, 4))
        """
        if len(states) < 2:
            raise ValueError("A progressive model needs two states or more.")
        return cls(list(zip(states[:-1], states[1:])), states=states)

    @classmethod
    def from_matrix(
        cls, matrix: npt.ArrayLike, states: Sequence[Any] | None = None
    ) -> TransitionMatrix:
        """From a ``(K, K)`` matrix that is true (non-zero) where a
        transition is allowed, read row by row, and the state labels
        (``0`` to ``K - 1`` by default). The diagonal must be zero.

        Examples
        --------
        >>> from surpyval.multistate import TransitionMatrix
        >>> TransitionMatrix.from_matrix(
        ...     [[0, 1, 1], [1, 0, 1], [0, 0, 0]], ["up", "down", "gone"]
        ... ).transitions
        (('up', 'down'), ('up', 'gone'), ('down', 'up'), ('down', 'gone'))
        """
        m = np.asarray(matrix)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError(
                f"The matrix must be square, (K, K); got shape {m.shape}."
            )
        k = m.shape[0]
        labels = list(range(k)) if states is None else list(states)
        if len(labels) != k:
            raise ValueError(
                f"{k} states in the matrix but {len(labels)} labels."
            )
        if np.any(np.diag(m) != 0):
            raise ValueError(
                "The matrix's diagonal must be zero: a transition is a "
                "move into another state."
            )
        pairs = [
            (labels[i], labels[j])
            for i in range(k)
            for j in range(k)
            if i != j and m[i, j]
        ]
        return cls(pairs, states=labels)

    # -- description --------------------------------------------------------

    @property
    def n_states(self) -> int:
        """The number of states."""
        return len(self.states)

    @property
    def n_transitions(self) -> int:
        """The number of transitions."""
        return len(self.transitions)

    def index(self, state: Any) -> int:
        """The position of ``state`` in :attr:`states`."""
        try:
            return self._index[state]
        except (KeyError, TypeError):
            raise ValueError(
                f"Unknown state {state!r}; the states are "
                f"{list(self.states)!r}."
            ) from None

    def transition_index(self, from_state: Any, to_state: Any) -> int:
        """The number of the transition from ``from_state`` to
        ``to_state``."""
        pair = (from_state, to_state)
        for k, t in enumerate(self.transitions):
            if t == pair:
                return k
        raise ValueError(
            f"There is no transition from {from_state!r} to {to_state!r}; "
            f"the transitions are {list(self.transitions)!r}."
        )

    @property
    def matrix(self) -> npt.NDArray:
        k = self.n_states
        out = np.full((k, k), -1, dtype=int)
        for h, (a, b) in enumerate(self.transitions):
            out[self._index[a], self._index[b]] = h
        return out

    @property
    def absorbing(self) -> tuple:
        """The states with no transition out of them."""
        leaving = {a for a, _ in self.transitions}
        return tuple(s for s in self.states if s not in leaving)

    @property
    def transient(self) -> tuple:
        """The states with a transition out of them."""
        leaving = {a for a, _ in self.transitions}
        return tuple(s for s in self.states if s in leaving)

    def from_index(self) -> npt.NDArray:
        """The position of each transition's 'from' state."""
        return np.array(
            [self._index[a] for a, _ in self.transitions], dtype=np.intp
        )

    def to_index(self) -> npt.NDArray:
        """The position of each transition's 'to' state."""
        return np.array(
            [self._index[b] for _, b in self.transitions], dtype=np.intp
        )

    def transition_names(self) -> list[str]:
        """Each transition as text, ``"from -> to"``."""
        return [f"{a} -> {b}" for a, b in self.transitions]

    def to_dict(self) -> dict:
        """The states and transitions as plain lists."""
        from surpyval.serialisation import to_native

        return {
            "states": [to_native(s) for s in self.states],
            "transitions": [
                [to_native(a), to_native(b)] for a, b in self.transitions
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TransitionMatrix:
        """Rebuild from :meth:`to_dict`."""
        from surpyval.univariate.competing_risks.labels import (
            label_from_native,
        )

        return cls(
            [
                (label_from_native(a), label_from_native(b))
                for a, b in d["transitions"]
            ],
            states=[label_from_native(s) for s in d["states"]],
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TransitionMatrix)
            and self.states == other.states
            and self.transitions == other.transitions
        )

    def __hash__(self) -> int:
        return hash((self.states, self.transitions))

    def __repr__(self) -> str:
        moves = ", ".join(f"{a!r} -> {b!r}" for a, b in self.transitions)
        return (
            f"TransitionMatrix(states={list(self.states)!r}, "
            f"transitions=[{moves}])"
        )
