"""Dataset recording conventions used on the inference wire.

A :class:`Convention` defines camera keys in model view-slot order, the state
key, the prompt key, and whether state is encoded into the prompt. Conventions
are independent of robot bodies and model implementations.

What a convention may depend on:

* :data:`~apxinf.policies.base.VIEW_SLOTS` — model vocabulary, needed to say
  which slot each camera key fills. That is a one-way read of a naming table; no
  policy class is imported and no model is loaded.

This module does not depend on robot bodies or policy implementations.

A convention alone is not deployable — pairing it with an
:class:`~apxinf.robots.presets.Embodiment` is, and
:class:`~apxinf.robots.presets.RobotPreset` is where the two are validated
against each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .policies.base import VIEW_SLOTS

__all__ = [
    "Convention",
    "LIBERO",
    "UNITREE_G1",
    "CONVENTIONS",
    "available_conventions",
    "get_convention",
    "register_convention",
]


@dataclass(frozen=True)
class Convention:
    """A dataset's recording convention: what the client calls things.

    Everything here is a string on the network, plus the one routing decision
    that follows from how the data was recorded. It survives a change of robot.
    Nothing here knows an action width or a camera count.
    """

    #: Convention name (a dataset, or an integrator's dialect).
    name: str
    #: Camera wire keys, in model view-slot order. Entry *i* fills slot *i*.
    image_keys: Tuple[str, ...]
    #: Wire key of the proprioceptive state vector.
    state_key: str
    #: Wire key of the task instruction. ``prompt`` is openpi's protocol-level
    #: name rather than any one dataset's, which is why it has a default here.
    prompt_key: str = "prompt"
    #: Optional policy override for state string encoding. ``None`` leaves the
    #: choice to the checkpoint's concrete policy.
    discrete_state: Optional[bool] = None

    def __post_init__(self) -> None:
        if not self.image_keys:
            raise ValueError(f"Convention {self.name!r}: at least one camera key is required")
        if len(self.image_keys) > len(VIEW_SLOTS):
            raise ValueError(
                f"Convention {self.name!r}: {len(self.image_keys)} camera keys exceeds "
                f"pi05's {len(VIEW_SLOTS)} view slots {VIEW_SLOTS}"
            )
        if len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError(
                f"Convention {self.name!r}: duplicate camera wire keys {list(self.image_keys)}"
            )

    @property
    def num_cameras(self) -> int:
        """Cameras this convention names — cross-checked against a body's count."""
        return len(self.image_keys)

    @property
    def slots(self) -> Tuple[Tuple[str, str], ...]:
        """``(view slot, wire key)`` pairs — the pairing, for logs and review.

        Derived rather than written by hand: a checkpoint fills view slots from 0
        up, so "base + right wrist" is not a thing to spell wrong.
        """
        return tuple(zip(VIEW_SLOTS, self.image_keys))


#: LIBERO (the 7-DoF sim benchmark), mirroring openpi's ``LiberoInputs``: flat
#: ``observation/...`` keys. State encoding is checkpoint-specific: PI0.5 drops
#: it by default, while WallOSS discretizes it.
LIBERO = Convention(
    name="libero",
    image_keys=("observation/image", "observation/wrist_image"),
    state_key="observation/state",
    discrete_state=None,
)

#: The ``pi05_UnitreeG1`` fine-tune's dialect, from its
#: ``unitreeG1Inputs``/``Outputs``: cameras nested one level under ``images``
#: (spelled as a path for :func:`~apxinf.processors.transforms.lookup_key`), a
#: flat top-level ``state`` rather than LIBERO's ``observation/state``, and state
#: discretized into the prompt.
UNITREE_G1 = Convention(
    name="unitree_g1",
    image_keys=("images/cam_high", "images/cam_left_wrist", "images/cam_right_wrist"),
    state_key="state",
    discrete_state=True,
)

#: Name -> convention. A registry rather than a module-level lookup so a
#: third-party dialect can join it without editing this file
#: (:func:`register_convention`).
CONVENTIONS: Dict[str, Convention] = {c.name: c for c in (LIBERO, UNITREE_G1)}


def available_conventions() -> Tuple[str, ...]:
    """Registered convention names, for error messages and ``--help`` text."""
    return tuple(CONVENTIONS)


def get_convention(name: str) -> Convention:
    """Look up a convention by name, with a listing in the error."""
    try:
        return CONVENTIONS[name]
    except KeyError:
        raise KeyError(
            f"unknown convention {name!r}; known: {list(available_conventions())}"
        ) from None


def register_convention(convention: Convention, *, replace: bool = False) -> Convention:
    """Add ``convention`` to the registry and return it, for use at module scope.

    Lets a deployment that speaks its own dialect register it from its own
    package instead of patching this file. Re-registering a name is an error
    unless ``replace=True``, because a silent overwrite would change what an
    already-written preset resolves to.
    """
    if not isinstance(convention, Convention):
        raise TypeError(f"expected a Convention, got {type(convention).__name__}")
    existing = CONVENTIONS.get(convention.name)
    if existing is not None and not replace:
        raise ValueError(
            f"convention {convention.name!r} is already registered "
            f"(image_keys={list(existing.image_keys)}); pass replace=True to override it"
        )
    CONVENTIONS[convention.name] = convention
    return convention
