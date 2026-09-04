"""Named robot presets for deployable inference contracts.

A preset pairs an :class:`Embodiment` (camera/action geometry and robot
processing) with a :class:`~apxinf.conventions.Convention` (wire keys and state
routing). ``image_keys`` order maps directly to model :data:`VIEW_SLOTS`.
External packages may add pairings with :func:`register_robot_preset`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .. import conventions
from ..conventions import Convention
from ..policies.auto import AutoPolicy
from ..policies.base import VIEW_SLOTS, Policy
from ..processors.robots.unitree_g1 import G1_ROBOT_DIM
from .unitree_g1 import build_unitree_g1_policy

__all__ = [
    "VIEW_SLOTS",
    "Embodiment",
    "Convention",
    "RobotPreset",
    "ROBOT_PRESETS",
    "ROBOT_ALIASES",
    "available_robots",
    "get_robot_preset",
    "register_robot_preset",
    "build_robot_policy",
]


def _build_generic(model_dir, **kwargs) -> Policy:
    """Default builder: the stock policy, no robot-specific pre/post steps."""
    return AutoPolicy.from_pretrained(model_dir, **kwargs)


@dataclass(frozen=True)
class Embodiment:
    """Robot camera/action geometry and its optional processing builder."""

    #: Robot name, used in the served metadata and in preset names.
    name: str
    #: Cameras this body carries; must equal the checkpoint's ``num_views``.
    num_cameras: int
    #: Deployable action width. ``None`` keeps the model's full vector — correct
    #: when a robot output step does the truncation itself (G1's 32→16 encode).
    action_dim: Optional[int] = None
    #: Width of this body's state vector, i.e. how wide ``norm_stats["state"]``
    #: has to be. A fact about the hardware, so it does not follow
    #: :attr:`action_dim`: a robot's state and action spaces need not match
    #: (Franka under LIBERO has 8-dim state and a 7-dim EEF-delta action).
    #: ``None`` means this body makes no claim and the check is skipped.
    state_dim: Optional[int] = None
    #: Width of this body's action vector, i.e. how wide ``norm_stats["actions"]``
    #: has to be. Distinct from :attr:`action_dim`, which is a *loading* knob
    #: saying what to trim the model down to: the G1 declares ``action_dim=None``
    #: because ``UnitreeG1EncodeActions`` owns the truncation, but its actions are
    #: still 16 wide and its statistics must be. ``None`` skips the check.
    action_width: Optional[int] = None
    #: Factory that loads the checkpoint and wires this robot's pre/post steps.
    builder: Callable[..., Policy] = _build_generic
    #: Extra keyword arguments the builder always receives.
    builder_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 1 <= self.num_cameras <= len(VIEW_SLOTS):
            raise ValueError(
                f"Embodiment {self.name!r}: num_cameras={self.num_cameras} is outside "
                f"1..{len(VIEW_SLOTS)}, the view slots {VIEW_SLOTS} a checkpoint fills"
            )
        if (
            self.action_dim is not None
            and self.action_width is not None
            and self.action_dim > self.action_width
        ):
            # Trimming is a slice of the statistics, so it cannot ask for more
            # columns than the body's actions have. Getting this pair backwards
            # would make the preflight check the wrong width and pass a
            # checkpoint that the unnormalizer then fails on, mid-serve.
            raise ValueError(
                f"Embodiment {self.name!r}: action_dim={self.action_dim} trims wider "
                f"than action_width={self.action_width}, but action_width is how many "
                "columns this body's actions (and its norm_stats) have"
            )

    @property
    def has_robot_steps(self) -> bool:
        """Whether this body adds robot-specific pre/post processing."""
        return self.builder is not _build_generic


@dataclass(frozen=True)
class RobotPreset:
    """A validated, deployable pairing of an embodiment and wire convention."""

    #: Registry name, used as ``--robot <name>``. Spelled ``<arm>_<convention>``
    #: when the arm is shared, bare when the body owns its convention.
    name: str
    #: The body being served.
    embodiment: Embodiment
    #: The key convention its client speaks.
    convention: Convention
    #: One-line description for ``--help`` and the served metadata.
    summary: str = ""

    def __post_init__(self) -> None:
        if self.convention.num_cameras != self.embodiment.num_cameras:
            raise ValueError(
                f"RobotPreset {self.name!r}: convention {self.convention.name!r} names "
                f"{self.convention.num_cameras} cameras but embodiment "
                f"{self.embodiment.name!r} has {self.embodiment.num_cameras}; a "
                "convention can only be paired with a body it was recorded on"
            )

    # Flat accessors expose the combined serving contract.

    @property
    def image_keys(self) -> Tuple[str, ...]:
        """Camera wire keys in model slot order — what ``ImageStack`` consumes."""
        return self.convention.image_keys

    @property
    def state_key(self) -> str:
        return self.convention.state_key

    @property
    def prompt_key(self) -> str:
        return self.convention.prompt_key

    @property
    def discrete_state(self) -> Optional[bool]:
        return self.convention.discrete_state

    @property
    def slots(self) -> Tuple[Tuple[str, str], ...]:
        """``(view slot, wire key)`` pairs in model slot order."""
        return self.convention.slots

    @property
    def action_dim(self) -> Optional[int]:
        return self.embodiment.action_dim

    @property
    def state_dim(self) -> Optional[int]:
        return self.embodiment.state_dim

    @property
    def action_width(self) -> Optional[int]:
        return self.embodiment.action_width

    @property
    def builder(self) -> Callable[..., Policy]:
        return self.embodiment.builder

    @property
    def builder_kwargs(self) -> Mapping[str, Any]:
        return self.embodiment.builder_kwargs

    @property
    def num_views(self) -> int:
        """Cameras this preset sends; must equal the checkpoint's ``num_views``."""
        return self.embodiment.num_cameras

    @property
    def has_robot_steps(self) -> bool:
        """Whether this preset wires robot-specific pre/post steps."""
        return self.embodiment.has_robot_steps

    def synthetic_gaps(
        self, *, discrete_state: bool, served_action_dim: int
    ) -> Tuple[str, ...]:
        """Describe preset behavior unavailable to a random-weight server."""
        gaps = []
        if discrete_state:
            gaps.append(
                "discrete_state=True — the synthetic tokenizer ignores state, so the "
                "served metadata says discrete_state=False"
            )
        if self.has_robot_steps:
            gap = (
                f"its robot pre/post steps — {self.builder.__name__} never runs, so no "
                "state decode, delta->absolute or action re-encode is wired"
            )
            if self.action_dim is None:
                # action_dim=None means an output step owns the truncation (G1's
                # 32->16 encode). Without that step the full model vector ships.
                gap += (
                    f"; the served action_dim is the model's full {served_action_dim}, "
                    "not the width that skipped step would emit"
                )
            gaps.append(gap)
        return tuple(gaps)

    def describe(self) -> str:
        """Human-readable slot→wire-key mapping, for ``--help`` and startup logs."""
        rendered = ", ".join(f"{slot}={key}" for slot, key in self.slots)
        return f"{self.name}: {rendered}; state={self.state_key}"


#: --- bodies -----------------------------------------------------------------

#: Franka Emika Panda under LIBERO: 2 cameras, 8 state values, 7 actions.
FRANKA = Embodiment(name="franka", num_cameras=2, action_dim=7, state_dim=8, action_width=7)

#: Unitree G1: 3 cameras and 16 state/action values. The encode step owns output
#: selection, so ``action_dim`` remains ``None`` while ``action_width`` is 16.
UNITREE_G1_BODY = Embodiment(
    name="unitree_g1",
    num_cameras=3,
    action_dim=None,
    state_dim=G1_ROBOT_DIM,
    action_width=G1_ROBOT_DIM,
    builder=build_unitree_g1_policy,
    builder_kwargs={
        "use_delta_joint_actions": True,
        "adapt_to_pi": True,
        # Match OpenPI's float64 normalization. State values near a discretization
        # bin edge can otherwise produce different token ids.
        "norm_dtype": "float64",
    },
)

#: --- deployable pairings -----------------------------------------------------

#: Franka Panda under LIBERO's keys: 2 cameras, 7-dim action.
FRANKA_LIBERO = RobotPreset(
    name="franka_libero",
    embodiment=FRANKA,
    convention=conventions.LIBERO,
    summary="Franka Panda, LIBERO keys: 2 cameras, 7-dim action (6 EEF deltas + gripper)",
)

#: Unitree G1 under its native wire convention.
UNITREE_G1 = RobotPreset(
    name="unitree_g1",
    embodiment=UNITREE_G1_BODY,
    convention=conventions.UNITREE_G1,
    summary="Unitree G1: 3 cameras, 16-DoF state, delta joint actions, 32->16 encode",
)

#: Canonical preset registry; external packages use :func:`register_robot_preset`.
ROBOT_PRESETS: Dict[str, RobotPreset] = {p.name: p for p in (FRANKA_LIBERO, UNITREE_G1)}

#: Accepted non-canonical preset names.
ROBOT_ALIASES: Dict[str, str] = {"libero": "franka_libero"}


def available_robots(*, include_aliases: bool = False) -> Tuple[str, ...]:
    """Registered preset names, for ``--robot`` choices and error messages."""
    names = tuple(ROBOT_PRESETS)
    return names + tuple(ROBOT_ALIASES) if include_aliases else names


def get_robot_preset(name: str) -> RobotPreset:
    """Look up a preset by canonical name or alias, with a listing in the error."""
    canonical = ROBOT_ALIASES.get(name, name)
    try:
        return ROBOT_PRESETS[canonical]
    except KeyError:
        raise KeyError(
            f"unknown robot preset {name!r}; known: {list(available_robots())}"
            f" (aliases: {list(ROBOT_ALIASES)})"
        ) from None


def register_robot_preset(
    preset: RobotPreset,
    *,
    aliases: Iterable[str] = (),
    replace: bool = False,
) -> RobotPreset:
    """Register and return ``preset`` for use at module scope.

    External packages can register during import::

        MY_ROBOT = register_robot_preset(
            RobotPreset(name="myarm_mydataset", embodiment=MY_ARM, convention=MY_KEYS),
            aliases=("myarm",),
        )

    Re-registering needs ``replace=True``; aliases cannot shadow canonical names.
    Registration is process-local.
    """
    if not isinstance(preset, RobotPreset):
        raise TypeError(f"expected a RobotPreset, got {type(preset).__name__}")
    existing = ROBOT_PRESETS.get(preset.name)
    if existing is not None and not replace:
        raise ValueError(
            f"robot preset {preset.name!r} is already registered ({existing.describe()}); "
            "pass replace=True to override it"
        )
    for alias in aliases:
        if alias in ROBOT_PRESETS:
            raise ValueError(
                f"alias {alias!r} for {preset.name!r} is already a canonical preset name; "
                "an alias that shadows a real preset would silently redirect --robot"
            )
        target = ROBOT_ALIASES.get(alias)
        if target is not None and target != preset.name and not replace:
            raise ValueError(
                f"alias {alias!r} already resolves to {target!r}; pass replace=True to move it"
            )
    ROBOT_PRESETS[preset.name] = preset
    for alias in aliases:
        ROBOT_ALIASES[alias] = preset.name
    return preset


def build_robot_policy(
    robot: str,
    model_dir,
    *,
    image_keys: Optional[Sequence[str]] = None,
    state_key: Optional[str] = None,
    prompt_key: Optional[str] = None,
    action_dim: Optional[int] = None,
    discrete_state: Optional[bool] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Policy:
    """Load ``model_dir`` under the named preset, with per-argument overrides.

    Each override defaults to the preset's value; passing one replaces just that
    field. ``image_keys``, ``state_key``, and ``prompt_key`` exist because a
    deployed client may already speak a fixed dialect — they let a server match
    an installed robot stack without editing the preset or touching the client.

    The resulting wire contract is published in the policy ``metadata``
    (``robot`` / ``robot_steps`` / ``image_keys`` / ``state_key`` /
    ``discrete_state``), which the server pushes on connect, so a client can
    assert it rather than guess. ``robot_steps`` says whether this robot's
    pre/post steps are actually wired, so a server that serves the preset's keys
    without its arithmetic cannot pass for the real thing.
    """
    preset = get_robot_preset(robot)
    keys = tuple(image_keys) if image_keys is not None else preset.image_keys
    state = state_key if state_key is not None else preset.state_key
    prompt = prompt_key if prompt_key is not None else preset.prompt_key
    discrete = preset.discrete_state if discrete_state is None else bool(discrete_state)
    width = preset.action_dim if action_dim is None else action_dim
    policy_kwargs = {
        "image_keys": keys,
        "state_key": state,
        "prompt_key": prompt,
        "action_dim": width,
        "metadata": {
            "robot": preset.name,
            "robot_steps": preset.has_robot_steps,
            "robot_slots": [list(pair) for pair in zip(VIEW_SLOTS, keys)],
            "state_dim": preset.state_dim,
            **(dict(metadata) if metadata else {}),
        },
        **dict(preset.builder_kwargs),
        **kwargs,
    }
    if discrete is not None:
        policy_kwargs["discrete_state"] = discrete
    return preset.builder(model_dir, **policy_kwargs)
