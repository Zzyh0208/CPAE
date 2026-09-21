from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np

@dataclass(frozen=True)
class ActionSpec:
    name: str
    lower: np.ndarray
    upper: np.ndarray

@dataclass(frozen=True)
class NetworkSpec:
    nodes: tuple[str, ...]
    lanes: tuple[str, ...]
    adjacency: np.ndarray
    actions: tuple[ActionSpec, ...]

@dataclass
class Observation:
    history: np.ndarray
    adjacency: np.ndarray
    active_pairs: frozenset[tuple[int, int]]
    control_history: frozenset[tuple[int, int]]

@dataclass
class Transition:
    observation: Observation
    control_pairs: frozenset[tuple[int, int]]
    action: np.ndarray
    reward: float
    next_observation: Observation
    terminated: bool
    diagnostics: Mapping[str, float]

@dataclass(frozen=True)
class Scenario:
    identifier: str
    arguments: tuple[str, ...]
    fault: Mapping[str, object]
