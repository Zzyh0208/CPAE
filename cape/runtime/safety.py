from dataclasses import dataclass
import numpy as np


@dataclass
class Phase:
    state: str
    duration: float
    minimum: float


def project_program(states, nominal, raw, rules):
    expanded_states, expanded_nominal = [], []
    for index, state in enumerate(states):
        expanded_states.append(state); expanded_nominal.append(nominal[index])
        nxt = states[(index + 1) % len(states)]
        if 'y' in state.lower() and any(c in 'gG' for c in nxt):
            if any(c in 'gG' for c in state):
                expanded_states.append(''.join('y' if c in 'gG' else c for c in state))
                expanded_nominal.append(rules.get('yellow', 3.))
            expanded_states.append('r' * len(state)); expanded_nominal.append(rules.get('all_red', 1.))
    states, nominal = expanded_states, expanded_nominal
    minimum = np.array([rules.get('yellow', 3.) if 'y' in state.lower()
                        else rules.get('all_red', 1.) if not any(c in 'gG' for c in state)
                        else rules.get('min_green', 5.) for state in states], dtype=float)
    maximum = np.maximum(minimum, [rules.get('max_green', 90.) if any(c in 'gG' for c in state)
                                  else minimum[i] for i, state in enumerate(states)])
    low = max(float(minimum.sum()), float(rules.get('cycle_min', minimum.sum())))
    high = min(float(maximum.sum()), float(rules.get('cycle_max', maximum.sum())))
    if high < low:
        raise ValueError('Signal cycle constraints are infeasible.')
    cycle = low + (float(np.clip(raw[0], -1, 1)) + 1) * .5 * (high - low)
    weights = np.maximum(np.asarray(nominal, dtype=float) - minimum, 1e-3)
    green = [i for i, state in enumerate(states) if any(c in 'gG' for c in state)]
    if len(raw) < len(green) + 2:
        raise ValueError('Signal actions require cycle, offset, and one coordinate per green phase.')
    for j, index in enumerate(green):
        weights[index] *= np.exp(float(np.clip(raw[2 + j], -1, 1)))
    weights[maximum == minimum] = 0
    durations = minimum.copy()
    remaining = cycle - durations.sum()
    for _ in range(len(states) + 1):
        capacity = maximum - durations
        available = capacity > 1e-8
        if remaining < 1e-8 or not available.any():
            break
        w = weights * available
        addition = np.minimum(capacity, remaining * w / max(w.sum(), 1e-12))
        durations += addition
        remaining = cycle - durations.sum()
    if abs(durations.sum() - cycle) > 1e-5:
        raise ValueError('Cannot project signal cycle.')
    return [Phase(state, float(duration), float(bound)) for state, duration, bound in zip(states, durations, minimum)]


def validate_states(states):
    if not states or len({len(s) for s in states}) != 1:
        raise ValueError('Signal phase states must have consistent nonzero length.')
    for index, state in enumerate(states):
        nxt = states[(index + 1) % len(states)]
        for old, new in zip(state, nxt):
            if old in 'gG' and new not in 'gGyY':
                raise ValueError('Signal program removes green without a yellow phase.')


def metering_program(states, release, rules):
    release = float(np.clip(release, .4, 1.))
    if len(states) != 3:
        raise ValueError('Meter needs explicit green, yellow and all-red phases.')
    if any(c in 'gG' for c in states[2]) or 'y' not in states[1].lower():
        raise ValueError('Invalid metering clearances.')
    cycle = float(rules.get('cycle', 20))
    yellow, red = float(rules.get('yellow', 3)), float(rules.get('all_red', 1))
    green_max = cycle - yellow - red
    minimum = float(rules.get('min_green', 3))
    green = minimum + release * (green_max - minimum)
    if green_max < minimum:
        raise ValueError('Infeasible metering cycle.')
    return [Phase(states[0], green, minimum), Phase(states[1], yellow, yellow),
            Phase(states[2], cycle - green - yellow, red)]
