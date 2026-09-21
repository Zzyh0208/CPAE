import numpy as np

def recovery_gain(normal, fault, value): return 100 * (fault - value) / max(fault - normal, 1e-6)
def time_to_80(gains, period_seconds):
    target = .8 * max(gains)
    for index, value in enumerate(gains, 1):
        if value >= target: return index * period_seconds
    return len(gains) * period_seconds
def summary(records):
    keys = records[0].keys()
    return {key: float(np.mean([item[key] for item in records])) for key in keys}
