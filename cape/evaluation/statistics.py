import numpy as np


def paired_hierarchical_bootstrap(rows, reference, candidate, samples=10000, seed=17):
    if reference == candidate or samples < 1:
        raise ValueError('Need distinct methods and positive bootstrap samples.')
    indexed = {}
    for row in rows:
        if row['method'] not in {reference, candidate}:
            continue
        key = (row['network'], row['fault_type'], row['scenario'],
               row['environment_seed'], str(row['run_id']))
        slot = indexed.setdefault(key, {})
        if row['method'] in slot:
            raise ValueError('Duplicate method for scenario/environment seed/run_id; use a distinct run_id per repeat.')
        slot[row['method']] = float(row['gain'])
    if not indexed or any(reference not in value or candidate not in value for value in indexed.values()):
        raise ValueError('Missing paired method for scenario/environment seed/run_id.')
    strata = {}
    for (network, fault, scenario, environment_seed, run_id), methods in indexed.items():
        strata.setdefault((network, fault), {}).setdefault(run_id, []).append(methods[candidate] - methods[reference])
    arrays = [[np.asarray(values, dtype=float) for values in by_run.values()] for by_run in strata.values()]
    if any(not np.isfinite(values).all() for group in arrays for values in group):
        raise ValueError('Nonfinite paired gains.')
    rng = np.random.default_rng(seed)
    distribution = []
    for _ in range(samples):
        group = []
        for runs in arrays:
            chosen = rng.integers(len(runs), size=len(runs))
            group.append(np.mean([rng.choice(runs[i], size=len(runs[i]), replace=True).mean() for i in chosen]))
        distribution.append(np.mean(group))
    distribution = np.asarray(distribution)
    observed = float(np.mean([np.mean([values.mean() for values in group]) for group in arrays]))
    centered = distribution - observed
    p = min(1., (1 + np.count_nonzero(np.abs(centered) >= abs(observed))) / (samples + 1))
    return {'difference': observed, 'ci95': np.quantile(distribution, [.025, .975]).tolist(), 'p': p}


def benjamini_hochberg(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError('Invalid p-values.')
    order = np.argsort(values)
    adjusted = np.minimum.accumulate((values[order] * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    result = np.empty_like(values); result[order] = np.minimum(1, adjusted)
    return result.tolist()
