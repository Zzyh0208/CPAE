import numpy as np

def bootstrap(values, samples=10000, seed=17):
    values = np.asarray(values, dtype=np.float64); generator = np.random.default_rng(seed)
    means = np.asarray([generator.choice(values, len(values), replace=True).mean() for _ in range(samples)])
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))

def paired_probability(reference, candidate, samples=10000, seed=17):
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64); generator = np.random.default_rng(seed)
    means = np.asarray([generator.choice(difference, len(difference), replace=True).mean() for _ in range(samples)])
    return float(2 * min((means <= 0).mean(), (means >= 0).mean()))
