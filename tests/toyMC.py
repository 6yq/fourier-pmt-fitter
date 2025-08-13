import numpy as np
import matplotlib.pyplot as plt

from tweedie import tweedie
from scipy.stats import norm, gamma, poisson, bernoulli


def _map_args(args):
    frac, mean, sigma, lam, mean_t, sigma_t = args
    alpha_ts = (mean_t**2) / (sigma_t**2)
    beta_ts = mean_t / (sigma_t**2)
    k = (mean / sigma) ** 2
    theta = mean / k
    mu = lam * alpha_ts * mean / beta_ts
    p = 1 + 1 / (alpha_ts + 1)
    phi = (alpha_ts + 1) * pow(lam * alpha_ts, 1 - p) / pow(beta_ts / mean, 2 - p)
    return (frac, k, theta, mu, p, phi)


def split_and_sum_compact(samples, pes):
    pes = np.array(pes)
    assert samples.shape[0] == pes.sum()
    idxs = np.nonzero(pes)[0]
    result = []
    idx = 0
    for i in idxs:
        count = pes[i]
        result.append(samples[idx : idx + count].sum())
        idx += count
    return np.array(result)


def sample_from_ped(n, args, seed):
    mean, sigma = args
    return norm.rvs(loc=mean, scale=sigma, size=n, random_state=seed)


def sample_from_spe(n, args, seed):
    frac, k, theta, mu, p, phi = _map_args(args)
    samples = bernoulli.rvs(
        p=frac, size=n, random_state=seed
    )  # 1 from gamma, 0 from tweedie
    gamma_samples = sum(samples)
    tweedie_samples = n - gamma_samples
    gamma_rvs = gamma.rvs(a=k, scale=theta, size=gamma_samples, random_state=seed)
    tweedie_rvs = tweedie.rvs(
        mu=mu, p=p, phi=phi, size=tweedie_samples, random_state=seed
    )
    all_rvs = np.zeros_like(samples)
    all_rvs[samples.astype(bool)] = gamma_rvs
    all_rvs[~samples.astype(bool)] = tweedie_rvs
    return all_rvs


def sample_from_pe(n, args, occ, seed):
    mu = -np.log(1 - occ)
    print(f"mu = {mu:.2f} equals occupancy = {occ}")
    pes = poisson.rvs(mu, size=n, random_state=seed)

    nonZeroPEs = sum(pes)
    nonZeroSamples = sample_from_spe(nonZeroPEs, args[2:], seed)
    zeroSamples = sample_from_ped(sum(pes == 0), args[:2], seed)
    nonZeroRes = split_and_sum_compact(nonZeroSamples, pes)
    return nonZeroRes[nonZeroRes != 0]
