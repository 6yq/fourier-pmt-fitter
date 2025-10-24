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


# ===============
#    Recursive
# ===============


def _map_args_recursive(args):
    frac, mean, sigma, lam, lam_r, mean_r, sigma_r = args
    k = (mean / sigma) ** 2
    theta = mean / k
    k_r = (mean_r / sigma_r) ** 2
    theta_r = mean * (sigma_r**2) / mean_r
    return (frac, k, theta, lam, lam_r, k_r, theta_r)


def _sample_s_batch_recursive(rng, m, w, lam2, k2, theta2):
    if m == 0:
        return np.zeros(0, dtype=float)

    charge = np.zeros(m, dtype=float)
    active = np.ones(m, dtype=np.int64)

    while True:
        alive_mask = active > 0
        if not np.any(alive_mask):
            break

        a = active[alive_mask]
        direct = rng.binomial(a, w)
        spawn = a - direct

        nz = direct > 0
        if np.any(nz):
            idxs = np.where(alive_mask)[0][nz]
            for i, d in zip(idxs, direct[nz]):
                charge[i] += rng.gamma(shape=k2 * d, scale=theta2)

        children = np.zeros_like(a)
        nzs = spawn > 0
        if np.any(nzs):
            children[nzs] = rng.poisson(lam2 * spawn[nzs])

        active[alive_mask] = children

    return charge


def _sample_S_batch_recursive(rng, m, w, k1, theta1, lam1, lam2, k2, theta2):
    if m == 0:
        return np.zeros(0, dtype=float)

    out = np.zeros(m, dtype=float)
    u = rng.random(m)
    direct_mask = u < w
    rec_mask = ~direct_mask

    nd = int(np.sum(direct_mask))
    if nd > 0:
        out[direct_mask] = rng.gamma(shape=k1, scale=theta1, size=nd)

    nr = int(np.sum(rec_mask))
    if nr > 0:
        K = rng.poisson(lam1, size=nr)
        tot_s = int(np.sum(K))
        if tot_s > 0:
            s_all = _sample_s_batch_recursive(rng, tot_s, w, lam2, k2, theta2)
            owner = np.repeat(np.arange(nr), K)
            sums = np.bincount(owner, weights=s_all, minlength=nr)
            out[rec_mask] = sums

    return out


def sample_from_spe_recursive(n, args, seed):
    rng = np.random.default_rng(seed)
    w, k1, theta1, lam1, lam2, k2, theta2 = _map_args_recursive(args)
    if (1.0 - w) * lam2 >= 1.0:
        raise ValueError("(1-w)*lam2 must be < 1 for finite mean in recursive sampler.")

    return _sample_S_batch_recursive(rng, n, w, k1, theta1, lam1, lam2, k2, theta2)


def sample_from_pe_recursive(n_events, args, occ, seed):
    rng = np.random.default_rng(seed)
    mu = -np.log(1.0 - occ)
    N = rng.poisson(mu, size=n_events)

    tot = int(np.sum(N))
    if tot == 0:
        return np.zeros(0, dtype=float)

    S_all = sample_from_spe_recursive(tot, args, seed)
    owners = np.repeat(np.arange(n_events), N)
    Q = np.bincount(owners, weights=S_all, minlength=n_events)
    return Q[Q != 0.0]
