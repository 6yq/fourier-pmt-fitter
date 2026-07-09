# ===========================================================================
# tests/conftest.py
#
# Toy Monte Carlo samplers and shared fixtures.
#
# All samplers generate per-trigger total charges in "gain units"
# (SPE mean of order 1) and return everything needed to build a fitter:
# the recorded charges, the total number of triggers A, and the truth.
# ===========================================================================

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)


N_EVENTS = 200_000
SEED = 1234


# ==============================
#     SER samplers
# ==============================


def spe_gauss(rng, n, mean=1.0, sigma=0.3):
    return rng.normal(mean, sigma, size=n)


def spe_polya(rng, n, mean=1.0, sigma=0.4):
    k = (mean / sigma) ** 2
    theta = mean / k
    return rng.gamma(shape=k, scale=theta, size=n)


def spe_gamma_tweedie(rng, n, frac=0.6, mean=0.6, sigma=0.25, lam_c=5.0,
                      mean_t=0.6, sigma_t=0.2):
    """frac: Gamma(mean, sigma); 1-frac: Poisson(lam_c) sum of small Gammas."""
    out = np.zeros(n)
    is_main = rng.random(n) < frac
    n_main = int(is_main.sum())
    if n_main:
        k = (mean / sigma) ** 2
        out[is_main] = rng.gamma(shape=k, scale=mean / k, size=n_main)
    idx = np.where(~is_main)[0]
    if len(idx):
        counts = rng.poisson(lam_c, size=len(idx))
        k_ts = (mean_t / sigma_t) ** 2
        theta_ts = mean * sigma_t**2 / mean_t
        tot = int(counts.sum())
        if tot:
            draws = rng.gamma(shape=k_ts, scale=theta_ts, size=tot)
            owner = np.repeat(np.arange(len(idx)), counts)
            out[idx] += np.bincount(owner, weights=draws, minlength=len(idx))
    return out


def spe_recursive_polya(rng, n, frac=0.4, mean=0.6, sigma=0.25, lam=4.5,
                        lam_r=0.8, mean_r=0.6, sigma_r=0.2):
    """Recursive secondary-emission sampler (matches RecursivePolyaFitter)."""
    if (1.0 - frac) * lam_r >= 1.0:
        raise ValueError("(1 - frac) * lam_r must be < 1.")
    k1 = (mean / sigma) ** 2
    theta1 = mean / k1
    k2 = (mean_r / sigma_r) ** 2
    theta2 = mean * sigma_r**2 / mean_r

    def sample_cascade(m):
        """m cascade charges s."""
        charge = np.zeros(m)
        active = np.ones(m, dtype=np.int64)
        while True:
            alive = active > 0
            if not alive.any():
                break
            a = active[alive]
            direct = rng.binomial(a, frac)
            nz = direct > 0
            if nz.any():
                idxs = np.where(alive)[0][nz]
                for i, d in zip(idxs, direct[nz]):
                    charge[i] += rng.gamma(shape=k2 * d, scale=theta2)
            children = np.zeros_like(a)
            spawn = a - direct
            nzs = spawn > 0
            if nzs.any():
                children[nzs] = rng.poisson(lam_r * spawn[nzs])
            active[alive] = children
        return charge

    out = np.zeros(n)
    is_direct = rng.random(n) < frac
    nd = int(is_direct.sum())
    if nd:
        out[is_direct] = rng.gamma(shape=k1, scale=theta1, size=nd)
    idx = np.where(~is_direct)[0]
    if len(idx):
        K = rng.poisson(lam, size=len(idx))
        tot = int(K.sum())
        if tot:
            s_all = sample_cascade(tot)
            owner = np.repeat(np.arange(len(idx)), K)
            out[idx] += np.bincount(owner, weights=s_all, minlength=len(idx))
    return out


# ==============================
#     Trigger-level sampler
# ==============================


def erf_efficiency(q, loc, scale):
    from scipy.special import erf

    return 0.5 * (1.0 + erf((q - loc) / (scale * np.sqrt(2.0))))


def sample_spectrum(
    n_events,
    lam,
    spe_sampler,
    ped=None,
    thr=None,
    seed=SEED,
):
    """Per-trigger charges through the full readout chain.

    Parameters
    ----------
    lam : float
        Poisson mean PE per trigger.
    spe_sampler : callable(rng, n) -> charges
    ped : None or (mean, sigma)
        Gaussian pedestal added to every trigger.  When None, triggers
        with exactly zero total charge join the zero category.
    thr : None or (loc, scale)
        erf threshold; each trigger is recorded with probability eff(Q).

    Returns
    -------
    dict with Q (recorded charges), A (total triggers), zero count.
    """
    rng = np.random.default_rng(seed)
    N = rng.poisson(lam, size=n_events)
    Q = np.zeros(n_events)
    tot = int(N.sum())
    if tot:
        draws = spe_sampler(rng, tot)
        owner = np.repeat(np.arange(n_events), N)
        Q = np.bincount(owner, weights=draws, minlength=n_events)

    if ped is not None:
        Q = Q + rng.normal(ped[0], ped[1], size=n_events)
        recorded = np.ones(n_events, dtype=bool)
    else:
        recorded = Q != 0.0

    if thr is not None:
        eff = erf_efficiency(Q, thr[0], thr[1])
        recorded &= rng.random(n_events) < eff

    return dict(
        Q=Q[recorded],
        A=n_events,
        zero=n_events - int(recorded.sum()),
        N=N,
    )


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(SEED)
