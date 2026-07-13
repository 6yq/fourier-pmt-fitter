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


def spe_recursive_polya(rng, n, w=0.4, Q1=0.6, s1=0.25, d1=4.5,
                        d2=0.8, Q2=0.6, s2=0.2):
    """kup recursive sampler; returns DETECTED (nonzero) PE charges.

    Primary: direct Gamma(Q1, s1) w.p. w, else a Poisson(d1) sum of
    cascade-electron charges; a cascade electron is Gamma(Q2, s2)
    (absolute units) w.p. w, else a Poisson(d2) brood of further cascade
    electrons.  Matches RecursivePolyaFitter, whose SER is conditioned on
    nonzero charge (zero atoms stripped + renormalized).
    """
    if (1.0 - w) * d2 >= 1.0:
        raise ValueError("(1 - w) * d2 must be < 1 (subcritical).")
    k1 = (Q1 / s1) ** 2
    k2 = (Q2 / s2) ** 2

    def cascade(m):
        """Total charge of m independent cascade electrons."""
        out = np.zeros(m)
        active = np.arange(m)          # owner index of each live electron
        while len(active):
            direct = rng.random(len(active)) < w
            nd = int(direct.sum())
            if nd:
                q = rng.gamma(shape=k2, scale=Q2 / k2, size=nd)
                np.add.at(out, active[direct], q)
            children = rng.poisson(d2, size=int((~direct).sum()))
            active = np.repeat(active[~direct], children)
        return out

    def cascade_conditioned(m):
        """m cascade quanta conditioned on nonzero charge (the model's
        delta1 counts DETECTED cascade electrons: s~ is conditioned
        before the Poisson(d1) compound)."""
        out = cascade(m)
        z = out <= 0.0
        while z.any():
            out[z] = cascade(int(z.sum()))
            z = out <= 0.0
        return out

    out = np.zeros(n)
    is_direct = rng.random(n) < w
    nd = int(is_direct.sum())
    if nd:
        out[is_direct] = rng.gamma(shape=k1, scale=Q1 / k1, size=nd)
    idx = np.where(~is_direct)[0]
    if len(idx):
        K = rng.poisson(d1, size=len(idx))
        tot = int(K.sum())
        if tot:
            s_all = cascade_conditioned(tot)
            owner = np.repeat(np.arange(len(idx)), K)
            out[idx] += np.bincount(owner, weights=s_all, minlength=len(idx))
    return out[out > 0]


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
