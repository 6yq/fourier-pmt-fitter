# ===========================================================================
# models/polya.py
#
# Polya (Gamma) family SER models, JAX edition.
#
# The Polya distribution is a reparameterized Gamma: with mean m and
# standard deviation s, the shape is k = (m/s)^2 and the scale is
# theta = m/k.  The characteristic function under the numpy.fft sign
# convention is
#
#     f~(w) = (1 + i * theta * w)^(-k)
#
# Default parameter blocks are expressed in units of the data-driven SPE
# charge scale (self.scale); pass an explicit spe_block for full control.
# ===========================================================================

import numpy as np
import jax.numpy as jnp

from ..core.base import PMTSpectrumFitter, ParamBlock
from ..core.lambert_w import lambert_w0


def ft_gamma(freq, k, theta):
    return (1.0 + 1j * theta * freq) ** (-k)


def _k_theta(mean, sigma):
    k = (mean / sigma) ** 2
    return k, mean / k


# ======================
#       Polya
# ======================


def _ser_ft_polya(freq, spe):
    k, theta = _k_theta(spe[0], spe[1])
    return ft_gamma(freq, k, theta)


class PolyaFitter(PMTSpectrumFitter):
    """Single-Polya (Gamma) SER: parameterized by (mean, sigma)."""

    def _model_callables(self):
        return _ser_ft_polya, None, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["spe_mean", "spe_sigma"],
            init=np.array([1.0, 0.4]) * s,
            bounds=[(0.2 * s, 5.0 * s), (0.02 * s, 2.0 * s)],
        )

    def get_gain(self, spe, kind="gm"):
        mean, sigma = float(spe[0]), float(spe[1])
        if kind == "gm":
            return mean
        if kind == "gp":
            k, theta = _k_theta(mean, sigma)
            return (k - 1.0) * theta
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        return {"spe_mean": float(spe[0]), "spe_sigma": float(spe[1])}


# ======================
#       Bi-Polya
# ======================


def _ser_ft_bipolya(freq, spe):
    frac, mean, sigma, mean_t, sigma_t = spe
    k, theta = _k_theta(mean, sigma)
    k_ts = (mean_t / sigma_t) ** 2
    theta_ts = mean * sigma_t**2 / mean_t
    return (1.0 - frac) * ft_gamma(freq, k, theta) + frac * ft_gamma(
        freq, k_ts, theta_ts
    )


class BiPolyaFitter(PMTSpectrumFitter):
    """Normal plus missing-first-dynode Polya.

    spe = (frac, mean, sigma, mean_t, sigma_t); the missing component is
    a Gamma with mean = mean * mean_t and sigma = mean * sigma_t.
    """

    def _model_callables(self):
        return _ser_ft_bipolya, None, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["frac", "spe_mean", "spe_sigma", "mean_t", "sigma_t"],
            init=np.array([0.10, 1.0 * s, 0.4 * s, 0.5, 0.1]),
            bounds=[
                (0.0, 0.5),
                (0.2 * s, 5.0 * s),
                (0.02 * s, 2.0 * s),
                (0.01, 1.0),
                (0.005, 1.0),
            ],
        )

    def get_gain(self, spe, kind="gm"):
        frac, mean, sigma, mean_t, _ = (float(v) for v in spe)
        if kind == "gp":
            k, theta = _k_theta(mean, sigma)
            return (k - 1.0) * theta
        if kind == "gm":
            return (1.0 - frac) * mean + frac * mean * mean_t
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        frac, mean, sigma, mean_t, sigma_t = (float(v) for v in spe)
        return {
            "frac": frac,
            "spe_mean": mean,
            "spe_sigma": sigma,
            "miss_mean": mean * mean_t,
            "miss_sigma": mean * sigma_t,
        }


# ======================
#   Polya Exponential
# ======================


def _ser_ft_polya_exp(freq, spe):
    frac, mean, sigma, scale_e = spe
    k, theta = _k_theta(mean, sigma)
    ft_e = (1.0 + 1j * scale_e * freq) ** (-1.0)
    return (1.0 - frac) * ft_gamma(freq, k, theta) + frac * ft_e


class PolyaExpFitter(PMTSpectrumFitter):
    """Polya plus exponential SER.

    spe = (frac, mean, sigma, scale_e); frac is the exponential fraction.

    Notes
    -----
    - Introduced by Marcos Dracos;
    - See JUNO-doc-14081.
    """

    def _model_callables(self):
        return _ser_ft_polya_exp, None, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["frac", "spe_mean", "spe_sigma", "exp_scale"],
            init=np.array([0.4, 0.8 * s, 0.2 * s, 1.6 * s]),
            bounds=[
                (0.0, 1.0),
                (0.2 * s, 5.0 * s),
                (0.02 * s, 2.0 * s),
                (0.05 * s, 10.0 * s),
            ],
        )

    def get_gain(self, spe, kind="gm"):
        frac, mean, sigma, scale_e = (float(v) for v in spe)
        if kind == "gp":
            return mean
        if kind == "gm":
            return frac * scale_e + (1.0 - frac) * mean
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        frac, mean, sigma, scale_e = (float(v) for v in spe)
        return {
            "frac": frac,
            "spe_mean": mean,
            "spe_sigma": sigma,
            "exp_scale": scale_e,
        }


# ======================
#     Gamma Tweedie
# ======================


def _ser_ft_gamma_tweedie(freq, spe):
    frac, mean, sigma, lam_c, mean_t, sigma_t = spe
    k, theta = _k_theta(mean, sigma)
    k_ts = (mean_t / sigma_t) ** 2
    theta_ts = mean * sigma_t**2 / mean_t
    ts = ft_gamma(freq, k_ts, theta_ts)
    # compound Poisson of small Gammas; includes its mass exp(-lam_c) at 0
    compound = jnp.exp(lam_c * (ts - 1.0))
    return frac * ft_gamma(freq, k, theta) + (1.0 - frac) * compound


def _p0_gamma_tweedie(spe):
    frac, lam_c = spe[0], spe[3]
    return (1.0 - frac) * jnp.exp(-lam_c)


class GammaTweedieFitter(PMTSpectrumFitter):
    """Gamma SER plus a compound-Poisson-of-Gammas (Tweedie) component.

    spe = (frac, mean, sigma, lam_c, mean_t, sigma_t).  With probability
    1 - frac the PE charge is a Poisson(lam_c) sum of small Gammas with
    mean = mean * mean_t and sigma = mean * sigma_t; that branch carries
    a discrete mass (1 - frac) * exp(-lam_c) at zero charge.
    """

    def _model_callables(self):
        return _ser_ft_gamma_tweedie, _p0_gamma_tweedie, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["frac", "spe_mean", "spe_sigma", "lam_c", "mean_t", "sigma_t"],
            init=np.array([0.60, 0.6 * s, 0.25 * s, 5.0, 0.6, 0.2]),
            bounds=[
                (0.3, 1.0),
                (0.1 * s, 5.0 * s),
                (0.02 * s, 2.0 * s),
                (1.0, 20.0),
                (0.05, 1.0),
                (0.01, 1.0),
            ],
        )

    def get_gain(self, spe, kind="gm"):
        frac, mean, sigma, lam_c, mean_t, _ = (float(v) for v in spe)
        if kind == "gp":
            k, theta = _k_theta(mean, sigma)
            return (k - 1.0) * theta
        if kind == "gm":
            frac_nz = frac / (1.0 - (1.0 - frac) * np.exp(-lam_c))
            return frac_nz * mean + (1.0 - frac_nz) * mean * mean_t * lam_c
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        frac, mean, sigma, lam_c, mean_t, sigma_t = (float(v) for v in spe)
        return {
            "frac": frac,
            "spe_mean": mean,
            "spe_sigma": sigma,
            "lam_c": lam_c,
            "ts_mean": mean * mean_t,
            "ts_sigma": mean * sigma_t,
            "p0": (1.0 - frac) * np.exp(-lam_c),
        }


# ======================
#    Recursive Polya
# ======================
#
# MCP-PMT recursive secondary-emission model.  A PE charge S is either a
# direct Gamma (probability frac), or a Poisson(lam) sum of cascade
# charges s, where each s is itself a Gamma (probability frac) or a
# Poisson(lam_r) sum of further s (the recursion).  The cascade
# characteristic function solves the fixed point
#
#     s~ = frac * g_r~ + (1 - frac) * exp(lam_r * (s~ - 1))
#
# whose closed form uses the principal Lambert-W branch:
#
#     s~ = frac * g_r~ - W( lam_r (frac - 1) exp(lam_r (frac g_r~ - 1)) ) / lam_r
#
# Subcriticality (finite cascade) requires (1 - frac) * lam_r < 1, which
# the default bounds enforce.


def _recursive_cascade_ft(ft_r, frac, lam_r):
    arg = lam_r * (frac - 1.0) * jnp.exp(lam_r * (frac * ft_r - 1.0))
    return frac * ft_r - lambert_w0(arg) / lam_r


def _ser_ft_recursive(freq, spe):
    frac, mean, sigma, lam, lam_r, mean_r, sigma_r = spe
    k, theta = _k_theta(mean, sigma)
    k_r = (mean_r / sigma_r) ** 2
    theta_r = mean * sigma_r**2 / mean_r
    ft_g = ft_gamma(freq, k, theta)
    s_tilde = _recursive_cascade_ft(ft_gamma(freq, k_r, theta_r), frac, lam_r)
    return frac * ft_g + (1.0 - frac) * jnp.exp(lam * (s_tilde - 1.0))


def _p0_recursive(spe):
    frac, lam, lam_r = spe[0], spe[3], spe[4]
    c_r = _recursive_cascade_ft(jnp.zeros(()) * 1j, frac, lam_r)
    return jnp.real((1.0 - frac) * jnp.exp(lam * (c_r - 1.0)))


class RecursivePolyaFitter(PMTSpectrumFitter):
    """Recursive secondary-emission Polya model for MCP PMTs.

    spe = (frac, mean, sigma, lam, lam_r, mean_r, sigma_r).  The cascade
    Gamma has mean = mean * mean_r and sigma = mean * sigma_r.  Zero
    charge occurs when the recursive branch dies without emission, with
    probability p0 = (1 - frac) * exp(lam * (c_r - 1)) where c_r is the
    cascade extinction probability.
    """

    def _model_callables(self):
        return _ser_ft_recursive, _p0_recursive, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["frac", "spe_mean", "spe_sigma", "lam", "lam_r", "mean_r", "sigma_r"],
            init=np.array([0.40, 0.6 * s, 0.25 * s, 4.5, 0.8, 0.6, 0.2]),
            bounds=[
                (0.05, 1.0),
                (0.1 * s, 5.0 * s),
                (0.02 * s, 2.0 * s),
                (1.0, 20.0),
                (0.01, 0.99),
                (0.05, 1.0),
                (0.01, 1.0),
            ],
        )

    def get_gain(self, spe, kind="gm"):
        frac, mean, sigma, lam, lam_r, mean_r, _ = (float(v) for v in spe)
        if kind == "gp":
            k, theta = _k_theta(mean, sigma)
            return (k - 1.0) * theta
        if kind == "gm":
            mu1 = mean
            mu2 = mean * mean_r
            Es = frac * mu2 / (1.0 - (1.0 - frac) * lam_r)
            ES = frac * mu1 + (1.0 - frac) * lam * Es
            pS = float(_p0_recursive(jnp.asarray(spe, dtype=jnp.float64)))
            return ES / (1.0 - pS)
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        frac, mean, sigma, lam, lam_r, mean_r, sigma_r = (float(v) for v in spe)
        return {
            "frac": frac,
            "spe_mean": mean,
            "spe_sigma": sigma,
            "lam": lam,
            "lam_r": lam_r,
            "rec_mean": mean * mean_r,
            "rec_sigma": mean * sigma_r,
            "p0": float(_p0_recursive(jnp.asarray(spe, dtype=jnp.float64))),
        }
