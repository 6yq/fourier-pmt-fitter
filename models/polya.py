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
import jax
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
# MCP-PMT recursive secondary-emission model, following the official
# kup-gain kGammaRecursive (TF1Qspec::FFT_Recursive).  A PE charge S is
# either a direct Gamma(Q1, sigma1) (probability w), or a Poisson(delta1)
# sum of cascade charges s; each s is a Gamma(Q2, sigma2) (probability w)
# or a Poisson(delta2) sum of further s.  The cascade characteristic
# function solves the fixed point
#
#     s~ = w * g2~ + (1 - w) * exp(delta2 * (s~ - 1))
#
# via the principal Lambert-W branch:
#
#     s~ = w * g2~ - W( delta2 (w - 1) exp(delta2 (w g2~ - 1)) ) / delta2
#
# kup-gain convention: the zero-charge atoms are STRIPPED and the FT
# renormalized twice — cascade conditioned on nonzero (s~ - p_s0)/(1 -
# p_s0), then the PE response conditioned on detection (f - (1-w)e^-d1)
# / (1 - (1-w)e^-d1).  The SER is therefore the charge of a DETECTED PE:
# lam counts detected PEs, occupancy = 1 - exp(-lam), and the n-PE
# component is the n-fold convolution of nonzero charges (2 PE peaks at
# ~2x gain).  delta2 may exceed 1 (kup allows supercritical cascades;
# the same principal W branch is used).


def _recursive_cascade_ft(ft_r, w, delta2):
    arg = delta2 * (w - 1.0) * jnp.exp(delta2 * (w * ft_r - 1.0))
    return w * ft_r - lambert_w0(arg) / delta2


def _cascade_extinction(w, delta2):
    """P(cascade charge = 0): the fixed point with g2~ -> 0."""
    return jnp.real(_recursive_cascade_ft(jnp.zeros(()) * 1j, w, delta2))


def _ser_ft_recursive(freq, spe):
    w, Q1, s1, d1, d2, Q2, s2 = spe
    g1 = ft_gamma(freq, *_k_theta(Q1, s1))
    g2 = ft_gamma(freq, *_k_theta(Q2, s2))
    s_cas = _recursive_cascade_ft(g2, w, d2)
    p_s0 = _cascade_extinction(w, d2)
    s_cas = (s_cas - p_s0) / (1.0 - p_s0)          # cascade | nonzero
    f = w * g1 + (1.0 - w) * jnp.exp(d1 * (s_cas - 1.0))
    w0 = jnp.exp(-d1)                               # Poisson(0, delta1)
    return (f - (1.0 - w) * w0) / (1.0 - (1.0 - w) * w0)   # PE | detected


# kup-gain laser.yaml mcp/kGammaRecursive limits are in gain units; our
# charge is ADC*ns with 1 gain unit = 666.67 ADC*ns (user convention).
_REC_SCALE = 666.67


class RecursivePolyaFitter(PMTSpectrumFitter):
    """Official (kup-gain kGammaRecursive) recursive model for MCP PMTs.

    spe = (w, Q1, sigma1, delta1, delta2, Q2, sigma2); Q/sigma in ADC*ns
    (kup gain-unit limits x 666.67).  The SER is conditioned on nonzero
    charge (zero atoms stripped + renormalized, as in TF1Qspec), so there
    is no p0 and lam is the detected-PE intensity.
    """

    def _model_callables(self):
        return _ser_ft_recursive, None, None

    def _default_spe_block(self):
        s = _REC_SCALE
        return ParamBlock(
            name="spe",
            names=["w", "Q1", "sigma1", "delta1", "delta2", "Q2", "sigma2"],
            init=np.array([0.4, 1.0 * s, 0.3 * s, 3.0, 1.0, 0.5 * s, 0.2 * s]),
            bounds=[
                (0.1, 0.8),
                (0.4 * s, 2.0 * s),
                (0.1 * s, 0.8 * s),
                (1.0, 8.0),
                (0.1, 5.0),
                (0.05 * s, 0.7 * s),
                (0.01 * s, 0.4 * s),
            ],
        )

    def get_gain(self, spe, kind="gm"):
        w, Q1, s1, d1, d2, Q2, _ = (float(v) for v in spe)
        if kind == "gp":
            k, theta = _k_theta(Q1, s1)
            return (k - 1.0) * theta
        if kind == "gm":
            # mean of the detected-PE charge: -d Im[ser_ft]/df at f = 0
            spe_arr = jnp.asarray(spe, dtype=jnp.float64)
            dspe = jax.grad(
                lambda f: jnp.imag(_ser_ft_recursive(jnp.full((1,), f), spe_arr)[0])
            )(0.0)
            return float(-dspe)
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        w, Q1, s1, d1, d2, Q2, s2 = (float(v) for v in spe)
        p_s0 = float(_cascade_extinction(w, d2))
        return {
            "w": w,
            "Q1": Q1,
            "sigma1": s1,
            "delta1": d1,
            "delta2": d2,
            "Q2": Q2,
            "sigma2": s2,
            "cascade_extinction": p_s0,
            "subcritical": bool((1.0 - w) * d2 < 1.0),
            "gain": self.get_gain(spe, "gm"),
        }
