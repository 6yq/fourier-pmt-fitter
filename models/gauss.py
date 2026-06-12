# ===========================================================================
# models/gauss.py
#
# Gaussian-family SER models, JAX edition.  Each model supplies the SER
# characteristic function f~(w) (numpy.fft sign convention,
# N(mu, sigma) -> exp(-i*mu*w - sigma^2 w^2 / 2)) plus gain formulas.
#
# Default parameter blocks are expressed in units of the data-driven SPE
# charge scale (self.scale); pass an explicit spe_block for full control.
# ===========================================================================

import numpy as np
import jax.numpy as jnp

from ..core.base import PMTSpectrumFitter, ParamBlock


def ft_gauss(freq, mu, sigma):
    return jnp.exp(-1j * mu * freq - 0.5 * (sigma * freq) ** 2)


# ======================
#       Gauss
# ======================


def _ser_ft_gauss(freq, spe):
    return ft_gauss(freq, spe[0], spe[1])


class GaussFitter(PMTSpectrumFitter):
    """Single-Gaussian SER: f ~ N(mean, sigma)."""

    def _model_callables(self):
        return _ser_ft_gauss, None, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["spe_mean", "spe_sigma"],
            init=np.array([1.0, 0.4]) * s,
            bounds=[(0.2 * s, 5.0 * s), (0.02 * s, 2.0 * s)],
        )

    def get_gain(self, spe, kind="gm"):
        if kind in ("gm", "gp"):
            return float(spe[0])
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        return {"spe_mean": float(spe[0]), "spe_sigma": float(spe[1])}


# ======================
#       Bi-Gauss
# ======================


def _ser_ft_bigauss(freq, spe):
    ratio, mean, sigma, mean_r, sigma_r = spe
    main = ft_gauss(freq, mean, sigma)
    miss = ft_gauss(freq, mean * mean_r, mean * sigma_r)
    return (1.0 - ratio) * main + ratio * miss


class BiGaussFitter(PMTSpectrumFitter):
    """Normal plus missing-first-dynode Gaussian.

    spe = (ratio, mean, sigma, mean_r, sigma_r); the missing component
    has mean = mean * mean_r and sigma = mean * sigma_r.
    """

    def _model_callables(self):
        return _ser_ft_bigauss, None, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["ratio", "spe_mean", "spe_sigma", "mean_r", "sigma_r"],
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
        ratio, mean, _, mean_r, _ = (float(v) for v in spe)
        if kind == "gp":
            return mean
        if kind == "gm":
            return mean * (1.0 - ratio * (1.0 - mean_r))
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        ratio, mean, sigma, mean_r, sigma_r = (float(v) for v in spe)
        return {
            "ratio": ratio,
            "spe_mean": mean,
            "spe_sigma": sigma,
            "miss_mean": mean * mean_r,
            "miss_sigma": mean * sigma_r,
        }


# ======================
#      Linear Gauss
# ======================


def _ser_ft_linear_gauss(freq, spe):
    df, ds, mean, sigma = spe
    main = ft_gauss(freq, mean, sigma)
    miss = ft_gauss(freq, mean / ds, sigma / ds)
    return (1.0 - df) * main + df * miss


class LinearGaussFitter(PMTSpectrumFitter):
    """Dynode Normal plus missing-first model.

    spe = (df, ds, mean, sigma); the missing component is the main one
    scaled down by ds.

    Notes
    -----
    - Introduced to JUNO by Zhangming, Junting et al;
    - First proposed by K. Lang, J. Day, S. Eilerts et al;
    - See JUNO-doc-13627, 14075.
    """

    def _model_callables(self):
        return _ser_ft_linear_gauss, None, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["df", "ds", "spe_mean", "spe_sigma"],
            init=np.array([0.10, 2.55, 1.0 * s, 0.4 * s]),
            bounds=[
                (0.0, 1.0),
                (1.0, 20.0),
                (0.2 * s, 5.0 * s),
                (0.02 * s, 2.0 * s),
            ],
        )

    def get_gain(self, spe, kind="gm"):
        df, ds, mean, _ = (float(v) for v in spe)
        if kind == "gp":
            return mean
        if kind == "gm":
            return df * mean / ds + (1.0 - df) * mean
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        df, ds, mean, sigma = (float(v) for v in spe)
        return {"df": df, "ds": ds, "spe_mean": mean, "spe_sigma": sigma}


# ======================
#       Tri-Gauss
# ======================


def _trigauss_weights(v1, v2):
    """Stick-breaking simplex: (v1, v2) in (0,1)^2 -> (w1, w2, w3)."""
    w1 = v1
    w2 = (1.0 - v1) * v2
    return w1, w2, 1.0 - w1 - w2


def _ser_ft_trigauss(freq, spe):
    v1, v2, m1, d12, d23, s1, s2, s3 = spe
    w1, w2, w3 = _trigauss_weights(v1, v2)
    m2 = m1 + d12
    m3 = m2 + d23
    return (
        w1 * ft_gauss(freq, m1, s1)
        + w2 * ft_gauss(freq, m2, s2)
        + w3 * ft_gauss(freq, m3, s3)
    )


class TriGaussFitter(PMTSpectrumFitter):
    """Three-Gaussian SER with stick-breaking weights.

    spe = (v1, v2, m1, d12, d23, s1, s2, s3) where the weights are
    w1 = v1, w2 = (1-v1)*v2, w3 = 1 - w1 - w2 (always a valid simplex)
    and the means are m1, m1 + d12, m1 + d12 + d23 (always ordered).
    """

    def _model_callables(self):
        return _ser_ft_trigauss, None, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["v1", "v2", "m1", "d12", "d23", "s1", "s2", "s3"],
            init=np.array(
                [0.20, 0.80, 0.6 * s, 0.4 * s, 0.6 * s, 0.06 * s, 0.2 * s, 0.5 * s]
            ),
            bounds=[
                (0.0, 1.0),
                (0.0, 1.0),
                (0.05 * s, 3.0 * s),
                (1e-3 * s, 3.0 * s),
                (1e-3 * s, 3.0 * s),
                (0.01 * s, 2.0 * s),
                (0.01 * s, 2.0 * s),
                (0.01 * s, 2.0 * s),
            ],
        )

    def get_gain(self, spe, kind="gm"):
        v1, v2, m1, d12, d23, *_ = (float(v) for v in spe)
        w = _trigauss_weights(v1, v2)
        m = (m1, m1 + d12, m1 + d12 + d23)
        if kind == "gm":
            return sum(wi * mi for wi, mi in zip(w, m))
        if kind == "gp":
            return m[int(np.argmax(w))]
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        v1, v2, m1, d12, d23, s1, s2, s3 = (float(v) for v in spe)
        w1, w2, w3 = _trigauss_weights(v1, v2)
        return {
            "w1": w1,
            "w2": w2,
            "w3": w3,
            "m1": m1,
            "m2": m1 + d12,
            "m3": m1 + d12 + d23,
            "s1": s1,
            "s2": s2,
            "s3": s3,
        }


# ======================
#     Gauss Compound
# ======================


def _ser_ft_gauss_compound(freq, spe):
    frac, mean, sigma, lam_c, mean_ts, sigma_ts = spe
    main = ft_gauss(freq, mean, sigma)
    ts = ft_gauss(freq, mean * mean_ts, sigma * sigma_ts)
    # compound Poisson of small Gaussians; includes its mass exp(-lam_c) at 0
    compound = jnp.exp(lam_c * (ts - 1.0))
    return frac * main + (1.0 - frac) * compound


def _p0_gauss_compound(spe):
    frac, lam_c = spe[0], spe[3]
    return (1.0 - frac) * jnp.exp(-lam_c)


class GaussCompoundFitter(PMTSpectrumFitter):
    """Gaussian SER plus a compound-Poisson-of-Gaussians component.

    spe = (frac, mean, sigma, lam_c, mean_ts, sigma_ts).  With
    probability 1 - frac the PE charge is a Poisson(lam_c) sum of small
    Gaussians N(mean * mean_ts, sigma * sigma_ts); that branch has a
    discrete mass (1 - frac) * exp(-lam_c) at zero charge.
    """

    def _model_callables(self):
        return _ser_ft_gauss_compound, _p0_gauss_compound, None

    def _default_spe_block(self):
        s = self.scale
        return ParamBlock(
            name="spe",
            names=["frac", "spe_mean", "spe_sigma", "lam_c", "mean_ts", "sigma_ts"],
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
        frac, mean, _, lam_c, mean_ts, _ = (float(v) for v in spe)
        if kind == "gp":
            return mean
        if kind == "gm":
            # mean conditional on non-zero charge
            frac_nz = frac / (1.0 - (1.0 - frac) * np.exp(-lam_c))
            return frac_nz * mean + (1.0 - frac_nz) * mean * mean_ts * lam_c
        raise ValueError(f"Unknown gain kind: {kind!r}")

    def spe_report(self, spe):
        frac, mean, sigma, lam_c, mean_ts, sigma_ts = (float(v) for v in spe)
        return {
            "frac": frac,
            "spe_mean": mean,
            "spe_sigma": sigma,
            "lam_c": lam_c,
            "ts_mean": mean * mean_ts,
            "ts_sigma": sigma * sigma_ts,
            "p0": (1.0 - frac) * np.exp(-lam_c),
        }
