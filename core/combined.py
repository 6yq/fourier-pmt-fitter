# ===========================================================================
# core/combined.py
#
# Joint fitter for multiple PMT spectra sharing one set of pedestal
# (extra), threshold (thres) and SER (spe) parameters; one log_A and one
# lam per spectrum.  Typical use: the same PMT channel measured at
# several light intensities.
#
# Combined parameter vector layout:
#   theta = [log_A_0, ..., log_A_{N-1},   <- per-spectrum
#            extra..., thres..., spe...,  <- shared
#            lam_0,   ..., lam_{N-1}]     <- per-spectrum
#
# All fitters must share the same block dimensions and settings; shared
# initial values are taken from fitters[0].
# ===========================================================================

from __future__ import annotations

import jax
import numpy as np
import jax.numpy as jnp

from typing import List
from scipy.optimize import minimize
from dataclasses import dataclass

from .base import PMTSpectrumFitter


# ==============================
#     Combined fit result
# ==============================


@dataclass
class CombinedFitResult:
    converged: bool
    theta: np.ndarray
    theta_err: np.ndarray
    logl: float
    n_iter: int
    message: str
    logA_idx: List[int]
    extra_sl: slice
    thres_sl: slice
    spe_sl: slice
    lam_idx: List[int]

    def log_As(self):
        return self.theta[self.logA_idx], self.theta_err[self.logA_idx]

    def extra(self):
        return self.theta[self.extra_sl], self.theta_err[self.extra_sl]

    def thres(self):
        return self.theta[self.thres_sl], self.theta_err[self.thres_sl]

    def spe(self):
        return self.theta[self.spe_sl], self.theta_err[self.spe_sl]

    def lams(self):
        return self.theta[self.lam_idx], self.theta_err[self.lam_idx]


# ==============================
#     CombinedFitter
# ==============================


class CombinedFitter:
    """Joint MLE for multiple spectra with shared extra + thres + spe.

    Parameters
    ----------
    fitters : list of PMTSpectrumFitter
        Constructed fitters.  All must have identical pedestal/threshold
        settings and block dimensions; shared init values come from
        fitters[0].
    """

    def __init__(self, fitters: List[PMTSpectrumFitter]):
        if not fitters:
            raise ValueError("At least one fitter required.")
        self.fitters = fitters
        self.n_spectra = len(fitters)
        self._validate()
        self._build()
        # One fused, jitted forward+backward pass over all sub-fitters: the
        # per-iteration bottleneck was the python loop of separate float() /
        # np.asarray() device round-trips, not the iteration count.  jitting
        # value_and_grad of the joint objective collapses it to a single XLA
        # call (~11x on the 3-model combined fit).
        self._vg_jit = jax.jit(jax.value_and_grad(self._logl_combined))

    def _validate(self):
        ref = self.fitters[0]
        for i, f in enumerate(self.fitters[1:], 1):
            if f.pedestal != ref.pedestal or f.threshold != ref.threshold:
                raise ValueError(f"Fitter {i}: pedestal/threshold settings differ.")
            for blk in ("extra", "thres", "spe"):
                if (f.layout[blk].stop - f.layout[blk].start) != (
                    ref.layout[blk].stop - ref.layout[blk].start
                ):
                    raise ValueError(f"Fitter {i}: {blk} block dimension differs.")

    def _build(self):
        ref = self.fitters[0]
        n = self.n_spectra
        ly = ref.layout
        n_extra = ly["extra"].stop - ly["extra"].start
        n_thres = ly["thres"].stop - ly["thres"].start
        n_spe = ly["spe"].stop - ly["spe"].start
        n_shared = n_extra + n_thres + n_spe

        self.logA_idx = list(range(n))
        self.extra_sl = slice(n, n + n_extra)
        self.thres_sl = slice(n + n_extra, n + n_extra + n_thres)
        self.spe_sl = slice(n + n_extra + n_thres, n + n_shared)
        self.lam_idx = list(range(n + n_shared, 2 * n + n_shared))

        init_parts, bounds_parts = [], []
        for f in self.fitters:
            init_parts.append(f.init[ly["log_A"]])
            bounds_parts.extend(f.bounds[ly["log_A"].start : ly["log_A"].stop])
        for blk in ("extra", "thres", "spe"):
            init_parts.append(ref.init[ly[blk]])
            bounds_parts.extend(ref.bounds[ly[blk].start : ly[blk].stop])
        for f in self.fitters:
            init_parts.append(f.init[ly["lam"]])
            bounds_parts.extend(f.bounds[ly["lam"].start : ly["lam"].stop])

        self.init = np.concatenate(init_parts)
        self.bounds = bounds_parts
        self.dof = len(self.init)
        self._layout_ref = ly

    # ==============================
    #     Local theta reconstruction
    # ==============================

    def local_theta(self, theta, i):
        """Return fitter i's individual theta from the combined theta."""
        xp = jnp if isinstance(theta, jnp.ndarray) else np
        return xp.concatenate(
            [
                theta[self.logA_idx[i] : self.logA_idx[i] + 1],
                theta[self.extra_sl],
                theta[self.thres_sl],
                theta[self.spe_sl],
                theta[self.lam_idx[i] : self.lam_idx[i] + 1],
            ]
        )

    # ==============================
    #     Joint log-likelihood
    # ==============================

    def _logl_combined(self, theta: jnp.ndarray) -> jnp.ndarray:
        total = jnp.zeros(())
        for i, f in enumerate(self.fitters):
            total = total + f._logl_from_theta(self.local_theta(theta, i))
        return total

    def logl(self, theta):
        return float(self._logl_combined(jnp.asarray(theta, dtype=jnp.float64)))

    # ===========
    #     MLE
    # ===========

    def fit_mle(self, theta0=None, maxiter=1000, pg_tol=0.5) -> CombinedFitResult:
        """Joint L-BFGS-B MLE; gradients assembled per fitter.

        Like PMTSpectrumFitter.fit_mle, the fit is also accepted when the
        scaled projected gradient falls below pg_tol (line searches can
        fail on float noise at the optimum).
        """
        if theta0 is None:
            theta0 = self.init.copy()
        theta0 = np.asarray(theta0, dtype=np.float64)

        self._vg_jit(jnp.asarray(theta0))          # JIT warm-up (compile once)

        def neg_fg(x):
            v, g = self._vg_jit(jnp.asarray(x))
            return -float(v), -np.asarray(g)

        res = minimize(
            neg_fg,
            theta0,
            jac=True,
            method="L-BFGS-B",
            bounds=self.bounds,
            options={"maxiter": maxiter},
        )

        theta_hat = np.asarray(res.x, dtype=np.float64)
        theta_err = self._hessian_errors(theta_hat)

        converged = bool(res.success)
        if not converged:
            g = neg_fg(theta_hat)[1]
            pg = g.copy()
            for j, (lo, hi) in enumerate(self.bounds):
                if lo is not None and theta_hat[j] <= lo and g[j] > 0:
                    pg[j] = 0.0
                if hi is not None and theta_hat[j] >= hi and g[j] < 0:
                    pg[j] = 0.0
            converged = bool(
                np.max(np.abs(pg) * np.maximum(np.abs(theta_hat), 1.0)) < pg_tol
            )

        return CombinedFitResult(
            converged=converged,
            theta=theta_hat,
            theta_err=theta_err,
            logl=-float(res.fun),
            n_iter=int(res.nit),
            message=str(res.message),
            logA_idx=self.logA_idx,
            extra_sl=self.extra_sl,
            thres_sl=self.thres_sl,
            spe_sl=self.spe_sl,
            lam_idx=self.lam_idx,
        )

    def _hessian_errors(self, theta_hat):
        try:
            H = np.asarray(
                jax.hessian(self._logl_combined)(jnp.asarray(theta_hat))
            )
            cov = -np.linalg.pinv(H, hermitian=True)
            diag = np.diag(cov)
            diag = np.where(diag > 0, diag, np.nan)
            return np.sqrt(diag)
        except Exception:
            return np.full_like(theta_hat, np.nan)
