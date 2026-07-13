# ===========================================================================
# core/base.py
#
# PMTSpectrumFitter bundles, for one spectrum:
#   - a static FFT grid aligned with the histogram bin edges,
#   - a JAX-jitted extended-Poisson log-likelihood (binned or unbinned),
#   - fast MLE via L-BFGS-B on JAX gradients,
#   - Hessian-based 1-sigma errors,
#   - per-n-PE component and density accessors.
#
# Subclasses provide the SER characteristic function and parameter block.
# Flat parameter layout consumed by the optimiser:
#
#     theta = [log_A, *extra, *thres, *spe, lam]
#
# where ``extra`` is the Gaussian pedestal block (present iff
# pedestal=True) and ``thres`` is the threshold-efficiency block (present
# iff threshold is not None).  Either block may be empty.
# ===========================================================================

from __future__ import annotations

import jax
import numpy as np
import jax.numpy as jnp

from math import log
from scipy.optimize import minimize
from dataclasses import dataclass, field

from .fft_grid import build_grid
from .likelihood import (
    poisson_pgf,
    make_single_n_pgf,
    make_spectrum_fn,
    make_efficiency,
    make_binned_logl,
    make_unbinned_logl,
    make_bin_prob_fn,
    density_on_xsp,
    cumulative_integrals_at,
    _bin_integrals,
)


# ==============================
#     Parameter block spec
# ==============================


@dataclass
class ParamBlock:
    name: str
    names: list
    init: np.ndarray
    bounds: list


# ==============================
#     Fit result container
# ==============================


@dataclass
class FitResult:
    converged: bool
    theta: np.ndarray
    theta_err: np.ndarray
    logl: float
    n_iter: int
    message: str
    layout: dict = field(default_factory=dict)
    trace: list = None

    def block(self, name):
        sl = self.layout[name]
        return self.theta[sl], self.theta_err[sl]


# ==============================
#     Pedestal FT (universal)
# ==============================


def ft_gaussian_pedestal(freq, extra):
    """Analytic FT of the Gaussian pedestal, numpy.fft sign convention:
    g0~(w) = exp(-i * mu * w - sigma^2 * w^2 / 2).
    """
    return jnp.exp(-1j * extra[0] * freq - 0.5 * extra[1] ** 2 * freq**2)


def _ft_unity(freq, extra):
    return jnp.ones_like(freq) * (1.0 + 0.0j)


# ==============================
#     PMTSpectrumFitter
# ==============================


class PMTSpectrumFitter:
    """PMT charge spectrum fitter with JAX autodiff.

    Accepts either raw charge values (preferred; enables the unbinned
    NUFFT likelihood) or a pre-built histogram.

    Parameters
    ----------
    Q_raw : array-like or None
        Raw charge values.  When given, the histogram is built internally.
    hist, bins : array-like or None
        Pre-built histogram (uniform bins).  When ``Q_raw`` is given,
        ``bins`` is forwarded to numpy.histogram (default "fd").
    A : int or None
        Total triggers including events not in the histogram (undetected
        or below threshold).  Defaults to the number of recorded entries.
    pedestal : bool
        True for a whole spectrum including the pedestal peak; the model
        gains a Gaussian pedestal block (ped_mean, ped_sigma) convolved
        with the PE spectrum.  False for a pedestal-subtracted PE
        spectrum; zero-charge events belong to the zero category.
    threshold : {None, "erf", "logistic"}
        Hardware threshold efficiency applied to the recorded density in
        charge space; adds a block (thr_loc, thr_scale).
    q_min, q_max : float or None
        Override the FFT grid extent (must cover the physical support).
    sample : int or None
        Sub-sampling factor of the FFT grid (dq = bin_width / sample).
        Defaults to 8 with a threshold (Simpson quadrature) and 1 without.
    extra_block, thres_block, spe_block : ParamBlock or None
        Parameter blocks; data-driven defaults are built when None.
    lam_init : float or None
        Initial occupancy mean.  Estimated from the zero fraction when
        possible, else 1.0.
    lam_bounds : tuple
    log_A_err : float
        Fractional half-width of the log_A bound around log(A).
    scale : float or None
        SPE charge scale used by the default spe blocks.  Estimated from
        the data mean when None.
    mode : {"unbinned", "binned"} or None
        Defaults to "unbinned" when Q_raw is given, "binned" otherwise.
    """

    def __init__(
        self,
        Q_raw=None,
        hist=None,
        bins=None,
        A=None,
        pedestal=False,
        threshold=None,
        q_min=None,
        q_max=None,
        sample=None,
        extra_block: ParamBlock = None,
        thres_block: ParamBlock = None,
        spe_block: ParamBlock = None,
        lam_init: float = None,
        lam_bounds=(1e-6, 50.0),
        log_A_err: float = 0.05,
        scale: float = None,
        mode: str = None,
    ):
        # ===========================
        #     Data handling
        # ===========================
        if Q_raw is not None:
            Q_raw = np.asarray(Q_raw, dtype=float)
            _bins = "fd" if bins is None else bins
            hist, bins = np.histogram(Q_raw, bins=_bins)
            mode = mode or "unbinned"
        else:
            if hist is None or bins is None:
                raise ValueError("Provide either Q_raw or (hist, bins).")
            if mode == "unbinned":
                raise ValueError("Unbinned mode requires Q_raw.")
            mode = "binned"

        hist = np.asarray(hist, dtype=float)
        bins = np.asarray(bins, dtype=float)
        _A = int(A) if A is not None else int(hist.sum())

        self.pedestal = bool(pedestal)
        self.threshold = threshold
        self.mode = mode
        self.Q_raw = Q_raw
        self.hist = hist
        self.bins = bins

        if sample is None:
            sample = 8 if threshold is not None else 1
        self.grid = build_grid(
            hist, bins, A=_A, q_min=q_min, q_max=q_max, sample=sample
        )

        # ===========================
        #     Data-driven scales
        # ===========================
        bin_mid = 0.5 * (bins[:-1] + bins[1:])
        bin_width = float(bins[1] - bins[0])
        n_rec = float(hist.sum())
        mean_obs = float(np.average(bin_mid, weights=hist)) if n_rec > 0 else 0.0

        if lam_init is None:
            if self.grid.zero > 0:
                occ_est = min(n_rec / max(_A, 1), 1.0 - 1e-9)
                lam_hat = -log(1.0 - occ_est)
                lam_init_ = lam_hat - (np.exp(lam_hat) - 1.0) / (2.0 * _A)
                lam_init_ = max(lam_init_ * 0.95, 1e-3)
            else:
                lam_init_ = 1.0
        else:
            lam_init_ = float(lam_init)
        self.lam_init = lam_init_

        # ========================
        #     Parameter blocks
        # ========================
        if self.pedestal:
            self.extra_block = extra_block or self._default_extra_block()
        else:
            self.extra_block = extra_block or ParamBlock(
                "pedestal", [], np.empty(0), []
            )

        if scale is None:
            ped_mean0 = float(self.extra_block.init[0]) if self.pedestal else 0.0
            if self.grid.zero > 0:
                # recorded events are (mostly) >= 1 PE:
                # E[Q - ped | recorded] ~ lam * gain / occ
                occ_est = n_rec / max(_A, 1)
                scale = (mean_obs - ped_mean0) * occ_est / max(lam_init_, 1e-3)
            else:
                # whole spectrum: E[Q] = ped + lam * gain
                scale = (mean_obs - ped_mean0) / max(lam_init_, 1e-3)
            scale = max(float(scale), bin_width)
        self.scale = float(scale)

        if threshold is not None:
            self.thres_block = thres_block or self._default_thres_block()
        else:
            self.thres_block = thres_block or ParamBlock(
                "threshold", [], np.empty(0), []
            )
        self.spe_block = spe_block or self._default_spe_block()

        log_A_init = log(max(_A, 1))
        init_parts = [
            np.array([log_A_init]),
            np.asarray(self.extra_block.init, dtype=float),
            np.asarray(self.thres_block.init, dtype=float),
            np.asarray(self.spe_block.init, dtype=float),
            np.array([lam_init_]),
        ]
        bounds_parts = [
            [(log_A_init * (1 - log_A_err), log_A_init * (1 + log_A_err))],
            list(self.extra_block.bounds),
            list(self.thres_block.bounds),
            list(self.spe_block.bounds),
            [tuple(lam_bounds)],
        ]

        n_extra = len(self.extra_block.init)
        n_thres = len(self.thres_block.init)
        n_spe = len(self.spe_block.init)
        o1 = 1 + n_extra
        o2 = o1 + n_thres
        o3 = o2 + n_spe
        self.layout = {
            "log_A": slice(0, 1),
            "extra": slice(1, o1),
            "thres": slice(o1, o2),
            "spe": slice(o2, o3),
            "lam": slice(o3, o3 + 1),
        }

        self.init = np.concatenate(init_parts)
        self.bounds = sum(bounds_parts, [])
        self.dof = len(self.init)
        self.param_names = (
            ["log_A"]
            + list(self.extra_block.names)
            + list(self.thres_block.names)
            + list(self.spe_block.names)
            + ["lam"]
        )

        # ==================
        #     Likelihood
        # ==================
        ser_ft, p0_fn, count_pgf = self._model_callables()
        self._ser_ft = ser_ft
        self._p0_fn = p0_fn or (lambda spe: 0.0)
        self._count_pgf = count_pgf or poisson_pgf
        self._ft_extra = ft_gaussian_pedestal if self.pedestal else _ft_unity
        self._efficiency = make_efficiency(threshold)

        self._spectrum_fn = make_spectrum_fn(
            self.pedestal, self._ft_extra, self._ser_ft, self._count_pgf, self._p0_fn
        )

        # no-charge atom probability pgf(p0): part of the zero category in
        # the atom + below-edge scheme (pedestal=True folds the atom into
        # the continuous spectrum instead).
        if self.pedestal:
            self._atom_fn = lambda spe, lam: 0.0
        else:
            self._atom_fn = lambda spe, lam: jnp.real(
                self._count_pgf(self._p0_fn(spe) * (1.0 + 0.0j), lam, spe)
            )

        if mode == "unbinned":
            self._logl_raw = make_unbinned_logl(
                Q_raw, self.grid, self._spectrum_fn, efficiency=self._efficiency
            )
        else:
            self._logl_raw = make_binned_logl(
                self.grid, self._spectrum_fn, efficiency=self._efficiency,
                atom_fn=self._atom_fn,
            )

        self._bin_prob_fn = make_bin_prob_fn(
            self.grid, self._spectrum_fn, self._efficiency
        )

        self._logl_jit = jax.jit(self._logl_from_theta)
        self._grad_jit = jax.jit(jax.grad(self._logl_from_theta))

    # ==============================
    #     theta <-> blocks
    # ==============================

    def _unpack(self, theta):
        log_A = theta[self.layout["log_A"].start]
        extra = theta[self.layout["extra"]]
        thres = theta[self.layout["thres"]]
        spe = theta[self.layout["spe"]]
        lam = theta[self.layout["lam"].start]
        return log_A, extra, thres, spe, lam

    def _logl_from_theta(self, theta):
        log_A, extra, thres, spe, lam = self._unpack(theta)
        return self._logl_raw(log_A, extra, thres, spe, lam)

    def logl(self, theta):
        return float(self._logl_jit(jnp.asarray(theta, dtype=jnp.float64)))

    # ===========
    #     MLE
    # ===========

    def fit_mle(self, theta0=None, maxiter=500, pg_tol=0.5, multi_start=True, trace=False):
        """Maximum-likelihood fit (L-BFGS-B on JAX gradients).

        With a threshold the likelihood can be multi-modal (the truncated
        pedestal and the efficiency curve trade against each other), so a
        small set of curated starting points is tried and the best final
        likelihood wins.  Set multi_start=False (or pass theta0) to fit
        from a single start.

        L-BFGS-B can stop with an "ABNORMAL" line-search failure once the
        log-likelihood differences fall below float noise even though the
        optimum is reached; the fit is therefore also accepted when the
        scaled projected gradient max |g_j * max(|theta_j|, 1)| < pg_tol.

        With trace=True the per-iteration log-likelihood of every start
        is recorded in FitResult.trace (one array per start).
        """
        if theta0 is None:
            starts = [self.init.copy()]
            if multi_start:
                starts += self._extra_starts()
        else:
            starts = [np.asarray(theta0, dtype=np.float64)]

        # JIT warm-up
        self._logl_jit(jnp.asarray(starts[0]))
        self._grad_jit(jnp.asarray(starts[0]))

        def neg_logl(x):
            return -float(self._logl_jit(jnp.asarray(x)))

        def neg_grad(x):
            return -np.array(self._grad_jit(jnp.asarray(x)))

        res = None
        traces = []
        for start in starts:
            rec = []
            callback = None
            if trace:
                rec.append(float(self._logl_jit(jnp.asarray(start))))
                callback = lambda xk, rec=rec: rec.append(
                    float(self._logl_jit(jnp.asarray(xk)))
                )
            r = minimize(
                neg_logl,
                np.asarray(start, dtype=np.float64),
                jac=neg_grad,
                method="L-BFGS-B",
                bounds=self.bounds,
                options={"maxiter": maxiter},
                callback=callback,
            )
            traces.append(np.asarray(rec))
            if res is None or r.fun < res.fun:
                res = r

        theta_hat = np.asarray(res.x, dtype=np.float64)
        theta_err = self._hessian_errors(theta_hat)

        converged = bool(res.success)
        if not converged:
            pg = self._projected_gradient(theta_hat)
            converged = bool(
                np.max(np.abs(pg) * np.maximum(np.abs(theta_hat), 1.0)) < pg_tol
            )

        return FitResult(
            converged=converged,
            theta=theta_hat,
            theta_err=theta_err,
            logl=-float(res.fun),
            n_iter=int(res.nit),
            message=str(res.message),
            layout=self.layout,
            trace=traces if trace else None,
        )

    def _extra_starts(self):
        """Curated extra starting points for multi-modal settings.

        Only the threshold settings need them: the threshold location and
        width trade against the (truncated) pedestal shape, creating
        distinct likelihood basins.
        """
        if self.threshold is None:
            return []

        bins, hist = self.bins, self.hist
        bin_width = float(bins[1] - bins[0])
        peak_loc = float(0.5 * (bins[:-1] + bins[1:])[int(np.argmax(hist))])
        span = max(peak_loc - float(bins[0]), bin_width)

        sl_t = self.layout["thres"]
        sl_e = self.layout["extra"]
        starts = []
        for f_loc, f_scale in [(0.25, 2.0), (0.5, 4.0)]:
            t = self.init.copy()
            t[sl_t.start] = float(bins[0]) + f_loc * span
            t[sl_t.start + 1] = f_scale * bin_width
            if self.pedestal:
                t[sl_e.start + 1] = f_scale * 2.0 * bin_width
            starts.append(np.clip(t, *self._bound_arrays()))
        return starts

    def _bound_arrays(self):
        lo = np.array([-np.inf if b[0] is None else b[0] for b in self.bounds])
        hi = np.array([np.inf if b[1] is None else b[1] for b in self.bounds])
        return lo, hi

    def _projected_gradient(self, theta):
        """Gradient of -logl with active-bound components projected out."""
        g = -np.asarray(self._grad_jit(jnp.asarray(theta)))
        pg = g.copy()
        for j, (lo, hi) in enumerate(self.bounds):
            if lo is not None and theta[j] <= lo and g[j] > 0:
                pg[j] = 0.0
            if hi is not None and theta[j] >= hi and g[j] < 0:
                pg[j] = 0.0
        return pg

    def hessian_cov(self, theta):
        """Covariance matrix from the inverse Hessian of -logl."""
        H = np.asarray(jax.hessian(self._logl_from_theta)(jnp.asarray(theta)))
        return -np.linalg.pinv(H, hermitian=True)

    def hessian_cov_psd(self, theta, floor=1e-10):
        """Positive-(semi)definite covariance via the absolute-value Hessian.

        At a flat optimum the log-likelihood Hessian can pick up a tiny
        wrong-sign eigenvalue from FFT/quadrature noise, so the plain
        ``-inv(H)`` is indefinite and its diagonal can go negative (NaN
        errors).  Replacing each curvature eigenvalue ``w`` by ``-|w|``
        (covariance eigenvalue ``1/|w|``) yields a PSD covariance whose
        well-constrained directions are unchanged and whose flat
        directions get a large but finite variance.
        """
        H = np.asarray(jax.hessian(self._logl_from_theta)(jnp.asarray(theta)))
        H = 0.5 * (H + H.T)
        w, V = np.linalg.eigh(H)
        inv_abs = 1.0 / np.maximum(np.abs(w), floor)
        return (V * inv_abs) @ V.T

    def _hessian_errors(self, theta):
        try:
            cov = self.hessian_cov(theta)
            diag = np.diag(cov)
            diag = np.where(diag > 0, diag, np.nan)
            return np.sqrt(diag)
        except Exception:
            return np.full_like(theta, np.nan)

    # ==============================
    #     Expected counts & density
    # ==============================

    def estimate_bin_counts(self, theta):
        """Expected counts per histogram bin."""
        t = jnp.asarray(theta)
        log_A, extra, thres, spe, lam = self._unpack(t)
        p = self._bin_prob_fn(extra, thres, spe, lam)
        return np.asarray(jnp.exp(log_A) * p)

    def estimate_zero_count(self, theta):
        """Expected count of the zero category.

        Matches the likelihood: atom + below-edge integral when there is
        no threshold efficiency; 1 - window probability otherwise.
        """
        t = jnp.asarray(theta)
        log_A, extra, thres, spe, lam = self._unpack(t)
        A = float(np.exp(log_A))
        if self._efficiency is None:
            freq = jnp.asarray(self.grid.freq)
            G_tilde = self._spectrum_fn(freq, extra, spe, lam)
            below = _bin_integrals(
                G_tilde,
                jnp.asarray([float(self.grid.xsp[0]), float(self.grid.bins[0])]),
                freq, len(self.grid.xsp), float(self.grid.xsp_width),
            )[0]
            return float(A * max(float(self._atom_fn(spe, lam) + below), 0.0))
        p = self.estimate_bin_counts(theta) / A
        return float(A * max(1.0 - p.sum(), 0.0))

    def estimate_bin_counts_at(self, theta, edges):
        """Expected counts for arbitrary bin edges."""
        return self._counts_at(theta, edges, self._spectrum_fn)

    def estimate_density(self, theta):
        """Recorded density A * G(q) * eff(q) on the xsp grid (plotting)."""
        t = jnp.asarray(theta)
        log_A, extra, thres, spe, lam = self._unpack(t)
        freq = jnp.asarray(self.grid.freq)
        G_tilde = self._spectrum_fn(freq, extra, spe, lam)
        g = density_on_xsp(G_tilde, freq, float(self.grid.xsp[0]), self.grid.xsp_width)
        g = jnp.maximum(g, 0.0)
        if self._efficiency is not None:
            g = g * self._efficiency(jnp.asarray(self.grid.xsp), thres)
        return np.asarray(jnp.exp(log_A) * g)

    def estimate_component_counts(self, theta, n):
        """Expected counts per bin for exactly n PE."""
        return self.estimate_component_counts_at(theta, n, self.grid.bins)

    def estimate_component_counts_at(self, theta, n, edges):
        """Expected counts for exactly n PE over arbitrary bin edges."""
        spectrum_n = make_spectrum_fn(
            self.pedestal,
            self._ft_extra,
            self._ser_ft,
            self._single_n_pgf(n),
            self._p0_fn,
        )
        return self._counts_at(theta, edges, spectrum_n)

    def _counts_at(self, theta, edges, spectrum_fn):
        t = jnp.asarray(theta)
        log_A, extra, thres, spe, lam = self._unpack(t)
        freq = jnp.asarray(self.grid.freq)
        dq = float(self.grid.xsp_width)
        N = len(self.grid.xsp)
        G_tilde = spectrum_fn(freq, extra, spe, lam)
        if self._efficiency is None:
            p = _bin_integrals(G_tilde, jnp.asarray(edges), freq, N, dq)
        else:
            g = density_on_xsp(G_tilde, freq, float(self.grid.xsp[0]), dq)
            g = jnp.maximum(g, 0.0) * self._efficiency(jnp.asarray(self.grid.xsp), thres)
            p = cumulative_integrals_at(g, jnp.asarray(self.grid.xsp), jnp.asarray(edges))
        return np.asarray(jnp.exp(log_A) * p)

    # ==============================
    #     Physics summaries
    # ==============================

    def occupancy(self, theta):
        """P(non-zero charge) = 1 - pgf(p0; lam)."""
        t = jnp.asarray(theta)
        _, _, _, spe, lam = self._unpack(t)
        p0 = self._p0_fn(spe) * (1.0 + 0.0j)
        return float(1.0 - jnp.real(self._count_pgf(p0, lam, spe)))

    def _single_n_pgf(self, n):
        return make_single_n_pgf(n)

    # ==============================
    #     Default parameter blocks
    # ==============================

    def _default_extra_block(self) -> ParamBlock:
        """Data-driven Gaussian pedestal block.

        Without a threshold the pedestal is the tallest peak of the whole
        spectrum.  With a threshold the pedestal peak is largely cut away
        and only its right edge survives at the histogram left edge, so
        the initial mean is placed there (the bounds extend below it).
        """
        hist, bins = self.hist, self.bins
        bin_mid = 0.5 * (bins[:-1] + bins[1:])
        bin_width = float(bins[1] - bins[0])

        if self.threshold is not None:
            ped_mean0 = float(bins[0])
            ped_sigma0 = 2.0 * bin_width
            return ParamBlock(
                name="pedestal",
                names=["ped_mean", "ped_sigma"],
                init=np.array([ped_mean0, ped_sigma0]),
                bounds=[
                    (ped_mean0 - 10 * ped_sigma0, ped_mean0 + 10 * ped_sigma0),
                    (ped_sigma0 / 20.0, 20.0 * ped_sigma0),
                ],
            )

        i_peak = int(np.argmax(hist))
        half = hist[i_peak] / 2.0
        lo = i_peak
        while lo > 0 and hist[lo - 1] >= half:
            lo -= 1
        hi = i_peak
        while hi < len(hist) - 1 and hist[hi + 1] >= half:
            hi += 1
        w = hist[lo : hi + 1]
        x = bin_mid[lo : hi + 1]
        ped_mean0 = float(np.average(x, weights=w))
        fwhm = max((hi - lo + 1) * bin_width, bin_width)
        ped_sigma0 = max(fwhm / 2.355, 0.5 * bin_width)

        return ParamBlock(
            name="pedestal",
            names=["ped_mean", "ped_sigma"],
            init=np.array([ped_mean0, ped_sigma0]),
            bounds=[
                (ped_mean0 - 5 * ped_sigma0, ped_mean0 + 5 * ped_sigma0),
                (ped_sigma0 / 10.0, 10.0 * ped_sigma0),
            ],
        )

    def _default_thres_block(self) -> ParamBlock:
        bins, hist = self.bins, self.hist
        bin_width = float(bins[1] - bins[0])
        loc0 = float(bins[0]) + bin_width
        scale0 = 0.5 * bin_width
        peak_loc = float(0.5 * (bins[:-1] + bins[1:])[int(np.argmax(hist))])
        return ParamBlock(
            name="threshold",
            names=["thr_loc", "thr_scale"],
            init=np.array([loc0, scale0]),
            bounds=[
                (min(float(bins[0]), 0.0), max(peak_loc, loc0 + bin_width)),
                (1e-3 * bin_width, 5.0 * bin_width),
            ],
        )

    # ==============================
    #     Subclass interface
    # ==============================

    def _model_callables(self):
        """Return (ser_ft, p0_fn, count_pgf); the latter two may be None.

        ser_ft(freq, spe)      : SER characteristic function, including
                                 any discrete mass at q = 0.
        p0_fn(spe)             : discrete SER mass at q = 0 (default 0).
        count_pgf(s, lam, spe) : PE-count PGF (default Poisson).
        """
        raise NotImplementedError

    def _default_spe_block(self) -> ParamBlock:
        raise NotImplementedError

    def get_gain(self, spe, kind="gm"):
        raise NotImplementedError

    def spe_report(self, spe) -> dict:
        raise NotImplementedError
