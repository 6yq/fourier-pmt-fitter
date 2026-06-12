# ===========================================================================
# core/likelihood.py
#
# JAX log-likelihood for PMT charge spectra, with analytic Fourier-domain
# bin integration and explicit handling of the four PMT settings
# (pedestal on/off x threshold on/off).
#
# Model
# -----
# The number of photoelectrons is Poisson(lam) (the count PGF is
# injectable, Poisson by default).  The single-PE charge has the
# characteristic function f~(w) = ser_ft(freq, spe), which may include a
# discrete mass p0 = p0_fn(spe) at q = 0 (compound SER models).  In
# Fourier space the total-charge spectrum is
#
#     pgf( f~(w); lam ) = exp( lam * (f~(w) - 1) )          (Poisson)
#
# With a pedestal (whole spectrum, every trigger is recorded):
#
#     G~(w) = g0~(w) * pgf( f~(w) )
#
# Without a pedestal (PE spectrum: only events with non-zero charge are
# histogrammed), the discrete mass at q = 0 must be removed from the
# continuous density.  Total charge is exactly zero when every PE leaves
# zero charge, with probability pgf(p0) (the n = 0 term gives exp(-lam)):
#
#     G~_c(w) = pgf( f~(w) ) - pgf( p0 )
#
# A hardware threshold multiplies the recorded density by an efficiency
# eff(q; thres) in charge space.
#
# Zero category
# -------------
# All four settings share one bookkeeping rule: with A total triggers and
# the observed density d(q) (G or G_c, times eff if any), the probability
# of landing in the histogram window is P_obs and the "zero" category
# (undetected, below threshold, or out of window) has probability
# 1 - P_obs.  The extended-Poisson log-likelihood is then
#
#     log L = sum_k [ n_k log(A y_k) - A y_k ]
#           + n_0 log(A (1 - P_obs)) - A (1 - P_obs) - log C
#
# Bin integration
# ---------------
# Without a threshold the bin integrals of G(q) are computed analytically
# in Fourier space.  Given G~_n on a uniform grid of length N and spacing
# dq (period L = N * dq), the antiderivative of the inverse-DFT
# reconstruction is
#
#     I(q) = (G~_0 / L) * q + (1/L) * sum_{n!=0} (G~_n / (i w_n)) e^{+i w_n q}
#
# so each bin integral is I(b_{k+1}) - I(b_k), one matrix product over all
# edges.  With a threshold the product G(q) * eff(q) is no longer analytic
# in Fourier space; the density is evaluated by IFFT on the sub-sampled
# grid (aligned with the bin edges) and integrated per bin with a
# composite Simpson rule.
# ===========================================================================

import numpy as np
import jax
import jax.numpy as jnp


# ==============================
#     Spectrum assembly
# ==============================


def poisson_pgf(s, lam, spe):
    """Poisson count PGF evaluated at (possibly complex) s."""
    return jnp.exp(lam * (s - 1.0))


def make_single_n_pgf(n):
    """Return a count_pgf-compatible closure p_n(lam) * s^n for Poisson."""
    log_fact = float(np.sum(np.log(np.arange(1, n + 1))))

    def pgf(s, lam, spe):
        log_p = n * jnp.log(lam) - lam - log_fact
        return jnp.exp(log_p) * s**n

    return pgf


def make_spectrum_fn(pedestal, ft_extra, ser_ft, count_pgf, p0_fn):
    """Return spectrum(freq, extra, spe, lam) -> G~ on the frequency grid.

    pedestal = True : G~ = g0~ * pgf(f~)
    pedestal = False: G~ = pgf(f~) - pgf(p0)   (continuous part only)
    """
    if pedestal:

        def spectrum(freq, extra, spe, lam):
            s = ser_ft(freq, spe)
            return count_pgf(s, lam, spe) * ft_extra(freq, extra)

    else:

        def spectrum(freq, extra, spe, lam):
            s = ser_ft(freq, spe)
            p0 = p0_fn(spe) * (1.0 + 0.0j)
            return count_pgf(s, lam, spe) - count_pgf(p0, lam, spe)

    return spectrum


# ==============================
#     Threshold efficiency
# ==============================


def make_efficiency(kind):
    """Return eff(q, thres) with thres = (location, scale), or None."""
    if kind is None:
        return None
    if kind == "erf":
        sqrt2 = float(np.sqrt(2.0))

        def eff(q, thres):
            return 0.5 * (1.0 + jax.scipy.special.erf((q - thres[0]) / (thres[1] * sqrt2)))

        return eff
    if kind == "logistic":

        def eff(q, thres):
            return jax.nn.sigmoid((q - thres[0]) / thres[1])

        return eff
    raise ValueError(f"Unknown threshold kind: {kind!r}")


# ==============================
#     Analytic bin integrals
# ==============================


def _bin_integrals(G_tilde, edges, freq, N, dq):
    """Return integral_{edges[k]}^{edges[k+1]} G(q) dq for each k."""
    L = N * dq
    inv_L = 1.0 / L

    # AC part: coef_n = G~_n / (i w_n) for n != 0; coef_0 := 0.
    safe_w = jnp.where(freq == 0.0, 1.0, freq)
    coef = G_tilde / (1j * safe_w)
    coef = coef.at[0].set(0.0 + 0.0j)

    phase = jnp.exp(1j * jnp.outer(edges, freq))  # (n_edges, N)

    I_ac = (phase @ coef) * inv_L
    I_dc = (G_tilde[0] * inv_L) * edges
    I = jnp.real(I_ac + I_dc)

    return I[1:] - I[:-1]


# ==============================
#     Grid-space density
# ==============================


def density_on_xsp(G_tilde, freq, q_lo, dq):
    """Evaluate G(q) on the uniform grid xsp[n] = q_lo + n * dq.

    The grid origin is arbitrary: the offset enters as an explicit phase
    factor before the IFFT, so q = 0 does not need to lie on the grid.
    """
    shifted = G_tilde * jnp.exp(1j * freq * q_lo)
    return jnp.real(jnp.fft.ifft(shifted)) / dq


def _simpson_weights(sample, dq):
    """Composite Simpson weights over one bin split into `sample` steps."""
    if sample == 1:
        return np.array([0.5, 0.5]) * dq
    w = np.ones(sample + 1)
    w[1:-1:2] = 4.0
    w[2:-2:2] = 2.0
    return w * dq / 3.0


def cumulative_integrals_at(g_xsp, xsp, edges):
    """Integrals of a grid-sampled density over arbitrary bin edges.

    Trapezoidal cumulative integral on xsp, interpolated at the edges.
    Used by the diagnostics accessors when a threshold efficiency makes
    the analytic Fourier integration unavailable.
    """
    dq = xsp[1] - xsp[0]
    seg = 0.5 * (g_xsp[1:] + g_xsp[:-1]) * dq
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    I = jnp.interp(edges, xsp, cum)
    return I[1:] - I[:-1]


# ==============================
#     Binned log-likelihood
# ==============================


def make_bin_prob_fn(grid, spectrum_fn, efficiency):
    """Return bin_probs(extra, thres, spe, lam) -> per-bin probabilities.

    Uses the analytic Fourier bin integrals when there is no threshold,
    and the aligned-grid Simpson quadrature otherwise.
    """
    freq = jnp.asarray(grid.freq)
    edges = jnp.asarray(grid.bins)
    dq = float(grid.xsp_width)
    N = len(grid.xsp)
    q_lo = float(grid.xsp[0])
    xsp = jnp.asarray(grid.xsp)
    nbin = len(grid.hist)
    sample = int(grid.sample)

    if efficiency is None:

        def bin_probs(extra, thres, spe, lam):
            G_tilde = spectrum_fn(freq, extra, spe, lam)
            return _bin_integrals(G_tilde, edges, freq, N, dq)

        return bin_probs

    simp_w = jnp.asarray(_simpson_weights(sample, dq))
    idx = (
        grid.i_edge0
        + sample * np.arange(nbin)[:, None]
        + np.arange(sample + 1)[None, :]
    )
    idx_j = jnp.asarray(idx)

    def bin_probs(extra, thres, spe, lam):
        G_tilde = spectrum_fn(freq, extra, spe, lam)
        g = density_on_xsp(G_tilde, freq, q_lo, dq)
        g = jnp.maximum(g, 0.0) * efficiency(xsp, thres)
        return g[idx_j] @ simp_w

    return bin_probs


def make_binned_logl(grid, spectrum_fn, efficiency=None):
    """Build a binned extended-Poisson log-likelihood closure.

    Returns logl(log_A, extra, thres, spe, lam) -> scalar.  ``extra`` and
    ``thres`` may be empty arrays when the corresponding setting is off.
    """
    hist = jnp.asarray(grid.hist)
    zero = float(grid.zero)
    log_C = float(grid.log_C)
    bin_probs = make_bin_prob_fn(grid, spectrum_fn, efficiency)

    def logl(log_A, extra, thres, spe, lam):
        A = jnp.exp(log_A)
        p_bins = bin_probs(extra, thres, spe, lam)
        y_est = jnp.maximum(A * p_bins, 1e-32)
        z_prob = jnp.maximum(1.0 - jnp.sum(p_bins), 1e-32)
        z_est = A * z_prob

        ll_bins = jnp.sum(hist * jnp.log(y_est) - y_est)
        ll_zero = zero * jnp.log(z_est) - z_est
        return ll_bins + ll_zero - log_C

    return logl


# ==============================
#     Unbinned log-likelihood
# ==============================


def _density_at_q(G_tilde, Q, N, dq):
    """Evaluate G(q) at arbitrary positions via type-2 NUFFT.

    G(q) = (1/L) * sum_n G~_n * exp(+i w_n q),  w_n = 2 pi n / L.
    jax_finufft expects modes in fftshift order and x in [-pi, pi];
    the reconstruction is L-periodic so folding is harmless.
    """
    import jax_finufft

    L = N * dq
    x = (2.0 * jnp.pi / L) * Q
    vals = jax_finufft.nufft2(
        jnp.fft.fftshift(G_tilde).astype(jnp.complex128), x, iflag=1
    )
    return jnp.real(vals) / L


def make_unbinned_logl(Q_raw, grid, spectrum_fn, efficiency=None):
    """Build an unbinned extended-Poisson log-likelihood closure.

    The density is evaluated at each observed charge via NUFFT, so the
    shape term needs no binning.  The zero category (undetected events,
    zero = A - N_obs) is kept, generalizing the plain extended form:

        log L = N_obs log A + sum_i log d(Q_i) - A P_obs
              + zero log(A (1 - P_obs)) - A (1 - P_obs) - log(zero!)
    """
    freq = jnp.asarray(grid.freq)
    dq = float(grid.xsp_width)
    N = len(grid.xsp)
    q_lo = float(grid.xsp[0])
    xsp = jnp.asarray(grid.xsp)
    zero = float(grid.zero)
    log_C0 = float(np.sum(np.log(np.arange(1, grid.zero + 1))))

    Q = jnp.asarray(Q_raw, dtype=jnp.float64)
    N_obs = int(Q.shape[0])

    def logl(log_A, extra, thres, spe, lam):
        A = jnp.exp(log_A)
        G_tilde = spectrum_fn(freq, extra, spe, lam)
        d = _density_at_q(G_tilde, Q, N, dq)

        if efficiency is None:
            p_obs = jnp.real(G_tilde[0])
        else:
            d = d * efficiency(Q, thres)
            g = jnp.maximum(density_on_xsp(G_tilde, freq, q_lo, dq), 0.0)
            p_obs = jnp.trapezoid(g * efficiency(xsp, thres), dx=dq)

        d = jnp.maximum(d, 1e-300)
        z_prob = jnp.maximum(1.0 - p_obs, 1e-32)

        ll = N_obs * log_A + jnp.sum(jnp.log(d)) - A * p_obs
        ll = ll + zero * jnp.log(A * z_prob) - A * z_prob
        return ll - log_C0

    return logl
