# ===========================================================================
# core/fft_grid.py
#
# Static uniform FFT grid for one PMT charge spectrum.
#
# The spectrum density G(q) is represented by its Fourier coefficients
# G_tilde sampled on a uniform frequency grid.  Reconstruction at any q is
#
#     G(q) = (1/L) * sum_n G_tilde_n * exp(+i * w_n * q),     L = N * dq
#
# so the grid only needs to
#   1. be uniform in q (spacing dq = bin_width / sample),
#   2. cover the full physical support of G so the periodic image of the
#      IFFT does not wrap into the fit window,
#   3. have its points aligned with the histogram bin edges, so that the
#      threshold-efficiency quadrature (Simpson per bin) can index the
#      sub-sampled density directly.
#
# Alignment is achieved by extending the grid from bins[0] downwards and
# from bins[-1] upwards by integer multiples of bin_width.  The grid origin
# does not need to contain q = 0: densities at xsp are obtained from the
# Fourier coefficients with an explicit phase shift (see likelihood.py).
# ===========================================================================

import numpy as np

from dataclasses import dataclass


# ==============================
#     Grid container
# ==============================


@dataclass(frozen=True)
class FFTGrid:
    bins: np.ndarray  # histogram bin edges, len = nbin + 1
    hist: np.ndarray  # bin counts, len = nbin
    A: int  # total events (including the zero category)
    zero: int  # A - sum(hist)
    xsp: np.ndarray  # uniform grid for FFT, len = N
    xsp_width: float  # grid spacing dq = bin_width / sample
    sample: int  # sub-sampling factor (points per bin = sample + 1)
    i_edge0: int  # index of bins[0] in xsp
    freq: np.ndarray  # angular frequencies, len = N
    log_C: float  # log-factorial constant for the Poisson log-L


# ==============================
#     Builder
# ==============================


def build_grid(hist, bins, A=None, q_min=None, q_max=None, sample=1):
    """Build the uniform FFT grid for one spectrum.

    Parameters
    ----------
    hist, bins : array-like
        Bin counts and edges (uniform bins required).
    A : int or None
        Total events including the zero category.  Defaults to sum(hist).
    q_min : float or None
        Requested left extent.  The grid is extended from bins[0] downwards
        by whole bins until it covers min(q_min, 0, bins[0] - 5 * bin_width).
    q_max : float or None
        Requested right extent.  Extended from bins[-1] upwards by whole
        bins until it covers max(q_max, 0, bins[-1] + 5 * bin_width).
    sample : int
        Sub-sampling factor: dq = bin_width / sample.  Must be even (or 1)
        so the per-bin composite Simpson rule applies.
    """
    hist = np.asarray(hist, dtype=float)
    bins = np.asarray(bins, dtype=float)
    A = int(A) if A is not None else int(hist.sum())
    zero = A - int(hist.sum())
    if zero < 0:
        raise ValueError(f"A = {A} smaller than sum(hist) = {int(hist.sum())}.")

    widths = np.diff(bins)
    bin_width = float(widths[0])
    if not np.allclose(widths, bin_width, rtol=1e-6):
        raise ValueError("build_grid requires uniform histogram bins.")

    sample = int(sample)
    if sample != 1 and sample % 2 != 0:
        raise ValueError("sample must be 1 or an even integer (Simpson rule).")
    dq = bin_width / sample

    # ==============================
    #     Aligned grid extent
    # ==============================
    lo_target = min(bins[0] - 5 * bin_width, 0.0)
    if q_min is not None:
        lo_target = min(lo_target, float(q_min))
    hi_target = max(bins[-1] + 5 * bin_width, 0.0)
    if q_max is not None:
        hi_target = max(hi_target, float(q_max))

    n_below = int(np.ceil((bins[0] - lo_target) / bin_width))
    n_above = int(np.ceil((hi_target - bins[-1]) / bin_width))

    q_lo = bins[0] - n_below * bin_width
    n_bins_total = n_below + len(hist) + n_above
    N = n_bins_total * sample + 1

    xsp = q_lo + np.arange(N) * dq
    i_edge0 = n_below * sample

    freq = 2.0 * np.pi * np.fft.fftfreq(N, d=dq)

    # log-factorial constant for the extended-Poisson log-likelihood
    log_C = float(
        sum(np.sum(np.log(np.arange(1, int(n) + 1))) for n in hist)
        + np.sum(np.log(np.arange(1, zero + 1)))
    )

    return FFTGrid(
        bins=bins,
        hist=hist,
        A=A,
        zero=zero,
        xsp=xsp,
        xsp_width=dq,
        sample=sample,
        i_edge0=i_edge0,
        freq=freq,
        log_C=log_C,
    )
