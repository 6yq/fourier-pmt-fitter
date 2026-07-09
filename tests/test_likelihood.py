# ===========================================================================
# tests/test_likelihood.py
#
# Numerical checks of the likelihood building blocks:
#   - analytic Fourier bin integrals against the Simpson quadrature path,
#   - density normalization and the zero-category probability,
#   - binned log-likelihood maximum close to the truth.
# ===========================================================================

import numpy as np
import jax.numpy as jnp

from fourier_fitter.core.fft_grid import build_grid
from fourier_fitter.core.likelihood import (
    poisson_pgf,
    make_spectrum_fn,
    make_bin_prob_fn,
    density_on_xsp,
    _bin_integrals,
)
from fourier_fitter.models.polya import _ser_ft_polya
from fourier_fitter.tests.conftest import sample_spectrum, spe_polya


def _polya_grid(sample=1):
    data = sample_spectrum(50_000, lam=1.0, spe_sampler=spe_polya)
    hist, bins = np.histogram(data["Q"], bins=100)
    return build_grid(hist, bins, A=data["A"], sample=sample), data


def test_analytic_vs_simpson_quadrature():
    """The two bin-integration paths must agree without a threshold."""
    grid, _ = _polya_grid(sample=8)
    spectrum = make_spectrum_fn(
        False, None, _ser_ft_polya, poisson_pgf, lambda spe: 0.0
    )
    empty = jnp.empty(0)
    spe = jnp.array([1.0, 0.4])
    lam = 1.0

    analytic = make_bin_prob_fn(grid, spectrum, None)
    unit_eff = make_bin_prob_fn(grid, spectrum, lambda q, t: jnp.ones_like(q))

    p1 = np.asarray(analytic(empty, empty, spe, lam))
    p2 = np.asarray(unit_eff(empty, empty, spe, lam))
    # Simpson with sample=8 carries O(h^4) discretization error
    assert np.allclose(p1, p2, rtol=1e-4, atol=1e-9)


def test_density_normalization_and_zero_prob():
    grid, _ = _polya_grid()
    spectrum = make_spectrum_fn(
        False, None, _ser_ft_polya, poisson_pgf, lambda spe: 0.0
    )
    freq = jnp.asarray(grid.freq)
    spe = jnp.array([1.0, 0.4])
    lam = 1.0

    G = spectrum(freq, jnp.empty(0), spe, lam)
    # DC coefficient = integral of the continuous part = 1 - exp(-lam)
    assert np.isclose(float(jnp.real(G[0])), 1.0 - np.exp(-lam), atol=1e-10)

    # grid-space density integrates to the same value
    g = density_on_xsp(G, freq, float(grid.xsp[0]), grid.xsp_width)
    total = float(jnp.trapezoid(g, dx=grid.xsp_width))
    assert np.isclose(total, 1.0 - np.exp(-lam), atol=1e-6)

    # bin integrals: 1 - sum = zero-category probability ~ exp(-lam) + tails
    p = np.asarray(
        _bin_integrals(G, jnp.asarray(grid.bins), freq, len(grid.xsp), grid.xsp_width)
    )
    z = 1.0 - p.sum()
    assert z >= np.exp(-lam) - 1e-6
    assert np.isclose(z, np.exp(-lam), atol=5e-3)


def test_binned_counts_match_toy_histogram():
    grid, data = _polya_grid()
    spectrum = make_spectrum_fn(
        False, None, _ser_ft_polya, poisson_pgf, lambda spe: 0.0
    )
    bin_probs = make_bin_prob_fn(grid, spectrum, None)
    p = np.asarray(bin_probs(jnp.empty(0), jnp.empty(0), jnp.array([1.0, 0.4]), 1.0))
    y = data["A"] * p

    # chi-square per dof against the toy histogram should be ~1
    mask = y > 10
    chi2 = np.sum((grid.hist[mask] - y[mask]) ** 2 / y[mask])
    ndf = int(mask.sum())
    assert chi2 / ndf < 1.5
