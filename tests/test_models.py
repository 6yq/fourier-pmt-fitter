# ===========================================================================
# tests/test_models.py
#
# Model-level checks, independent of the fit machinery:
#   - characteristic function normalization f~(0) = 1,
#   - f~(w) against the empirical characteristic function of the
#     matching toy MC sampler (validates sign convention and algebra),
#   - p0 (discrete mass at zero) against the MC zero fraction,
#   - gain formulas against MC moments.
# ===========================================================================

import numpy as np
import jax.numpy as jnp
import pytest

from fourier_fitter.models import (
    GaussFitter,
    BiGaussFitter,
    LinearGaussFitter,
    TriGaussFitter,
    GaussCompoundFitter,
    PolyaFitter,
    BiPolyaFitter,
    PolyaExpFitter,
    GammaTweedieFitter,
    RecursivePolyaFitter,
)
from fourier_fitter.tests.conftest import (
    spe_gauss,
    spe_polya,
    spe_gamma_tweedie,
    spe_recursive_polya,
)

ALL_MODELS = [
    GaussFitter,
    BiGaussFitter,
    LinearGaussFitter,
    TriGaussFitter,
    GaussCompoundFitter,
    PolyaFitter,
    BiPolyaFitter,
    PolyaExpFitter,
    GammaTweedieFitter,
    RecursivePolyaFitter,
]


def _make_fitter(cls, **kw):
    rng = np.random.default_rng(3)
    q = rng.gamma(4.0, 0.25, size=5000)
    hist, bins = np.histogram(q, bins=60)
    return cls(hist=hist, bins=bins, A=8000, **kw)


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_char_fn_normalization(cls):
    f = _make_fitter(cls)
    spe = jnp.asarray(f.spe_block.init)
    val = f._ser_ft(jnp.zeros(1), spe)
    assert np.allclose(np.asarray(val), 1.0, atol=1e-10)
    p0 = float(np.real(np.asarray(f._p0_fn(spe))))
    assert 0.0 <= p0 < 1.0


@pytest.mark.parametrize("cls", ALL_MODELS)
def test_default_init_within_bounds(cls):
    f = _make_fitter(cls)
    for v, (lo, hi) in zip(f.init, f.bounds):
        if lo is not None:
            assert v >= lo - 1e-12
        if hi is not None:
            assert v <= hi + 1e-12


def _empirical_char(Q, w):
    return np.array([np.mean(np.exp(-1j * wi * Q)) for wi in w])


@pytest.mark.parametrize(
    "cls,sampler,spe",
    [
        (GaussFitter, lambda r, n: spe_gauss(r, n, 1.0, 0.3), [1.0, 0.3]),
        (PolyaFitter, lambda r, n: spe_polya(r, n, 1.0, 0.4), [1.0, 0.4]),
        (
            GammaTweedieFitter,
            lambda r, n: spe_gamma_tweedie(r, n, 0.6, 0.6, 0.25, 5.0, 0.6, 0.2),
            [0.6, 0.6, 0.25, 5.0, 0.6, 0.2],
        ),
        (
            RecursivePolyaFitter,
            lambda r, n: spe_recursive_polya(r, n, 0.4, 0.6, 0.25, 4.5, 0.8, 0.6, 0.2),
            [0.4, 0.6, 0.25, 4.5, 0.8, 0.6, 0.2],
        ),
    ],
)
def test_char_fn_against_mc(cls, sampler, spe):
    rng = np.random.default_rng(42)
    Q = sampler(rng, 400_000)
    w = np.array([0.3, 0.7, 1.5, 3.0])

    f = _make_fitter(cls)
    model = np.asarray(f._ser_ft(jnp.asarray(w), jnp.asarray(spe, dtype=jnp.float64)))
    emp = _empirical_char(Q, w)
    assert np.allclose(model, emp, atol=0.01), f"{model} vs {emp}"

    # discrete mass at zero
    p0 = float(np.real(np.asarray(f._p0_fn(jnp.asarray(spe, dtype=jnp.float64)))))
    assert np.isclose(p0, np.mean(Q == 0.0), atol=5e-3)

    # mean gain conditional on non-zero charge
    gm = f.get_gain(np.asarray(spe), "gm")
    assert np.isclose(gm, Q[Q != 0].mean(), rtol=0.02)


def test_trigauss_weights_simplex():
    from fourier_fitter.models.gauss import _trigauss_weights

    for v1, v2 in [(0.0, 0.0), (1.0, 1.0), (0.3, 0.7), (0.9, 0.1)]:
        w = _trigauss_weights(v1, v2)
        assert np.isclose(sum(w), 1.0)
        assert all(wi >= -1e-12 for wi in w)
