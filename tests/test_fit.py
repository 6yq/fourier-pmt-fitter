# ===========================================================================
# tests/test_fit.py
#
# Closure tests: toy MC generated from the model, fitted with default
# (data-driven) initial values, parameters recovered within tolerance.
# Covers the four PMT settings (pedestal on/off x threshold on/off),
# binned and unbinned modes, and the zero-mass (compound SER) case.
# ===========================================================================

import numpy as np
import pytest

from fourier_fitter import (
    GaussFitter,
    PolyaFitter,
    GammaTweedieFitter,
)
from fourier_fitter.tests.conftest import (
    N_EVENTS,
    sample_spectrum,
    spe_gauss,
    spe_polya,
    spe_gamma_tweedie,
)


def _check(res, fitter, expected, rtol=0.05, nsigma=5.0):
    """Each named parameter within rtol or nsigma of truth."""
    assert res.converged, res.message
    values = dict(zip(fitter.param_names, res.theta))
    errors = dict(zip(fitter.param_names, res.theta_err))
    for name, truth in expected.items():
        v, e = values[name], errors[name]
        tol = max(abs(truth) * rtol, nsigma * (e if np.isfinite(e) else 0.0))
        assert abs(v - truth) < tol, (
            f"{name}: fit {v:.4f} +- {e:.4f}, truth {truth:.4f}"
        )


def test_pe_spectrum_polya_binned():
    """No pedestal, no threshold."""
    lam, mean, sigma = 1.0, 1.0, 0.4
    data = sample_spectrum(N_EVENTS, lam, spe_polya)
    hist, bins = np.histogram(data["Q"], bins=150)
    f = PolyaFitter(hist=hist, bins=bins, A=data["A"])
    res = f.fit_mle()
    _check(res, f, {"spe_mean": mean, "spe_sigma": sigma, "lam": lam}, rtol=0.02)
    assert np.isclose(f.occupancy(res.theta), 1 - np.exp(-lam), atol=0.01)


def test_pe_spectrum_polya_unbinned():
    lam, mean, sigma = 1.0, 1.0, 0.4
    data = sample_spectrum(N_EVENTS, lam, spe_polya, seed=77)
    f = PolyaFitter(Q_raw=data["Q"], A=data["A"])
    assert f.mode == "unbinned"
    res = f.fit_mle()
    _check(res, f, {"spe_mean": mean, "spe_sigma": sigma, "lam": lam}, rtol=0.02)


def test_whole_spectrum_pedestal():
    """Pedestal, no threshold."""
    lam, mean, sigma = 1.2, 1.0, 0.3
    ped = (0.02, 0.08)
    data = sample_spectrum(
        N_EVENTS, lam, lambda r, n: spe_gauss(r, n, mean, sigma), ped=ped, seed=7
    )
    hist, bins = np.histogram(data["Q"], bins=200)
    f = GaussFitter(hist=hist, bins=bins, pedestal=True)
    res = f.fit_mle()
    _check(
        res,
        f,
        {
            "ped_mean": ped[0],
            "ped_sigma": ped[1],
            "spe_mean": mean,
            "spe_sigma": sigma,
            "lam": lam,
        },
        rtol=0.03,
    )


def test_pe_spectrum_threshold():
    """No pedestal, erf threshold."""
    lam, mean, sigma = 0.8, 1.0, 0.3
    thr = (0.25, 0.08)
    data = sample_spectrum(
        N_EVENTS, lam, lambda r, n: spe_gauss(r, n, mean, sigma), thr=thr, seed=11
    )
    hist, bins = np.histogram(data["Q"], bins=150)
    f = GaussFitter(hist=hist, bins=bins, A=data["A"], threshold="erf")
    res = f.fit_mle()
    _check(
        res,
        f,
        {
            "thr_loc": thr[0],
            "thr_scale": thr[1],
            "spe_mean": mean,
            "spe_sigma": sigma,
            "lam": lam,
        },
        rtol=0.05,
    )


def test_whole_spectrum_pedestal_threshold():
    """Pedestal and erf threshold together."""
    lam, mean, sigma = 0.8, 1.0, 0.3
    ped = (0.0, 0.1)
    thr = (0.25, 0.08)
    data = sample_spectrum(
        N_EVENTS,
        lam,
        lambda r, n: spe_gauss(r, n, mean, sigma),
        ped=ped,
        thr=thr,
        seed=13,
    )
    hist, bins = np.histogram(data["Q"], bins=150)
    f = GaussFitter(
        hist=hist, bins=bins, A=data["A"], pedestal=True, threshold="erf"
    )
    res = f.fit_mle()
    _check(
        res,
        f,
        {
            "ped_sigma": ped[1],
            "thr_loc": thr[0],
            "thr_scale": thr[1],
            "spe_mean": mean,
            "spe_sigma": sigma,
            "lam": lam,
        },
        rtol=0.05,
    )
    # ped_mean is close to zero: check absolutely
    assert abs(res.theta[f.layout["extra"].start] - ped[0]) < 0.02


def test_gamma_tweedie_zero_mass():
    """Compound SER with a discrete mass at zero charge."""
    truth = dict(frac=0.6, mean=0.6, sigma=0.25, lam_c=5.0, mean_t=0.6, sigma_t=0.2)
    lam = 1.2
    data = sample_spectrum(
        N_EVENTS,
        lam,
        lambda r, n: spe_gamma_tweedie(r, n, **truth),
        seed=5,
    )
    hist, bins = np.histogram(data["Q"], bins=150)
    f = GammaTweedieFitter(hist=hist, bins=bins, A=data["A"])
    res = f.fit_mle()
    _check(
        res,
        f,
        {
            "frac": truth["frac"],
            "spe_mean": truth["mean"],
            "spe_sigma": truth["sigma"],
            "lam": lam,
        },
        rtol=0.05,
    )
    # the occupancy (P(non-zero charge), including the SER zero mass)
    occ_true = 1.0 - data["zero"] / data["A"]
    assert np.isclose(f.occupancy(res.theta), occ_true, atol=0.005)


def test_estimators_consistent():
    """Bin counts, density and components integrate consistently."""
    lam = 1.0
    data = sample_spectrum(100_000, lam, spe_polya, seed=3)
    hist, bins = np.histogram(data["Q"], bins=120)
    f = PolyaFitter(hist=hist, bins=bins, A=data["A"])
    res = f.fit_mle()

    y = f.estimate_bin_counts(res.theta)
    # zero category = atom + below-edge integral: the above-window tail is
    # excluded by construction, so the identity holds only up to that tail
    # (negligible for this toy spectrum).
    assert np.isclose(y.sum() + f.estimate_zero_count(res.theta),
                      np.exp(res.theta[0]), rtol=1e-4)

    # n-PE components sum to the total within the window
    comp = sum(f.estimate_component_counts(res.theta, n) for n in range(1, 9))
    assert np.allclose(comp, y, rtol=1e-3, atol=0.5)

    # arbitrary-edge integration agrees with the native bins
    y2 = f.estimate_bin_counts_at(res.theta, bins)
    assert np.allclose(y, y2, rtol=1e-9)
