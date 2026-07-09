# ===========================================================================
# tests/test_combined.py
#
# Joint fit of several spectra sharing pedestal/threshold/SER parameters,
# with one log_A and one lam per spectrum.
# ===========================================================================

import numpy as np
import pytest

from fourier_fitter import PolyaFitter, GaussFitter, CombinedFitter
from fourier_fitter.tests.conftest import sample_spectrum, spe_polya, spe_gauss


def test_combined_shared_ser():
    mean, sigma = 1.0, 0.4
    lams = [0.5, 2.0]
    fitters = []
    for i, lam in enumerate(lams):
        data = sample_spectrum(100_000, lam, spe_polya, seed=100 + i)
        hist, bins = np.histogram(data["Q"], bins=120)
        fitters.append(PolyaFitter(hist=hist, bins=bins, A=data["A"]))

    cf = CombinedFitter(fitters)
    res = cf.fit_mle()
    assert res.converged, res.message

    spe, spe_err = res.spe()
    assert abs(spe[0] - mean) < max(0.02, 5 * spe_err[0])
    assert abs(spe[1] - sigma) < max(0.02, 5 * spe_err[1])

    lam_hat, lam_err = res.lams()
    for lh, le, lt in zip(lam_hat, lam_err, lams):
        assert abs(lh - lt) < max(0.03 * lt, 5 * le)


def test_combined_local_theta_roundtrip():
    data = sample_spectrum(50_000, 1.0, spe_polya, seed=200)
    hist, bins = np.histogram(data["Q"], bins=100)
    f1 = PolyaFitter(hist=hist, bins=bins, A=data["A"])
    f2 = PolyaFitter(hist=hist, bins=bins, A=data["A"])
    cf = CombinedFitter([f1, f2])

    theta = cf.init
    for i, f in enumerate([f1, f2]):
        local = cf.local_theta(theta, i)
        assert len(local) == f.dof
        # local logl evaluates without error
        assert np.isfinite(f.logl(local))


def test_combined_rejects_mismatched_settings():
    data = sample_spectrum(20_000, 1.0, spe_polya, seed=300)
    hist, bins = np.histogram(data["Q"], bins=80)
    f1 = PolyaFitter(hist=hist, bins=bins, A=data["A"])
    f2 = PolyaFitter(hist=hist, bins=bins, A=data["A"], threshold="erf")
    with pytest.raises(ValueError):
        CombinedFitter([f1, f2])
