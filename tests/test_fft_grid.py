import numpy as np
import pytest

from fourier_fitter.core.fft_grid import build_grid


def _toy_hist():
    rng = np.random.default_rng(0)
    q = rng.gamma(2.0, 0.5, size=10000)
    return np.histogram(q, bins=50)


def test_edges_on_grid():
    hist, bins = _toy_hist()
    for sample in (1, 2, 8):
        g = build_grid(hist, bins, sample=sample)
        # all bin edges coincide with grid points
        i = g.i_edge0 + sample * np.arange(len(hist) + 1)
        assert np.allclose(g.xsp[i], bins, atol=1e-9 * (bins[-1] - bins[0]))


def test_grid_covers_zero_and_margins():
    hist, bins = _toy_hist()
    g = build_grid(hist, bins)
    bw = bins[1] - bins[0]
    assert g.xsp[0] <= min(0.0, bins[0] - 5 * bw) + 1e-12
    assert g.xsp[-1] >= bins[-1] + 5 * bw - 1e-12


def test_q_min_q_max_extend():
    hist, bins = _toy_hist()
    g = build_grid(hist, bins, q_min=-30.0, q_max=60.0)
    assert g.xsp[0] <= -30.0 + 1e-12
    assert g.xsp[-1] >= 60.0 - 1e-12


def test_zero_category():
    hist, bins = _toy_hist()
    g = build_grid(hist, bins, A=int(hist.sum()) + 123)
    assert g.zero == 123
    with pytest.raises(ValueError):
        build_grid(hist, bins, A=int(hist.sum()) - 1)


def test_odd_sample_rejected():
    hist, bins = _toy_hist()
    with pytest.raises(ValueError):
        build_grid(hist, bins, sample=3)


def test_freq_matches_spacing():
    hist, bins = _toy_hist()
    g = build_grid(hist, bins, sample=2)
    N = len(g.xsp)
    assert np.allclose(g.freq, 2 * np.pi * np.fft.fftfreq(N, d=g.xsp_width))
