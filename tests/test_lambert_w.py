import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import lambertw as scipy_lambertw

from fourier_fitter.core.lambert_w import lambert_w0


def test_identity_on_physical_domain():
    """The recursive model only queries |z| <= 1/e (subcriticality bound):
    arg = x e^x * e^{lam_r frac (ft_r - 1)} with x = lam_r (frac - 1) in
    (-1, 0) and |e^{...}| <= 1, so |arg| <= max |x e^x| = 1/e.
    """
    rng = np.random.default_rng(1)
    r = rng.uniform(0.0, 1.0 / np.e, size=5000)
    phi = rng.uniform(-np.pi, np.pi, size=5000)
    z = r * np.exp(1j * phi)
    w = np.asarray(lambert_w0(jnp.asarray(z)))
    assert np.max(np.abs(w * np.exp(w) - z)) < 1e-12


def test_identity_near_branch_point():
    rng = np.random.default_rng(4)
    t = rng.uniform(0, 0.05, size=2000)
    ph = rng.uniform(-np.pi, np.pi, size=2000)
    z = -1.0 / np.e + t * np.exp(1j * ph)
    z = z[np.abs(z) <= 1.0 / np.e]
    w = np.asarray(lambert_w0(jnp.asarray(z)))
    assert np.max(np.abs(w * np.exp(w) - z)) < 1e-12


def test_identity_positive_reals():
    z = np.linspace(0.0, 20.0, 2000) + 0j
    w = np.asarray(lambert_w0(jnp.asarray(z)))
    assert np.max(np.abs(w * np.exp(w) - z)) < 1e-12


def test_matches_scipy_principal_branch():
    rng = np.random.default_rng(2)
    z = rng.uniform(-0.3, 5.0, size=200) + 1j * rng.uniform(-1.0, 1.0, size=200)
    w = np.asarray(lambert_w0(jnp.asarray(z)))
    w_ref = scipy_lambertw(z, k=0)
    assert np.allclose(w, w_ref, rtol=1e-9, atol=1e-9)


def test_recursive_model_argument_range():
    """Arguments produced by the recursive Polya model stay on branch 0."""
    # arg = lam_r * (frac - 1) * exp(lam_r * (frac * ft_r - 1)), |ft_r| <= 1
    for frac, lam_r in [(0.05, 0.99), (0.4, 0.8), (0.9, 0.5)]:
        ft = np.exp(1j * np.linspace(0, 2 * np.pi, 50)) * np.linspace(0, 1, 50)
        arg = lam_r * (frac - 1.0) * np.exp(lam_r * (frac * ft - 1.0))
        w = np.asarray(lambert_w0(jnp.asarray(arg)))
        assert np.all(np.isfinite(w))
        assert np.allclose(w * np.exp(w), arg, rtol=1e-8, atol=1e-10)


def test_gradient_analytic():
    def f(x):
        return jnp.real(lambert_w0(x + 0j))

    x0 = 0.7
    g = jax.grad(f)(x0)
    eps = 1e-6
    g_fd = (f(x0 + eps) - f(x0 - eps)) / (2 * eps)
    assert np.isclose(float(g), float(g_fd), rtol=1e-6)
