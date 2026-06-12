# ===========================================================================
# core/lambert_w.py
#
# JAX implementation of the principal branch of the Lambert-W function,
# W(z) satisfying W(z) * exp(W(z)) = z, for real or complex input.
#
# Primal: fixed number of Halley iterations (cubic convergence; 6 steps
# reach double precision on the domain the models query: the closed disc
# |z| <= 1/e -- guaranteed by the subcriticality bound of the recursive
# model -- and the positive real axis).  A branch-point series seed keeps
# the iteration accurate near z = -1/e.
#
# The custom JVP uses the analytic derivative dW/dz = W / (z * (1 + W)),
# so gradients are exact regardless of the iteration count.
# ===========================================================================

import jax
import jax.numpy as jnp


def _initial_guess(z):
    """Principal-branch initial guess, valid on the complex plane.

    Taylor series near the origin, the branch-point expansion near
    z = -1/e (where Halley converges slowly from generic seeds),
    log(z) - log(log(z)) for large |z|, and a fixed real seed in the
    intermediate band.
    """
    z = z.astype(jnp.complex128)
    abs_z = jnp.abs(z)

    # Taylor series around z = 0 (good for |z| < ~0.3)
    series = z * (1.0 - z * (1.0 - z * (1.5 - z * (8.0 / 3.0))))

    # branch-point expansion around z = -1/e:
    # W = -1 + p - p^2/3 + 11 p^3/72, p = sqrt(2 (e z + 1))
    p = jnp.sqrt(2.0 * (jnp.e * z + 1.0))
    branch = -1.0 + p * (1.0 - p * (1.0 / 3.0 - p * (11.0 / 72.0)))

    # asymptotic guess; substitute a safe value where log would be singular
    z_safe = jnp.where(abs_z < 0.1, 1.0 + 0j, z)
    log_z = jnp.log(z_safe)
    log_z_safe = jnp.where(jnp.abs(log_z) < 0.5, 1.0 + 0j, log_z)
    asymp = log_z - jnp.log(log_z_safe)

    intermediate = jnp.full_like(z, 0.5 + 0j)

    guess = jnp.where(abs_z < 0.3, series, intermediate)
    guess = jnp.where(jnp.abs(z + jnp.exp(-1.0)) < 0.3, branch, guess)
    guess = jnp.where(abs_z > 2.0, asymp, guess)
    return guess


def _halley_step(w, z):
    """One Halley iteration for f(w) = w * exp(w) - z."""
    ew = jnp.exp(w)
    f = w * ew - z
    fp = ew * (1.0 + w)
    # f'' = ew * (2 + w), so f * f'' / (2 fp) = f * (2 + w) / (2 * (1 + w))
    denom = fp - f * (2.0 + w) / (2.0 * (1.0 + w))
    return w - f / denom


def _lambert_w_iterate(z, n_iter=6):
    z = z.astype(jnp.complex128)
    w = _initial_guess(z)

    def body(i, w):
        return _halley_step(w, z)

    return jax.lax.fori_loop(0, n_iter, body, w)


@jax.custom_jvp
def lambert_w0(z):
    """Principal-branch Lambert-W.

    Accepts real or complex input; output is always complex128 so the
    downstream Fourier arithmetic stays in complex space.
    """
    return _lambert_w_iterate(jnp.asarray(z))


@lambert_w0.defjvp
def _lambert_w0_jvp(primals, tangents):
    (z,) = primals
    (dz,) = tangents
    z = jnp.asarray(z)
    w = lambert_w0(z)
    # dW/dz = W / (z * (1 + W)); near z = 0 the limit is 1.
    safe = jnp.abs(z) < 1e-30
    z_safe = jnp.where(safe, 1.0 + 0j, z.astype(jnp.complex128))
    dw_dz = jnp.where(safe, 1.0 + 0j, w / (z_safe * (1.0 + w)))
    return w, dw_dz * jnp.asarray(dz).astype(jnp.complex128)
