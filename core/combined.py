# core/combined.py
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional
from contextlib import contextmanager

from scipy.fft import fft
from .fft_utils import roll_and_pad


@dataclass
class Spec:
    """Specification for a single spectrum included in the combined fit.

    Parameters
    ----------
    fitter : object
        An instance of your PMT_Fitter (or subclass). It must expose:
          - .init (np.ndarray)
          - .bounds (tuple of (lo, hi))
          - .log_l(args: np.ndarray) -> float
          - ._start_idx  (0 if no pedestal/threshold; 2 otherwise)
          - ._ser_pdf_time(args)      (time-domain SER PDF on .xsp)
          - grid attributes: .xsp, ._xsp_width, ._pad_safe, ._shift
    share_ser : bool, default True
        If True, this spectrum uses the shared [additional + SER] parameter block.
        If False, this spectrum has its own local SER parameters (rare).
    use_scale_axis : bool, default False
        If True, add a per-spectrum axis scaling s = exp(log_s) applied to the
        *shared* SER shape via p_s(x) = (1/s) * p(x/s). This is robust across
        PMT types because it does not depend on any particular "gain index".
    weight : float, default 1.0
        Optional weight in the summed log-likelihood.
    """

    fitter: object
    share_ser: bool = True
    use_scale_axis: bool = False
    weight: float = 1.0


class CombinedFitter:
    """Combine multiple spectra (different classes / bins) into a joint fit.

    Global parameter vector layout:
        theta = [ shared_additional..., shared_SER...,  (log_s1?) occ1,  (log_s2?) occ2, ...]
    where:
        - shared_additional are pedestal/threshold params (length = _start_idx: 0 or 2)
        - shared_SER are SER params (excluding occupancy)
        - each spectrum optionally adds log_s if use_scale_axis=True (s = exp(log_s))
        - each spectrum has its own occupancy occ in (0, 1)

    Notes
    -----
    1) All spectra must agree on pedestal/threshold usage: identical _start_idx.
    2) For share_ser=True spectra, SER dimensionality must be identical to be shareable.
    3) Axis scaling is implemented by *time-domain resampling*:
           p_s(x) = (1/s) * p(x/s),
       evaluated on the spectrum's existing .xsp grid, then fed into your
       standard FFT pipeline (roll_and_pad -> FFT). No need to change ._freq or ._xsp_width.
    """

    # -------------------------
    #      Initialization
    # -------------------------
    def __init__(
        self,
        specs: List[Spec],
        ref_idx: int = 0,
        init_override: Optional[np.ndarray] = None,
        bounds_override: Optional[
            Sequence[Tuple[Optional[float], Optional[float]]]
        ] = None,
    ):
        self.specs = specs
        self.ref_idx = ref_idx
        self.init_override = init_override
        self.bounds_override = bounds_override

        if not specs:
            raise ValueError("At least one spectrum (Spec) is required.")

        # 1) Enforce identical pedestal/threshold usage across spectra
        start_idxs = [sp.fitter._start_idx for sp in specs]
        if len(set(start_idxs)) != 1:
            raise ValueError(
                "All spectra must have identical _start_idx (pedestal/threshold usage)."
            )
        self._addl_len = start_idxs[0]  # 0 or 2

        # 2) SER dimensionality and shareability
        ser_dims = [len(sp.fitter.init[sp.fitter._start_idx : -1]) for sp in specs]
        self._ser_dims = ser_dims

        shared_dims = {d for d, sp in zip(ser_dims, specs) if sp.share_ser}
        if len(shared_dims) > 1:
            raise ValueError(
                "Shared SER requires identical SER dimension for all share_ser=True spectra."
            )
        self._shared_ser_dim = shared_dims.pop() if shared_dims else 0

        # 3) Build shared [additional + SER] block from reference spectrum (or overrides)
        ref = specs[ref_idx].fitter
        shared_addl0 = ref.init[: self._addl_len] if self._addl_len else np.array([])
        shared_addl_b = list(ref.bounds[: self._addl_len]) if self._addl_len else []

        shared_ser0 = (
            ref.init[ref._start_idx : -1][: self._shared_ser_dim]
            if self._shared_ser_dim
            else np.array([])
        )
        shared_ser_b = (
            list(ref.bounds[ref._start_idx : -1])[: self._shared_ser_dim]
            if self._shared_ser_dim
            else []
        )

        # Optional user overrides on the shared head [additional..., SER...]
        if init_override is not None:
            expect = len(shared_addl0) + len(shared_ser0)
            if len(init_override) != expect:
                raise ValueError(
                    f"init_override length {len(init_override)} != expected {expect}"
                )
            shared_addl0 = init_override[: len(shared_addl0)]
            shared_ser0 = init_override[len(shared_addl0) :]

        if bounds_override is not None:
            expect = len(shared_addl_b) + len(shared_ser_b)
            if len(bounds_override) != expect:
                raise ValueError(
                    f"bounds_override length {len(bounds_override)} != expected {expect}"
                )
            shared_addl_b = list(bounds_override[: len(shared_addl_b)])
            shared_ser_b = list(bounds_override[len(shared_addl_b) :])

        # 4) Assemble global theta and per-spectrum layout
        theta_parts = []
        bounds_parts: List[Tuple[Optional[float], Optional[float]]] = []

        # Shared head at the front
        theta_parts.append(shared_addl0)
        bounds_parts.extend(shared_addl_b)
        theta_parts.append(shared_ser0)
        bounds_parts.extend(shared_ser_b)

        cursor = sum(map(len, theta_parts))
        self._shared_addl_slice = slice(0, len(shared_addl0))
        self._shared_ser_slice = slice(
            self._shared_addl_slice.stop,
            self._shared_addl_slice.stop + len(shared_ser0),
        )

        self._layout = []  # one dict per spectrum
        for sp, d in zip(specs, ser_dims):
            f = sp.fitter

            # Local SER block only if not shared
            if sp.share_ser:
                local_ser_slice = slice(0, 0)  # empty
            else:
                local_ser0 = f.init[f._start_idx : -1]
                theta_parts.append(local_ser0)
                bounds_parts.extend(list(f.bounds[f._start_idx : -1]))
                local_ser_slice = slice(cursor, cursor + len(local_ser0))
                cursor += len(local_ser0)

            # Optional per-spectrum axis scaling parameter log_s (unbounded)
            if sp.use_scale_axis and sp.share_ser:
                log_s_index = cursor
                theta_parts.append(np.array([0.0]))  # init: log_s=0 => s=1
                bounds_parts.append((None, None))
                cursor += 1
            else:
                log_s_index = None

            # Occupancy
            occ_index = cursor
            theta_parts.append(np.array([f.init[-1]]))
            bounds_parts.append((0.0, 1.0))
            cursor += 1

            self._layout.append(
                dict(
                    local_ser_slice=local_ser_slice,
                    log_s_index=log_s_index,
                    occ_index=occ_index,
                    ser_dim=d,
                    share_ser=sp.share_ser,
                    use_scale_axis=sp.use_scale_axis,
                )
            )

        self.theta0 = np.concatenate(theta_parts) if theta_parts else np.array([])
        self.bounds = tuple(bounds_parts)

        # Keep original hooks to be safe if you ever need them later
        self._orig_ser_pdf = [sp.fitter._ser_pdf_time for sp in specs]

    # -------------------------
    #   Utility: temp patcher
    # -------------------------
    @contextmanager
    def _temporary_patch(self, obj, attr: str, new_callable):
        """Temporarily replace `obj.attr` with `new_callable`, then restore."""
        old = getattr(obj, attr)
        setattr(obj, attr, new_callable)
        try:
            yield
        finally:
            setattr(obj, attr, old)

    # -------------------------
    #   Axis scaling (time domain)
    # -------------------------
    def _make_scaled_ser_to_ft(self, sp_idx: int, s: float):
        """Return a _ser_to_ft(ser_args) applying axis scaling by s>0.

        Prefer analytic FT if available: P_s(ω) = P(ω s).
        Otherwise fall back to time-domain resampling:
            p_s(x) = (1/s) * p(x/s)
        evaluated on the fitter's own x-grid (.xsp), then roll+pad+FFT.
        """
        f = self.specs[sp_idx].fitter
        orig_ser_pdf = self._orig_ser_pdf[sp_idx]

        # guard
        s = float(max(s, 1e-8))

        def ser_to_ft_scaled(ser_args: np.ndarray):
            # --- Try analytic FT first
            ft = f._ser_ft(f._freq * s, ser_args)
            if ft is not None:
                return ft

            # --- Fallback: time-domain resampling with Jacobian 1/s
            pdf_base = orig_ser_pdf(ser_args)  # on f.xsp
            x = f.xsp
            pdf_scaled = (1.0 / s) * np.interp(x / s, x, pdf_base, left=0.0, right=0.0)

            pdf_padded, _, _ = roll_and_pad(pdf_scaled, f._shift, f._pad_safe)
            return fft(pdf_padded) * f._xsp_width

        return ser_to_ft_scaled

    # -------------------------
    #    Build local args
    # -------------------------
    def _args_for(self, theta: np.ndarray, i: int) -> np.ndarray:
        """Assemble local args for spectrum i as the fitter expects: [additional, SER, occ]."""
        f = self.specs[i].fitter
        lay = self._layout[i]

        # Shared head
        addl = theta[self._shared_addl_slice]
        shared_ser = theta[self._shared_ser_slice]

        # SER choice
        ser = shared_ser if lay["share_ser"] else theta[lay["local_ser_slice"]]

        # Occupancy
        occ = np.array([theta[lay["occ_index"]]])

        return np.r_[addl, ser, occ] if self._addl_len else np.r_[ser, occ]

    # -------------------------
    #   Joint log-likelihood
    # -------------------------
    def log_l(self, theta: np.ndarray) -> float:
        total = 0.0
        for i, sp in enumerate(self.specs):
            f = sp.fitter
            lay = self._layout[i]

            if lay["use_scale_axis"] and lay["share_ser"]:
                log_s = theta[lay["log_s_index"]]
                s = float(np.exp(log_s))
                ser_to_ft = self._make_scaled_ser_to_ft(i, s)
                # temporarily patch this fitter's _ser_to_ft to the scaled version
                with self._temporary_patch(f, "_ser_to_ft", ser_to_ft):
                    args_i = self._args_for(theta, i)
                    total += sp.weight * f.log_l(args_i)
            else:
                args_i = self._args_for(theta, i)
                total += sp.weight * f.log_l(args_i)

        return total

    # -------------------------
    #  Fitting API (reuses host)
    # -------------------------
    def fit_minuit(self, host_idx: int = 0, **kwargs):
        """Use one fitter as host and call its existing `.fit(method="minuit")`.

        We temporarily set on the host:
          - .log_l  -> CombinedFitter.log_l
          - .init   -> theta0
          - .bounds -> combined bounds
          - .dof    -> len(theta0)
        """
        host = self.specs[host_idx].fitter
        # backup
        _logl, _init, _bounds, _dof = host.log_l, host.init, host.bounds, host.dof
        try:
            host.log_l = self.log_l
            host.init = self.theta0.copy()
            host.bounds = self.bounds
            host.dof = len(host.init)
            host.fit(method="minuit", **kwargs)
        finally:
            host.log_l, host.init, host.bounds, host.dof = _logl, _init, _bounds, _dof

    def fit_mcmc(
        self, host_idx: int = 0, step_length: Optional[np.ndarray] = None, **kwargs
    ):
        """Same as fit_minuit but call the existing MCMC driver on the host."""
        host = self.specs[host_idx].fitter
        _logl, _init, _bounds, _dof = host.log_l, host.init, host.bounds, host.dof
        try:
            host.log_l = self.log_l
            host.init = self.theta0.copy()
            host.bounds = self.bounds
            host.dof = len(host.init)
            if step_length is None:
                hi = host.init
                step_length = np.maximum(np.abs(hi) * 0.05, 1e-3)
            host._fit_mcmc(step_length=step_length, **kwargs)
        finally:
            host.log_l, host.init, host.bounds, host.dof = _logl, _init, _bounds, _dof

    # -------------------------
    #  Convenience utilities
    # -------------------------
    def args_for(self, theta: np.ndarray, i: int) -> np.ndarray:
        """Public wrapper of _args_for to retrieve per-spectrum args."""
        return self._args_for(theta, i)

    def split_theta(self, theta: np.ndarray):
        """Return a dict view of the global theta:
        - 'shared_addl', 'shared_ser'
        - 'locals': list of {'log_s'(optional), 'occ', 'local_ser'(optional)}
        """
        out = dict(
            shared_addl=theta[self._shared_addl_slice],
            shared_ser=theta[self._shared_ser_slice],
            locals=[],
        )
        for lay in self._layout:
            entry = {}
            if lay["use_scale_axis"] and lay["share_ser"]:
                entry["log_s"] = theta[lay["log_s_index"]]
            entry["occ"] = theta[lay["occ_index"]]
            if not lay["share_ser"]:
                entry["local_ser"] = theta[lay["local_ser_slice"]]
            out["locals"].append(entry)
        return out
