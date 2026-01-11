# core/combined.py
from __future__ import annotations

import numpy as np
from typing import List


class CombinedFitter:
    """Combine multiple spectra into a joint fit with shared SER parameters.

    Parameter structure:
        theta = [logA_1, ..., logA_N (if fit_total), threshold_params (if threshold), shared_SER, occ_1, ..., occ_N]

    - logA_i: optional, independent for each spectrum if fit_total=True
    - threshold_params: optional, shared (pedestal or threshold effect params, length=_start_idx)
    - shared_SER: shared SER parameters (same for all spectra)
    - occ_i: independent occupancy for each spectrum i
    """

    def __init__(self, fitters: List[object]):
        if not fitters:
            raise ValueError("At least one fitter is required.")

        self.fitters = fitters
        self.n_spectra = len(fitters)

        self._validate_fitters()
        self._build_parameter_structure()

        print(f"[INFO] Combined {self.n_spectra} spectra", flush=True)
        print(f"[INFO] Total parameters: {self.dof}", flush=True)

    def _validate_fitters(self):
        ref = self.fitters[0]
        ref_fit_total = ref._fit_total
        ref_start_idx = ref._start_idx

        # Calculate reference SER dimension
        ref_head = 1 if ref_fit_total else 0
        ref_ser_dim = len(ref.init) - ref_head - ref_start_idx - 1

        for i, f in enumerate(self.fitters[1:], 1):
            if f._fit_total != ref_fit_total:
                raise ValueError(
                    f"Fitter {i}: fit_total={f._fit_total}, expected {ref_fit_total}"
                )

            if f._start_idx != ref_start_idx:
                raise ValueError(
                    f"Fitter {i}: _start_idx={f._start_idx}, expected {ref_start_idx}"
                )

            head = 1 if f._fit_total else 0
            ser_dim = len(f.init) - head - f._start_idx - 1
            if ser_dim != ref_ser_dim:
                raise ValueError(
                    f"Fitter {i}: SER dimension={ser_dim}, expected {ref_ser_dim}"
                )

    def _build_parameter_structure(self):
        ref = self.fitters[0]

        self._fit_total = ref._fit_total
        self._start_idx = ref._start_idx
        self._head = 1 if self._fit_total else 0

        # Extract shared parameter blocks from reference
        ref_init = ref.init
        ref_bounds = ref.bounds

        init_parts = []
        bounds_parts = []

        # logA (if fit_total)
        if self._fit_total:
            for f in self.fitters:
                init_parts.append(np.array([f.init[0]]))
                bounds_parts.append(f.bounds[0])
            self._logA_indices = list(range(len(self.fitters)))
        else:
            self._logA_indices = []

        cursor = len(self._logA_indices)

        # threshold/pedestal params (if any)
        if self._start_idx > 0:
            init_parts.append(ref_init[self._head : self._head + self._start_idx])
            bounds_parts.extend(ref_bounds[self._head : self._head + self._start_idx])
            self._threshold_slice = slice(cursor, cursor + self._start_idx)
            cursor += self._start_idx
        else:
            self._threshold_slice = slice(0, 0)

        # SPE
        ser_start = self._head + self._start_idx
        ser_end = len(ref_init) - 1
        self._ser_len = ser_end - ser_start
        init_parts.append(ref_init[ser_start:ser_end])
        bounds_parts.extend(ref_bounds[ser_start:ser_end])
        self._ser_slice = slice(cursor, cursor + self._ser_len)
        cursor += self._ser_len

        # occupancies
        self._occ_indices = []
        for f in self.fitters:
            init_parts.append(np.array([f.init[-1]]))
            bounds_parts.append(f.bounds[-1])
            self._occ_indices.append(cursor)
            cursor += 1

        self.init = np.concatenate(init_parts)
        self.bounds = tuple(bounds_parts)
        self.dof = len(self.init)

        # Store constraints from reference and remap indices to combined parameter space
        # Constraints apply to SER params, so we need to offset indices
        self.constraints = self._remap_constraints(ref.constraints)

    def _remap_constraints(self, constraints):
        """Remap constraint indices from local SER space to combined parameter space."""
        if not constraints:
            return []

        # In the reference fitter, SER params start at index: head + start_idx
        # In combined fitter, SER params start at: _ser_slice.start
        # Offset = combined_start - local_start
        local_ser_start = self._head + self._start_idx
        combined_ser_start = self._ser_slice.start
        offset = combined_ser_start - local_ser_start

        remapped = []
        for constraint in constraints:
            if isinstance(constraint, dict):
                new_constraint = constraint.copy()
                new_constraint["coeffs"] = [
                    (idx + offset, coeff) for idx, coeff in constraint["coeffs"]
                ]
                remapped.append(new_constraint)
            else:
                raise ValueError(f"Unknown constraint format: {constraint}")

        return remapped

    def _build_local_args(self, theta: np.ndarray, spec_idx: int) -> np.ndarray:
        """Build parameter array for individual fitter from combined theta."""
        parts = []

        # logA (individual)
        if self._fit_total:
            parts.append(
                theta[self._logA_indices[spec_idx] : self._logA_indices[spec_idx] + 1]
            )

        # threshold/pedestal (shared)
        if self._start_idx > 0:
            parts.append(theta[self._threshold_slice])

        # SPE
        parts.append(theta[self._ser_slice])

        # individual occupancy
        parts.append(
            theta[self._occ_indices[spec_idx] : self._occ_indices[spec_idx] + 1]
        )

        return np.concatenate(parts)

    def log_l(self, theta: np.ndarray) -> float:
        """Joint log-likelihood."""
        total_ll = 0.0

        for i, f in enumerate(self.fitters):
            args_i = self._build_local_args(theta, i)
            ll_i = f.log_l(args_i)

            if not np.isfinite(ll_i):
                return -np.inf

            total_ll += ll_i

        return total_ll

    def fit(self, strategy=1, tol=1e-01, max_calls=10000, print_level=0):
        """Fit using Minuit optimizer."""
        import ROOT

        ROOT.gErrorIgnoreLevel = ROOT.kError

        def nll_wrapper(par_ptr):
            args = np.array([par_ptr[i] for i in range(self.dof)], dtype=float)
            ll = self.log_l(args)
            return 1e30 if not np.isfinite(ll) else -ll

        self._nll_wrapper = nll_wrapper
        self._fcn = ROOT.Math.Functor(self._nll_wrapper, self.dof)

        def configure_minimizer(m, strat, tolerance, mc, pl):
            m.SetFunction(self._fcn)
            m.SetStrategy(strat)
            m.SetErrorDef(0.5)
            m.SetTolerance(tolerance)
            m.SetMaxFunctionCalls(mc)
            m.SetPrintLevel(pl)

            for i, (v0, lim) in enumerate(zip(self.init, self.bounds)):
                step = 0.1 * (abs(v0) if v0 else 1.0)
                lo, hi = lim
                name = f"p{i}"

                if lo is None and hi is None:
                    m.SetVariable(i, name, float(v0), step)
                elif lo is not None and hi is not None:
                    m.SetLimitedVariable(i, name, float(v0), step, float(lo), float(hi))
                elif lo is not None:
                    m.SetLowerLimitedVariable(i, name, float(v0), step, float(lo))
                else:
                    m.SetUpperLimitedVariable(i, name, float(v0), step, float(hi))

        print("[INFO] Initial parameters:", flush=True)
        self._print_params(self.init, self.bounds, prefix="  ")

        fail_count = 0
        converged = False

        while fail_count < 6:
            algo = ["Migrad", "Combined", "Migrad", "Combined", "Migrad", "Combined"][
                fail_count
            ]

            if fail_count < 2:
                this_tol = tol
            elif fail_count < 4:
                this_tol = max(10 * tol, 1.0)
            else:
                this_tol = max(50 * tol, 5.0)

            m = ROOT.Math.Factory.CreateMinimizer("Minuit2", algo)
            configure_minimizer(m, strategy, this_tol, max_calls, print_level)

            ok = m.Minimize()
            if ok:
                converged = True
                break
            else:
                print(f"[WARN] Minuit attempt {fail_count+1} failed", flush=True)
                fail_count += 1

        if not converged:
            print("[ERROR] Minuit did not converge after 6 attempts", flush=True)

        m.Hesse()

        fitted = np.array([m.X()[i] for i in range(self.dof)])
        errors = np.array([m.Errors()[i] for i in range(self.dof)])

        self.converged = converged
        self.fitted_params = fitted
        self.param_errors = errors
        self.likelihood = -m.MinValue()

        print(f"\n[INFO] Converged: {self.converged}", flush=True)
        print(f"[INFO] Log-likelihood: {self.likelihood:.6f}", flush=True)
        print("[INFO] Fitted parameters:", flush=True)
        self._print_params(fitted, self.bounds, errors, prefix="  ")

    def _print_params(self, params, bounds, errors=None, prefix=""):
        """Print parameters with their bounds and optionally errors."""
        cursor = 0

        if self._fit_total:
            cursor = len(self._logA_indices)

        # Threshold/pedestal params
        if self._start_idx > 0:
            label = "Pedestal" if self.fitters[0]._isWholeSpectrum else "Threshold"
            print(f"{prefix}{label} params (shared):", flush=True)
            for i, idx in enumerate(
                range(self._threshold_slice.start, self._threshold_slice.stop)
            ):
                lo, hi = bounds[idx]
                lo_str = f"{lo:.4g}" if lo is not None else "-inf"
                hi_str = f"{hi:.4g}" if hi is not None else "+inf"

                at_bound = ""
                if errors is not None:
                    err_str = f" ± {errors[idx]:.4g}"
                    # Check if hitting boundary
                    if lo is not None and abs(params[idx] - lo) < 1e-6:
                        at_bound = " [AT LOWER BOUND]"
                    elif hi is not None and abs(params[idx] - hi) < 1e-6:
                        at_bound = " [AT UPPER BOUND]"
                else:
                    err_str = ""

                print(
                    f"{prefix}  [{i}] = {params[idx]:.4g}{err_str}  (bounds: [{lo_str}, {hi_str}]){at_bound}",
                    flush=True,
                )

        # SPE
        print(f"{prefix}Shared SER params:", flush=True)
        for i, idx in enumerate(range(self._ser_slice.start, self._ser_slice.stop)):
            lo, hi = bounds[idx]
            lo_str = f"{lo:.4g}" if lo is not None else "-inf"
            hi_str = f"{hi:.4g}" if hi is not None else "+inf"

            at_bound = ""
            if errors is not None:
                err_str = f" ± {errors[idx]:.4g}"
                # Check if hitting boundary
                if lo is not None and abs(params[idx] - lo) < 1e-6:
                    at_bound = " [AT LOWER BOUND]"
                elif hi is not None and abs(params[idx] - hi) < 1e-6:
                    at_bound = " [AT UPPER BOUND]"
            else:
                err_str = ""

            print(
                f"{prefix}  [{i}] = {params[idx]:.4g}{err_str}  (bounds: [{lo_str}, {hi_str}]){at_bound}",
                flush=True,
            )

        if errors is not None:
            self.ser_args = params[self._ser_slice]
            self.ser_args_std = errors[self._ser_slice]

        # Occupancies
        print(f"{prefix}Occupancies:", flush=True)
        for i, idx in enumerate(self._occ_indices):
            lo, hi = bounds[idx]
            lo_str = f"{lo:.4g}" if lo is not None else "-inf"
            hi_str = f"{hi:.4g}" if hi is not None else "+inf"

            at_bound = ""
            if errors is not None:
                err_str = f" ± {errors[idx]:.4g}"
                # Check if hitting boundary
                if lo is not None and abs(params[idx] - lo) < 1e-6:
                    at_bound = " [AT LOWER BOUND]"
                elif hi is not None and abs(params[idx] - hi) < 1e-6:
                    at_bound = " [AT UPPER BOUND]"
            else:
                err_str = ""

            print(
                f"{prefix}  Spectrum {i}: {params[idx]:.4g}{err_str}  (bounds: [{lo_str}, {hi_str}]){at_bound}",
                flush=True,
            )

        if errors is not None:
            self.occs = params[self._occ_indices]
            self.occs_std = errors[self._occ_indices]
