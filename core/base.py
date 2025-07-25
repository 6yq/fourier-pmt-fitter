import numpy as np
from scipy.fft import fft
from scipy.stats import norm

from .utils import (
    composite_simpson,
    isInBound,
    isParamsInBound,
    isParamsWithinConstraints,
    compute_init,
    merged_pearson_chi2,
    modified_neyman_chi2_A,
    modified_neyman_chi2_B,
    mighell_chi2,
)
from .fft_utils import fft_and_ifft, roll_and_pad


class PMT_Fitter:
    """A class to fit MCP-PMT charge spectrum.

    Parameters
    ----------
    hist : ArrayLike
    bins : ArrayLike
    isWholeSpectrum : bool
        Whether the spectrum is whole spectrum
    A : int
        Total charge entries
    occ_init : float
        Initial occupancy
    sample : int
        The number of sample intervals between bins
    init : ArrayLike
        Initial params of SER charge model
    bounds : ArrayLike
        Initial bounds of SER charge model
    constraints : ArrayLike
        Constraints of SER charge model
        Example: `[
            {"coeffs": [(1, 1), (2, -2)], "threshold": 0, "op": ">"},
        ]` stands for `params[1] - 2 * params[2] > 0`
    threshold : None or str
        The threshold effect to be applied to the PDF.
        Should be one of "logistic", "erf", or None.
    auto_init : bool
        Whether to automatically initialize the model parameters.
    seterr : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        see https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
    """

    def __init__(
        self,
        hist,
        bins,
        isWholeSpectrum=False,
        A=None,
        occ_init=None,
        sample=None,
        init=None,
        bounds=None,
        constraints=None,
        threshold=None,
        auto_init=False,
        seterr: str = "warn",
        **peak_kwargs,
    ):
        # -------------------------
        #   Initial Data Handling
        # -------------------------
        np.seterr(all=seterr)
        self.seterr = seterr
        self._isWholeSpectrum = isWholeSpectrum
        self.A = A if A is not None else sum(hist)
        self._init = init if isinstance(init, np.ndarray) else np.array(init)
        self.bounds = (
            bounds.tolist() if isinstance(bounds, np.ndarray) else list(bounds)
        )
        self.constraints = constraints or []

        if occ_init:
            self._occ_init = occ_init
        elif self._isWholeSpectrum:
            self._occ_init = 0.1
        else:
            self._occ_init = sum(hist) / self.A

        self.sample = (
            16 * int(1 / (1 - self._occ_init) ** 0.673313) if sample is None else sample
        )

        self.hist = np.asarray(hist)
        self.bins = np.asarray(bins)

        self.zero = self.A - sum(self.hist)
        if self._isWholeSpectrum:
            assert self.zero == 0, "[ERROR] have a zero bug, please post an issue :)"

        # threshold effect & pedestal both need 2 parameters
        if self._isWholeSpectrum:
            self._start_idx = 2
        elif threshold is not None:
            self._start_idx = 2
        else:
            self._start_idx = 0

        # -------------------------
        #   Derived Attributes
        # -------------------------
        self._bin_width = self.bins[1] - self.bins[0]
        self._xs = (self.bins[:-1] + self.bins[1:]) / 2
        self._interval = self._bin_width / self.sample
        self._xsp_width = self._bin_width / self.sample
        self._shift = np.ceil(self.bins[0] / self._xsp_width).astype(int)

        self.xsp = np.linspace(
            self.bins[0] - abs(self._shift) * self._xsp_width,
            self.bins[-1],
            num=len(self.hist) * self.sample + abs(self._shift) + 1,
            endpoint=True,
        )

        _n_origin = len(self.xsp)
        self._pad_safe = 2 ** int(np.ceil(np.log2(_n_origin))) - _n_origin
        self._C = self._log_l_C()

        # -------------------------
        #     Produce Functions
        # -------------------------
        self._efficiency = self._produce_efficiency(threshold)
        self._all_PE_processor = self._produce_all_PE_processor()
        self._nPE_processor = self._produce_nPE_processor()
        self._b_sp = self._produce_b_sp()
        self._pdf_sr_n = self._produce_pdf_sr_n()
        self._estimate_count = self._produce_estimate_counter()
        self._constraint_checker = self._produce_constraint_checker()

        # -------------------------
        #     Auto Initialization
        # -------------------------

        # FUCK. There is pedestal leakage...
        # Might have to do both spectrum fitting and threshold correction...

        if self._isWholeSpectrum:
            if auto_init:
                ped_gp, ped_sigma = compute_init(
                    self.hist, self.bins, peak_idx=0, **peak_kwargs
                )
                print(f"[FIND PEAK] ped: {ped_gp} ± {ped_sigma}", flush=True)
                spe_gp, spe_sigma = compute_init(
                    self.hist, self.bins, peak_idx=1, **peak_kwargs
                )
                print(f"[FIND PEAK] spe: {spe_gp} ± {spe_sigma}", flush=True)
                self._replace_spe_params(spe_gp, spe_sigma, self._occ_init)
                self._replace_spe_bounds(spe_gp, spe_sigma, self._occ_init)
                self.init = np.array([ped_gp, ped_sigma, *self._init, self._occ_init])

                ped_peak_fluc = 5
                ped_sigma_percentile = 0.2
                self.bounds.insert(0, (ped_gp - ped_peak_fluc, ped_gp + ped_peak_fluc))
                self.bounds.insert(
                    1,
                    (
                        ped_sigma * (1 - ped_sigma_percentile),
                        ped_sigma * (1 + ped_sigma_percentile),
                    ),
                )
            else:
                self.init = np.append(self._init, self._occ_init)
        else:
            if auto_init:
                try:
                    spe_gp, spe_sigma = compute_init(
                        self.hist, self.bins, peak_idx=0, **peak_kwargs
                    )
                    print(f"[FIND PEAK] spe: {spe_gp} ± {spe_sigma}", flush=True)
                    self._replace_spe_params(spe_gp, spe_sigma, self._occ_init)
                    self._replace_spe_bounds(spe_gp, spe_sigma, self._occ_init)
                except:
                    print(f"[WARNING] Cannot find SPE peak.", flush=True)
            if threshold is not None:
                # TODO: is bins[1] good enough to be the initial value?
                # TODO: is bin_width good enought to be the initial value?
                threshold_center, threshold_scale = bins[1], self._bin_width
                self.init = np.array(
                    [threshold_center, threshold_scale, *self._init, self._occ_init]
                )
                # TODO: is 5 times bin width enough?
                threshold_scale_fluc = 4 * threshold_scale
                # threshold effect center should be between 0 and the SPE peak
                self.bounds.insert(
                    0,
                    (
                        0,
                        bins[np.argmax(hist) + 1],
                    ),
                )
                self.bounds.insert(
                    1,
                    (
                        0,
                        threshold_scale + threshold_scale_fluc,
                    ),
                )
            else:
                self.init = np.append(self._init, self._occ_init)

        self.dof = len(self.init)
        self.bounds.append(
            (
                0.8 * self._occ_init,
                min(1.2 * self._occ_init, 1.0),
            )
        )
        self.bounds = tuple(self.bounds)

        for i, b in zip(self.init, self.bounds):
            print(
                f"[INIT] init {i} with boundary {tuple(float(x) if x is not None else None for x in b)}",
                flush=True,
            )

    # -------------------------
    #  Produce Helper Functions
    # -------------------------

    def _produce_efficiency(self, threshold_type):
        if not self._isWholeSpectrum:
            if threshold_type == "logistic":
                return lambda x, center, scale: 1 / (1 + np.exp(-(x - center) / scale))
            elif threshold_type == "erf":
                from scipy.special import erf

                return lambda x, center, scale: 0.5 * (
                    1 + erf((x - center) / (scale * np.sqrt(2)))
                )
            elif threshold_type is None:
                return lambda x: np.ones_like(x)
            else:
                raise ValueError(f"Unknown threshold type: {threshold_type}")
        else:
            return lambda x: np.ones_like(x)

    def _produce_all_PE_processor(self):
        # TODO: correct proportions if SPE contains delta component
        if self._isWholeSpectrum:
            return (
                lambda occupancy, b_sp: lambda s_sp: np.exp(
                    -np.log(1 - occupancy) * (s_sp - 1)
                )
                * b_sp
            )
        else:
            # return lambda occupancy, b_sp: lambda s_sp: np.exp(
            #     -np.log(1 - occupancy) * (s_sp - 1)
            # )
            return lambda occupancy, b_sp: lambda s_sp: (1 - occupancy) * (
                np.exp(-np.log(1 - occupancy) * s_sp) - 1
            )

    def _produce_nPE_processor(self):
        return (
            lambda occupancy, n: lambda s_sp: (1 - occupancy)
            * ((-np.log(1 - occupancy) * s_sp) ** n)
            / np.prod(range(1, n + 1))
        )

    def _produce_b_sp(self):
        if self._isWholeSpectrum:
            return (
                lambda args: fft(
                    roll_and_pad(
                        self._pdf_ped(args[: self._start_idx]),
                        self._shift,
                        self._pad_safe,
                    )[0]
                )
                * self._xsp_width
            )
        else:
            return lambda args: None

    def _produce_zero_pe(self):
        if self._isWholeSpectrum:
            return lambda args: self._pdf_ped(args[: self._start_idx])
        else:
            return lambda args: np.zeros_like(self.xsp)

    def _produce_pdf_sr_n(self):
        """Return n-order pdf.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ) if only PE spectrum,
            (ped_mean, ped_sigma, ser_args_1, ..., ser_args_(dof-2), occ) otherwise
        n : int
            nPE

        Notes
        -----
        nPE contributes exp(-mu) / k! * [mu * s_sp]^k
        """
        if self._isWholeSpectrum:
            return lambda args, n: (
                self._pdf_ped(args[: self._start_idx])
                if n == 0
                else fft_and_ifft(
                    self._pdf(args[self._start_idx : -1]),
                    self._shift,
                    self._xsp_width,
                    self._pad_safe,
                    self._nPE_processor(args[-1], n),
                    self.const(args[self._start_idx : -1]),
                )
            )
        else:
            return lambda args, n: (
                np.zeros_like(self.xsp)
                if n == 0
                else fft_and_ifft(
                    self._pdf(args[self._start_idx : -1]),
                    self._shift,
                    self._xsp_width,
                    self._pad_safe,
                    self._nPE_processor(args[-1], n),
                    self.const(args[self._start_idx : -1]),
                )
            )

    def _produce_estimate_counter(self):
        """Return a function that estimates counts of every bin.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ) if only PE spectrum,
            (ped_mean, ped_sigma, ser_args_1, ..., ser_args_(dof-2), occ) otherwise

        Return
        ------
        y_est : ArrayLike
            (entry_est_in_bin_1, ..., entry_est_in_bin_n)
        z_est : float
            Expected zero entries.

        Notes
        -----
        For pdf which has a delta component, the bin containing 0 would finally has a delta proportion.
        If the spectrum edge contains 0, then the first sampling point should be masked with 0.
        """
        need_mask_delta = self.bins[0] == 0
        w = np.ones(self.sample + 1)
        w[1:-1:2] = 4
        w[2:-2:2] = 2
        w *= self._interval / 3
        self._simp_w = w

        def counter(args):
            y_sp = self.A * self._pdf_sr(args=args)

            if need_mask_delta:
                y_sp[0] = 0.0

            nbin = len(self.hist)
            # indices[i, j] = abs_shift + sample*i + j
            idx = (
                abs(self._shift)
                + self.sample * np.arange(nbin)[:, None]
                + np.arange(self.sample + 1)[None, :]
            )
            seg = y_sp[idx]  # (nbin, sample+1)

            y_est = seg @ self._simp_w

            # nonegative pdf set
            y_est[y_est <= 0] = 1e-20
            # for whole spectrum, z_est doesn't matter because self.zero is 0
            # otherwise, only SPE parameters are needed to calculate z_est
            z_est = self.A - y_est.sum()
            return y_est, z_est

        return counter

    def _produce_constraint_checker(self):
        if self.constraints:
            return lambda args: isParamsWithinConstraints(args, self.constraints)
        else:
            return lambda args: True

    # -------------------------
    #   Abstract & Utilities
    # -------------------------

    def _replace_spe_params(self, gp_init, sigma_init):
        raise NotImplementedError

    def _replace_spe_bounds(self, gp_bound, sigma_bound):
        raise NotImplementedError

    def _pdf(self, args):
        raise NotImplementedError

    def _zero(self, args):
        return 1 - args[-1]

    def const(self, args):
        return 0

    def get_gain(self, args):
        raise NotImplementedError

    def _log_l_C(self):
        """Return constant in log likelihood.

        Notes
        -----
        C = NlnN - ln(N!) + sum_j(ln(nj!))
        """
        N = sum(self.hist) + self.zero
        N_part = N * np.log(N) - sum(np.log(np.arange(1, N + 1)))
        n_part = sum([sum(np.log(np.arange(1, n + 1))) for n in self.hist]) + sum(
            np.log(np.arange(1, self.zero + 1))
        )
        return N_part + n_part

    def _pdf_ped(self, args):
        return norm.pdf(self.xsp, loc=args[0], scale=args[1])

    def _pdf_sr(self, args):
        """Applying DFT & IDFT to estimate pdf.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ) if only PE spectrum,
            (ped_mean, ped_sigma, ser_args_1, ..., ser_args_(dof-2), occ) otherwise
        """
        ser_args = args[self._start_idx : -1]
        pdf = self._pdf(ser_args)
        const = self.const(ser_args)
        if not np.all(np.isfinite(pdf)):
            raise ValueError("Non-finite value in PDF.")

        b_sp = self._b_sp(args)
        pass_threshold = self._efficiency(self.xsp, *args[: self._start_idx])
        fourier_pdf = fft_and_ifft(
            pdf,
            self._shift,
            self._xsp_width,
            self._pad_safe,
            self._all_PE_processor(args[-1], b_sp),
            const,
        )
        return fourier_pdf * pass_threshold

    def _estimate_smooth(self, args):
        return self.A * self._bin_width * self._pdf_sr(args=args)[abs(self._shift) :]

    def estimate_smooth_n(self, args, n):
        return (
            self.A
            * self._bin_width
            * self._pdf_sr_n(args=args, n=n)[abs(self._shift) :]
        )

    def log_l(self, args) -> float:
        """log likelihood of given args.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ) if only PE spectrum,
            (ped_mean, ped_sigma, ser_args_1, ..., ser_args_(dof-2), occ) otherwise
        """
        # make sure args are in range (an infinite "well")
        try:
            if isParamsInBound(args, self.bounds) and self._constraint_checker(
                args[self._start_idx :]
            ):
                y, z = self._estimate_count(args)
                log_l = self.zero * np.log(z) + np.sum(self.hist * np.log(y)) - self._C
                # temporary fix for NaN log likelihood
                if np.isnan(log_l):
                    return -np.inf
                return log_l
            else:
                return -np.inf
        except ValueError as e:
            if self.seterr != "ignore":
                print(
                    "[WARNING] Some chain(s) have Inf/NaN PDF value(s). Please improve the PDF robustness of your model.",
                    flush=True,
                )
            return -np.inf

    def get_chi_sq(self, args, chiSqFunc: callable, dof: int) -> float:
        """Get chi square.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ) if only PE spectrum,
            (ped_mean, ped_sigma, ser_args_1, ..., ser_args_(dof-2), occ) otherwise
        chiSqFunc : callable
            Function to compute chi-square.
        dof : int
            Degrees of freedom.

        Notes
        -----
        There are so many ways to define chi-square...
        We provide:
        - Merged Pearson chi-square (hmm, not good for low stat area)
        - Modified Neyman chi-square (from Cressie-Read family)
        - Modified Neyman chi-square (min(O, 1))
        - Mighell chi-square
        """
        y, z = self._estimate_count(args)
        return chiSqFunc(self.hist, y, self.zero, z, dof)

    def _fit_mcmc(
        self,
        nwalkers: int = 32,
        stage_steps: int = 200,
        max_stages: int = 20,
        seed: int = None,
        track: int = 1,
        step_length: list | np.ndarray = None,
        conv_factor: float = 20,
        conv_change: float = 0.02,
        processes: int = 8,
    ):
        """MCMC fit using `emcee`.

        Parameters
        ----------
        nwalkers : int
            Number of parallel chains for `emcee`.
        stage_steps : int
            MCMC step for `emcee` in each stage.
        max_stages : int
            Maximum stages for `emcee`.
        seed : int
            Seed for random.
        track : int
            Take only every `track` steps from the chain.
        step_length : ndarray[float]
            Step length to generate initial values.
        conv_factor : float
            Convergence factor, N > conv_factor * τ.
        conv_change : float
            Change of τ to trigger convergence, τ change < conv_change.
        processes : int
            You might want use Pool() to accelerate the fitting.

        Notes
        -----
        `nwalkers >= 2 * ndim`, credits to Xuewei.
        """
        import emcee
        from multiprocessing import Pool

        if seed is not None:
            np.random.seed(seed)
        rng = np.random.default_rng(42) if seed is None else np.random.default_rng(seed)

        ndim = self.dof
        p0 = self.init + rng.uniform(-1, 1, (nwalkers, ndim)) * step_length
        moves = [
            (emcee.moves.DEMove(sigma=1e-03), 0.8),
            (emcee.moves.DESnookerMove(), 0.2),
        ]

        with Pool(processes=processes) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers,
                ndim,
                self.log_l,
                moves=moves,
            )

            old_tau, state = np.inf, None
            for stage in range(max_stages):
                state = sampler.run_mcmc(state or p0, stage_steps, progress=True)

                # get autocorrelation time
                try:
                    tau = sampler.get_autocorr_time(tol=0)
                except emcee.autocorr.AutocorrError:
                    # the chain is too short
                    continue

                print(
                    rf"[Stage {stage+1}] τ ≈ {tau.max():.1f}  (mean {tau.mean():.1f})",
                    flush=True,
                )

                converged = np.all(tau * conv_factor < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < conv_change)
                old_tau = tau

                if converged:
                    print(">>> Converged!", flush=True)
                    break

        burn_in = int(5 * old_tau.max())

        print(f"[burn] steps: {burn_in}", flush=True)

        # (n_step, n_walker, n_param)
        self.samples_track = sampler.get_chain(discard=burn_in, thin=track, flat=False)
        self.log_l_track = sampler.get_log_prob(discard=burn_in, thin=track, flat=False)
        flat_chain = self.samples_track.reshape(
            -1, self.samples_track.shape[-1]
        )  # (Nsamples, ndim)
        args_complete = flat_chain.mean(axis=0)
        args_complete_std = flat_chain.std(axis=0, ddof=1)
        acceptance = sampler.acceptance_fraction

        # get integrated time and effective sample size
        try:
            tau_final = emcee.autocorr.integrated_time(
                self.samples_track, c=5, tol=conv_factor, quiet=True
            )
        except emcee.autocorr.AutocorrError:
            tau_final = np.full(ndim, np.nan)

        print(
            rf"[INFO] τ final (max / mean): {np.nanmax(tau_final):.1f} / {np.nanmean(tau_final):.1f}",
            flush=True,
        )

        N_tot = flat_chain.shape[0]  # total retained draws
        ess = N_tot / tau_final  # effective sample size
        # args_complete_std *= np.sqrt(tau_final)  # per-parameter MC error
        print(
            f"[INFO] ESS (min/max): {np.nanmin(ess):.0f} / {np.nanmax(ess):.0f}",
            flush=True,
        )

        sampler.reset()

        self.ser_args = args_complete[self._start_idx : -1]
        self.ser_args_std = args_complete_std[self._start_idx : -1]

        # for whole spectrum, additional args belong to pedestal
        # otherwise, they are from threshold effect
        self.additional_args = args_complete[: self._start_idx]
        self.additional_args_std = args_complete_std[: self._start_idx]

        # _zero() is a fix of real zero count
        occReg = 1 - np.apply_along_axis(
            self._zero, axis=1, arr=flat_chain[:, self._start_idx :]
        )
        self.occ = np.mean(occReg, axis=0)
        self.occ_std = np.std(occReg, axis=0)

        self.gps = np.apply_along_axis(
            self.get_gain,
            axis=1,
            arr=flat_chain[:, self._start_idx : -1],
            gain="gp",
        )
        self.gms = np.apply_along_axis(
            self.get_gain,
            axis=1,
            arr=flat_chain[:, self._start_idx : -1],
            gain="gm",
        )

        print(f"[INFO] Current burn-in: {burn_in} steps", flush=True)
        print(f"[INFO] Mean acceptance fraction: {np.mean(acceptance):.3f}", flush=True)
        print(
            f"[INFO] Acceptance percentile: {np.percentile(acceptance, [25, 50, 75])}",
            flush=True,
        )
        print(
            f"[INFO] Init params: " + ", ".join([f"{e:.4g}" for e in self.init]),
            flush=True,
        )

        additional_args_stream = (
            "Pedestal params: "
            if self._isWholeSpectrum
            else "Threshold effect params: "
        )
        print(
            "[INFO] "
            + additional_args_stream
            + ", ".join(
                [
                    f"{e:.4g} pm {f:.4g}"
                    for e, f in zip(self.additional_args, self.additional_args_std)
                ]
            ),
            flush=True,
        )

        print(
            "[INFO] SER params: "
            + ", ".join(
                [
                    f"{e:.4g} pm {f:.4g}"
                    for e, f in zip(self.ser_args, self.ser_args_std)
                ]
            ),
            flush=True,
        )
        print(
            "[INFO] Occupancy: " + ", ".join([f"{self.occ:.4g} pm {self.occ_std:.4g}"]),
            flush=True,
        )

        self.likelihood = self.log_l(args_complete)
        self.chi_sq_pearson, self.ndf_merged = self.get_chi_sq(
            args_complete, merged_pearson_chi2, dof=self.dof
        )
        self.chi_sq_neyman_A, self.ndf = self.get_chi_sq(
            args_complete, modified_neyman_chi2_A, dof=self.dof
        )
        self.chi_sq_neyman_B, _ = self.get_chi_sq(
            args_complete, modified_neyman_chi2_B, dof=self.dof
        )
        self.chi_sq_mighell, _ = self.get_chi_sq(
            args_complete, mighell_chi2, dof=self.dof
        )
        self.smooth = self._estimate_smooth(args_complete)
        self.ys, self.zs = self._estimate_count(args_complete)

    def _fit_minuit(self, *, strategy=1, tol=1e-01, max_calls=10000, print_level=0):
        """Fit with Minuit.

        Parameters
        ----------
        strategy : int
            Minuit strategy.
        tol : float
            Tolerance of convergence.
        max_calls : int
            Maximum number of function calls.
        print_level : int
            Print level of Minuit.
        """
        import ROOT

        ROOT.gErrorIgnoreLevel = ROOT.kError

        # consistent nll wrapper for log likelihood function
        def _nll_wrap(par_ptr):
            # par_ptr behaves like C double* (indexable)
            args = np.array([par_ptr[i] for i in range(self.dof)], dtype=float)
            ll = self.log_l(args)
            return 1e30 if not np.isfinite(ll) else -ll

        # init a Minuit minimizer
        def _configure_minimizer(m, strategy, tol, max_calls, print_level):
            m.SetFunction(self._fcn)
            m.SetStrategy(strategy)
            m.SetErrorDef(0.5)
            m.SetTolerance(tol)
            m.SetMaxFunctionCalls(max_calls)
            m.SetPrintLevel(print_level)

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

        # this is to prevent GC clear _nll_wrap
        self._nll_wrap = _nll_wrap

        self._fcn = ROOT.Math.Functor(self._nll_wrap, self.dof)

        failCnt = 0

        while True:
            algo = ["Migrad", "Combined", "Migrad", "Combined", "Migrad", "Combined"][
                failCnt
            ]
            if failCnt < 2:
                this_tol = tol
            elif failCnt < 4:
                this_tol = max(10 * tol, 1.0)
            else:
                this_tol = max(50 * tol, 5.0)
            m = ROOT.Math.Factory.CreateMinimizer("Minuit2", algo)
            _configure_minimizer(m, strategy, this_tol, max_calls, print_level)

            ok = m.Minimize()
            if ok:
                break
            else:
                print(
                    "[WARN] Minuit did not converge (EDM>tol or max_calls reached).",
                    flush=True,
                )
                failCnt += 1
                if failCnt >= 6:
                    raise Exception("Minuit failed all attempts.")

        m.Hesse()

        args_complete = np.array([m.X()[i] for i in range(self.dof)])
        args_complete_std = np.array([m.Errors()[i] for i in range(self.dof)])

        self.additional_args = args_complete[: self._start_idx]
        self.additional_args_std = args_complete_std[: self._start_idx]
        self.ser_args = args_complete[self._start_idx : -1]
        self.ser_args_std = args_complete_std[self._start_idx : -1]
        self.occ, self.occ_std = args_complete[-1], args_complete_std[-1]

        self.gps = self.get_gain(self.ser_args, "gp")
        self.gms = self.get_gain(self.ser_args, "gm")

        print(
            f"[INFO] Init params: " + ", ".join([f"{e:.4g}" for e in self.init]),
            flush=True,
        )

        additional_args_stream = (
            "Pedestal params: "
            if self._isWholeSpectrum
            else "Threshold effect params: "
        )
        print(
            "[INFO] "
            + additional_args_stream
            + ", ".join(
                [
                    f"{e:.4g} pm {f:.4g}"
                    for e, f in zip(self.additional_args, self.additional_args_std)
                ]
            ),
            flush=True,
        )

        print(
            "[INFO] SER params: "
            + ", ".join(
                [
                    f"{e:.4g} pm {f:.4g}"
                    for e, f in zip(self.ser_args, self.ser_args_std)
                ]
            ),
            flush=True,
        )
        print(
            "[INFO] Occupancy: " + ", ".join([f"{self.occ:.4g} pm {self.occ_std:.4g}"]),
            flush=True,
        )

        self.likelihood = -m.MinValue()
        self.chi_sq_pearson, self.ndf_merged = self.get_chi_sq(
            args_complete, merged_pearson_chi2, dof=self.dof
        )
        self.chi_sq_neyman_A, self.ndf = self.get_chi_sq(
            args_complete, modified_neyman_chi2_A, dof=self.dof
        )
        self.chi_sq_neyman_B, _ = self.get_chi_sq(
            args_complete, modified_neyman_chi2_B, dof=self.dof
        )
        self.chi_sq_mighell, _ = self.get_chi_sq(
            args_complete, mighell_chi2, dof=self.dof
        )
        self.smooth = self._estimate_smooth(args_complete)
        self.ys, self.zs = self._estimate_count(args_complete)

    def fit(self, method="minuit", **kwargs):
        """Fit with MCMC or Minuit.

        Parameters
        ----------
        method : str
            Fitting method.
        kwargs : dict
            Fitting parameters.
        """
        if method == "mcmc":
            return self._fit_mcmc(**kwargs)
        elif method == "minuit":
            return self._fit_minuit(**kwargs)
        else:
            raise ValueError("method must be 'mcmc' or 'minuit'")
