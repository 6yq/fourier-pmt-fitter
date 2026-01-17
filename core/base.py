import numpy as np

from math import log
from scipy.stats import norm
from scipy.fft import fft, ifft

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
from .fft_utils import roll_and_pad


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
        fit_total: bool = True,
        threshold_scale: float = 500.0,
        **peak_kwargs,
    ):
        # -------------------------
        #   Initial Data Handling
        # -------------------------
        np.seterr(all=seterr)
        self.seterr = seterr
        self._isWholeSpectrum = isWholeSpectrum
        self._fit_total = fit_total
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

        bins_mid = (self.bins[:-1] + self.bins[1:]) / 2
        mu_hat = -log(1 - self._occ_init)
        mu_est = mu_hat - (self._occ_init / (1 - self._occ_init)) / (2 * A)
        self.gms_prime = (
            np.average(bins_mid, weights=self.hist) * mu_est / (1 - np.exp(-mu_est))
        )

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

        self._n_full = len(self.xsp) + self._pad_safe
        self._freq = 2 * np.pi * np.fft.fftfreq(self._n_full, d=self._xsp_width)
        self._shift_padded = 2 * self._shift if self._shift < 0 else 0
        self._recover_slice = slice(0, len(self.xsp))

        # -------------------------
        #     Produce Functions
        # -------------------------
        self._efficiency = self._produce_efficiency(threshold)
        self._all_PE_processor = self._produce_all_PE_processor()
        self._nPE_processor = self._produce_nPE_processor()
        self._b_sp = self._produce_b_sp()
        self._ser_to_ft = self._produce_ser_to_ft()
        self._ifft_pipeline = self._produce_ifft_pipeline()
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

                ped_peak_fluc = 0.5 * (spe_gp - ped_gp)
                self.bounds.insert(0, (ped_gp - ped_peak_fluc, ped_gp + ped_peak_fluc))
                self.bounds.insert(
                    1,
                    (0, ped_peak_fluc),
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
                # The efficiency curve (for FSMP) is calibrated from ToyMC
                loc, scale = [0.08161452, 0.02022103]
                thres_center = loc * threshold_scale
                thres_scale = scale * threshold_scale

                self.init = np.array(
                    [thres_center, thres_scale, *self._init, self._occ_init]
                )
                threshold_scale_fluc = 0.100
                self.bounds.insert(
                    0,
                    (
                        max(0, thres_center * (1 - threshold_scale_fluc)),
                        thres_center * (1 + threshold_scale_fluc),
                    ),
                )
                self.bounds.insert(
                    1,
                    (
                        thres_scale * (1 - threshold_scale_fluc),
                        thres_scale * (1 + threshold_scale_fluc),
                    ),
                )
            else:
                self.init = np.append(self._init, self._occ_init)

        if self._fit_total:
            logA0 = log(self.A)
            self.init = np.insert(self.init, 0, logA0)
            self.bounds.insert(0, (None, None))

        self.dof = len(self.init)
        self.bounds.append(
            (
                self._occ_init * 0.90,
                min(self._occ_init * 1.10, 1.0 - 1e-06),
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
                return lambda x, *ps: 1 / (1 + np.exp(-(x - ps[-2]) / ps[-1]))
            elif threshold_type == "erf":
                from scipy.special import erf

                return lambda x, *ps: 0.5 * (
                    1 + erf((x - ps[-2]) / (ps[-1] * np.sqrt(2)))
                )
            elif threshold_type is None:
                return lambda x, *ps: np.ones_like(x)
            else:
                raise ValueError(f"Unknown threshold type: {threshold_type}")
        else:
            return lambda x, *ps: np.ones_like(x)

    def _produce_all_PE_processor(self):
        if self._isWholeSpectrum:
            return (
                lambda occupancy, b_sp: lambda s_sp: np.exp(
                    -np.log(1 - occupancy) * (s_sp - 1)
                )
                * b_sp
            )
        else:
            return lambda occupancy, b_sp: lambda s_sp: (1 - occupancy) * (
                np.expm1(-np.log(1 - occupancy) * s_sp)
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
                        self._pdf_ped(
                            args[self._head() : self._head() + self._start_idx]
                        ),  # CHG
                        self._shift,
                        self._pad_safe,
                    )[0]
                )
                * self._xsp_width
            )
        else:
            return lambda args: None

    def _produce_ser_to_ft(self):
        """Return SER PDF in Fourier domain."""

        def ser_to_ft(ser_args):
            ft = self._ser_ft(self._freq, ser_args)
            if ft is not None:
                return ft
            pdf = self._ser_pdf_time(ser_args)
            pdf_padded, _, _ = roll_and_pad(pdf, self._shift, self._pad_safe)
            return fft(pdf_padded) * self._xsp_width

        return ser_to_ft

    def _produce_ifft_pipeline(self):
        """Return processed PDF."""

        def ifft_back(s_sp_processed):
            ifft_full = np.real(ifft(s_sp_processed)) / self._xsp_width
            result = np.roll(ifft_full, -self._shift_padded)[self._recover_slice]
            return np.maximum(result, 0.0)

        return ifft_back

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

        def pdf_sr_n(args, n):
            a = np.asarray(args, float)
            h = self._head()
            n_ser = len(self.init) - h - self._start_idx - 1

            if a.size == len(self.init):
                ser_args = a[h + self._start_idx : -1]
                occ = a[-1]
            elif a.size == n_ser + 1:
                ser_args = a[:-1]
                occ = a[-1]
            else:
                raise ValueError(
                    f"args length {a.size} invalid; "
                    f"need full({len(self.init)}) or tail({n_ser+1})."
                )

            if n == 0:
                return (
                    self._pdf_ped(a[h : h + self._start_idx])
                    if self._isWholeSpectrum
                    else np.zeros_like(self.xsp)
                )

            const = self.const(ser_args)
            ft = self._ser_to_ft(ser_args)
            fft_in = ft + const
            return self._ifft_pipeline(self._nPE_processor(occ, n)(fft_in))

        return pdf_sr_n

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
        w *= self._xsp_width / 3
        self._simp_w = w

        def counter(args):
            A_now = self._A_from_args(args)
            y_sp = A_now * self._pdf_sr(args=args)

            if need_mask_delta:
                y_sp[0] = 0.0

            nbin = len(self.hist)
            abs_shift = abs(self._shift)
            idx = (
                abs_shift
                + self.sample * np.arange(nbin)[:, None]
                + np.arange(self.sample + 1)[None, :]
            )
            seg = y_sp[idx]  # (nbin, sample+1)
            y_est = seg @ self._simp_w
            y_est = np.maximum(y_est, 1e-32)

            front_count = 0.0
            if abs_shift > 0:
                front_count = np.trapz(y_sp[:abs_shift], dx=self._xsp_width)

            tail = args[self._head() + self._start_idx :]
            z_est = A_now * self._zero(tail) + front_count

            return y_est, z_est

        return counter

    def _produce_constraint_checker(self):
        if self.constraints:
            return lambda args: isParamsWithinConstraints(
                args[self._head() + self._start_idx :], self.constraints
            )
        else:
            return lambda args: True

    # -------------------------
    #   Abstract & Utilities
    # -------------------------

    def _replace_spe_params(self, gp_init, sigma_init):
        raise NotImplementedError

    def _replace_spe_bounds(self, gp_bound, sigma_bound):
        raise NotImplementedError

    def _ser_pdf_time(self, args):
        raise NotImplementedError

    def _ser_ft(self, freq, args):
        return None

    def _zero(self, args):
        return 1 - args[-1]

    def const(self, args):
        return 0

    def get_gain(self, args):
        raise NotImplementedError

    def _A_from_args(self, args):
        if self._fit_total:
            logA = np.clip(args[0], 0, np.log(1e12))
            return np.exp(logA)
        else:
            return self.A

    def _head(self):
        return 1 if getattr(self, "_fit_total", False) else 0

    def _log_l_C(self):
        """Return factorial constant in (extended) Poisson log-likelihood."""
        N0 = int(self.zero)
        hist = self.hist
        n_part = sum([sum(np.log(np.arange(1, int(n) + 1))) for n in hist])
        n0_part = sum(np.log(np.arange(1, N0 + 1)))
        return n_part + n0_part

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
        ser_args = args[self._head() + self._start_idx : -1]
        occ = args[-1]
        const = self.const(ser_args)

        ft_cont = self._ser_to_ft(ser_args)
        fft_input = ft_cont + const

        b_sp = self._b_sp(args)
        fft_processed = self._all_PE_processor(occ, b_sp)(fft_input)

        fourier_pdf = self._ifft_pipeline(fft_processed)
        extra = args[self._head() : self._head() + self._start_idx]
        pass_threshold = self._efficiency(self.xsp, *extra)
        return fourier_pdf * pass_threshold

    def _estimate_smooth(self, args):
        return self._A_from_args(args) * self._bin_width * self._pdf_sr(args=args)

    def estimate_smooth_n(self, args, n):
        return (
            self._A_from_args(args) * self._bin_width * self._pdf_sr_n(args=args, n=n)
        )

    def log_l(self, args) -> float:
        """log likelihood of given args.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ) if only PE spectrum,
            (ped_mean, ped_sigma, ser_args_1, ..., ser_args_(dof-2), occ) otherwise
        """
        if isParamsInBound(args, self.bounds) and self._constraint_checker(args):
            y, z = self._estimate_count(args)

            with np.errstate(divide="ignore", invalid="ignore"):
                ll_bins = np.sum(self.hist * np.log(y) - y)
                ll_zero = self.zero * np.log(z) - z

            log_l = ll_bins + ll_zero - self._C

            if np.isnan(log_l) or not np.isfinite(log_l):
                return -np.inf
            return log_l
        else:
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

        h = self._head()
        start = h + self._start_idx
        self.additional_args = args_complete[h:start]
        self.additional_args_std = args_complete_std[h:start]
        self.ser_args = args_complete[start:-1]
        self.ser_args_std = args_complete_std[start:-1]
        self.occ, self.occ_std = args_complete[-1], args_complete_std[-1]

        h = self._head()
        start = h + self._start_idx
        self.gps = np.apply_along_axis(
            self.get_gain, axis=1, arr=flat_chain[:, start:-1], gain="gp"
        )
        self.gms = np.apply_along_axis(
            self.get_gain, axis=1, arr=flat_chain[:, start:-1], gain="gm"
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
                this_tol = min(10 * tol, 1.0)
            else:
                this_tol = min(10 * tol, 1.0)
                strategy = 3 - strategy

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
                    break

        self.converged = failCnt < 6
        m.Hesse()

        self.full_args = np.array([m.X()[i] for i in range(self.dof)])
        self.full_args_std = np.array([m.Errors()[i] for i in range(self.dof)])

        h = self._head()
        start = h + self._start_idx
        self.additional_args = self.full_args[h:start]
        self.additional_args_std = self.full_args_std[h:start]
        self.ser_args = self.full_args[start:-1]
        self.ser_args_std = self.full_args_std[start:-1]
        self.occ, self.occ_std = self.full_args[-1], self.full_args_std[-1]

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
            self.full_args, merged_pearson_chi2, dof=self.dof
        )
        self.chi_sq_neyman_A, self.ndf = self.get_chi_sq(
            self.full_args, modified_neyman_chi2_A, dof=self.dof
        )
        self.chi_sq_neyman_B, _ = self.get_chi_sq(
            self.full_args, modified_neyman_chi2_B, dof=self.dof
        )
        self.chi_sq_mighell, _ = self.get_chi_sq(
            self.full_args, mighell_chi2, dof=self.dof
        )
        self.smooth = self._estimate_smooth(self.full_args)
        self.ys, self.zs = self._estimate_count(self.full_args)

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
