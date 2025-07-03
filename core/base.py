import emcee
import numpy as np

from scipy.fft import fft
from scipy.stats import norm

from .utils import (
    composite_simpson,
    isInBound,
    isParamsInBound,
    isParamsWithinConstraints,
    merge_bins,
    compute_init,
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
        Initial params of SER charge model, in the order of
        "main peak ratio, main peak k/shape, main peak theta/rate,
        secondary electron number, secondary alpha/shape, secondary beta/rate"
    bounds : ArrayLike
        Initial bounds of SER charge model
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
        seterr: str = "warn",
        init=None,
        bounds=None,
        constraints=None,
        auto_init=False,
    ):
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

        if auto_init:
            if self._isWholeSpectrum:
                ped_gp, ped_sigma = compute_init(self.hist, self.bins, peak_idx=0)
                print(f"ped: {ped_gp} ± {ped_sigma}")
                spe_gp, spe_sigma = compute_init(self.hist, self.bins, peak_idx=1)
                print(f"spe: {spe_gp} ± {spe_sigma}")
                self._replace_spe_params(spe_gp, spe_sigma)
                self._replace_spe_bounds(spe_gp, spe_sigma)
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
                spe_gp, spe_sigma = compute_init(self.hist, self.bins, peak_idx=0)
                print(f"spe: {spe_gp} ± {spe_sigma}")
                self._replace_spe_params(spe_gp, spe_sigma)
                self._replace_spe_bounds(spe_gp, spe_sigma)
                self.init = np.append(self._init, self._occ_init)
        else:
            self.init = np.append(self._init, self._occ_init)

        self.bounds.append((0, 1))
        self.bounds = tuple(self.bounds)

        self._bin_width = self.bins[1] - self.bins[0]
        self._xs = (self.bins[:-1] + self.bins[1:]) / 2
        self._interval = self._bin_width / self.sample
        self._xsp_width = self._bin_width / self.sample
        self._shift = int(round(self.bins[0] / self._xsp_width))

        self.xsp = np.linspace(
            self.bins[0] - abs(self._shift) * self._xsp_width,
            self.bins[-1],
            num=len(self.hist) * self.sample + abs(self._shift) + 1,
            endpoint=True,
        )

        print(self._shift)
        print(self.bins[0])
        print(self.xsp[abs(self._shift) + 1])

        _n_origin = len(self.xsp)
        self._pad_safe = 2 ** int(np.ceil(np.log2(_n_origin))) - _n_origin

        self.dof = len(self.init) - 1
        self._C = self._log_l_C()

    # -----------------
    # must be implemented
    # -----------------

    def _replace_spe_params(self, gp_init, sigma_init):
        raise NotImplementedError

    def _replace_spe_bounds(self, gp_bound, sigma_bound):
        raise NotImplementedError

    def _pdf(self, args):
        raise NotImplementedError

    def _zero(self, args):
        raise NotImplementedError

    def get_gain(self, args):
        raise NotImplementedError

    # --------
    # property
    # --------

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

    # ---------
    # implement
    # ---------

    def _pdf_ped(self, args):
        return norm.pdf(self.xsp, loc=args[0], scale=args[1])

    def _all_PE_processor(self, occupancy, b_sp):
        if self._isWholeSpectrum:
            return lambda s_sp: np.exp(-np.log(1 - occupancy) * (s_sp - 1)) * b_sp
        else:
            return lambda s_sp: np.exp(-np.log(1 - occupancy) * (s_sp - 1))

    def _nPE_processor(self, occupancy, n):
        return (
            lambda s_sp: (1 - occupancy)
            * ((-np.log(1 - occupancy) * s_sp) ** n)
            / np.prod(range(1, n + 1))
        )

    def _pdf_sr(self, args):
        """Applying DFT & IDFT to estimate pdf.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ) if only PE spectrum,
            (ped_mean, ped_sigma, ser_args_1, ..., ser_args_(dof-2), occ) otherwise
        """
        start_idx = 2 if self._isWholeSpectrum else 0
        ser_args = args[start_idx:-1]

        pdf = self._pdf(ser_args)
        if not np.all(np.isfinite(pdf)):
            raise ValueError("Non-finite value in PDF.")

        b_sp = None
        if self._isWholeSpectrum:
            ped_pdf_shifted, _, _ = roll_and_pad(
                self._pdf_ped(args[:start_idx]), self._shift, self._pad_safe
            )
            b_sp = fft(ped_pdf_shifted) * self._xsp_width

        return fft_and_ifft(
            pdf,
            self._shift,
            self._xsp_width,
            self._pad_safe,
            self._all_PE_processor(args[-1], b_sp),
        )

    def _pdf_sr_n(self, args, n):
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
        start_idx = 2 if self._isWholeSpectrum else 0
        ser_args = args[start_idx:-1]

        pdf = self._pdf(ser_args)
        if not np.all(np.isfinite(pdf)):
            raise ValueError("Non-finite value in PDF.")

        if n == 0:
            if self._isWholeSpectrum:
                return self._pdf_ped(args[:start_idx])
            else:
                return np.zeros_like(self.xsp)

        return fft_and_ifft(
            pdf,
            self._shift,
            self._xsp_width,
            self._pad_safe,
            self._nPE_processor(args[-1], n),
        )

    def _estimate_smooth(self, args):
        return self.A * self._bin_width * self._pdf_sr(args=args)[abs(self._shift) :]

    def estimate_smooth_n(self, args, n):
        return (
            self.A
            * self._bin_width
            * self._pdf_sr_n(args=args, n=n)[abs(self._shift) :]
        )

    def _estimate_count(self, args) -> tuple:
        """Estimate counts of every bin.

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
        """
        y_sp = self.A * self._pdf_sr(args=args)
        y_sp_slice = np.array(
            [
                y_sp[
                    abs(self._shift)
                    + self.sample * i : abs(self._shift)
                    + self.sample * (i + 1)
                    + 1
                ]
                for i in range(len(self.hist))
            ]
        )
        y_est = np.apply_along_axis(
            composite_simpson, 1, y_sp_slice, self._interval, self.sample
        )
        # nonegative pdf set
        y_est[y_est <= 0] = 1e-20
        # for whole spectrum, z_est doesn't matter because self.zero is 0
        # otherwise, only SPE parameters are needed to calculate z_est
        start_idx = 2 if self._isWholeSpectrum else 0
        z_est = self.A * self._zero(args[start_idx:])
        return y_est, z_est

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
            if isParamsInBound(args, self.bounds) and isParamsWithinConstraints(
                args, self.constraints
            ):
                y, z = self._estimate_count(args)
                return self.zero * np.log(z) + np.sum(self.hist * np.log(y)) - self._C
            else:
                return -np.inf
        except ValueError as e:
            if self.seterr != "ignore":
                print(
                    "[WARNING] Some chain(s) have Inf/NaN PDF value(s). Please improve the PDF robustness of your model."
                )
            return -np.inf

    def get_chi_sq(self, args) -> float:
        """Chi square.

        Parameters
        ----------
        ser_args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ) if only PE spectrum,
            (ped_mean, ped_sigma, ser_args_1, ..., ser_args_(dof-2), occ) otherwise
        """
        y, z = self._estimate_count(args)
        hist_reg, y_reg = merge_bins(self.hist, y)
        self.ndf = len(hist_reg) - self.dof
        return sum((y_reg - hist_reg) ** 2 / y_reg) + (z - self.zero) ** 2 / z

    def fit(
        self,
        nwalkers: int = 32,
        burn_in: int = 50,
        step: int = 200,
        seed: int = None,
        track: int = 1,
        step_length: list | np.ndarray = None,
    ):
        """MCMC fit using `emcee`.

        Parameters
        ----------
        nwalkers : int
            Number of parallel chains for `emcee`.
        burn_in : int
            Burn in step for `emcee`.
        step : int
            MCMC step for `emcee`.
        seed : int
            Seed for random.
        track : int
            Take only every `track` steps from the chain.
        step_length : ndarray[float]
            Step length to generate initial values.

        Notes
        -----
        `nwalkers >= 2 * ndim`, credits to Xuewei.
        """
        if seed is not None:
            np.random.seed(seed)

        ndim = self.dof + 1
        p0 = self.init + np.random.uniform(-1, 1, (nwalkers, ndim)) * step_length

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_l, moves=emcee.moves.WalkMove()
        )
        sampler.run_mcmc(p0, step)

        acceptance = sampler.acceptance_fraction
        # autocorr_time = sampler.get_autocorr_time(discard=burn_in)

        self.log_l_track = sampler.get_log_prob(thin=track)  # (step, nwalkers)
        # select the max log-likelihood (effective steps) chain
        ind = np.argmax(np.mean(self.log_l_track[burn_in - step :, :], axis=0))
        self.samples_track = sampler.get_chain(discard=burn_in)[
            :, ind, :
        ]  # (step, ndim)

        args_complete = np.mean(self.samples_track, axis=0)
        args_complete_std = np.std(self.samples_track, axis=0)

        start_idx = 2 if self._isWholeSpectrum else 0
        self.ser_args = args_complete[start_idx:-1]
        self.ser_args_std = args_complete_std[start_idx:-1]
        self.ped_args = args_complete[:start_idx] if self._isWholeSpectrum else None
        self.ped_args_std = (
            args_complete_std[:start_idx] if self._isWholeSpectrum else None
        )

        # _zero() is a fix of real zero count
        occReg = 1 - np.apply_along_axis(
            self._zero, axis=1, arr=self.samples_track[:, start_idx:]
        )
        self.occ = np.mean(occReg, axis=0)
        self.occ_std = np.std(occReg, axis=0)

        self.gps = np.apply_along_axis(
            self.get_gain, axis=1, arr=self.samples_track[:, start_idx:-1], gain="gp"
        )
        self.gms = np.apply_along_axis(
            self.get_gain, axis=1, arr=self.samples_track[:, start_idx:-1], gain="gm"
        )

        print("----------")
        # print(f'Mean autocorrelation time: {autocorr_time} steps')
        print(f"Current burn-in: {burn_in} steps")
        print(f"Mean acceptance fraction: {np.mean(acceptance):.3f}")
        print(f"Acceptance percentile: {np.percentile(acceptance, [25, 50, 75])}")
        print("----------")
        print("Init params: " + ", ".join([f"{e:.4g}" for e in self.init]))

        if self._isWholeSpectrum:
            print(
                "Pedetal params: "
                + ", ".join(
                    [
                        f"{e:.4g} pm {f:.4g}"
                        for e, f in zip(self.ped_args, self.ped_args_std)
                    ]
                )
            )

        print(
            "SER params: "
            + ", ".join(
                [
                    f"{e:.4g} pm {f:.4g}"
                    for e, f in zip(self.ser_args, self.ser_args_std)
                ]
            )
        )
        print("occ: " + ", ".join([f"{self.occ:.4g} pm {self.occ_std:.4g}"]))

        self.likelihood = self.log_l(args_complete)
        self.BIC = ndim * np.log(len(self.hist) + 1) - 2 * self.likelihood
        self.chi_sq = self.get_chi_sq(args_complete)
        self.smooth = self._estimate_smooth(args_complete)
        self.ys, self.zs = self._estimate_count(args_complete)
