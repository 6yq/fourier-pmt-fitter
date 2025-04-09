import time
import emcee
import numpy as np

from scipy.stats import gamma
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
from tweedie_pdf import tweedie_reckon


class PMT_Fitter:
    def __init__(
        self,
        hist,
        bins,
        A,
        occ_init=None,
        sample=None,
        seterr: str = "warn",
        init=None,
        bounds=None,
    ):
        np.seterr(all=seterr)

        self.A = A
        self.bounds = (
            bounds.tolist() if isinstance(bounds, np.ndarray) else list(bounds)
        )
        self._init = init if isinstance(init, np.ndarray) else np.array(init)
        self._occ_init = sum(hist) / A if occ_init is None else occ_init
        # see https://gitlab.airelinux.org/juno/OSIRIS/production/-/issues/48#note_42922
        self.sample = (
            16 * int(1 / (1 - self._occ_init) ** 0.673313) if sample is None else sample
        )

        self.init = np.append(self._init, self._occ_init)
        self.bounds.append((0, 1))
        self.bounds = tuple(self.bounds)

        self.hist = hist if isinstance(hist, np.ndarray) else np.array(hist)
        self.bins = bins if isinstance(bins, np.ndarray) else np.array(bins)
        self.zero = A - sum(self.hist)
        # there is no need to check these variables outside the class
        self._bin_width = self.bins[1] - self.bins[0]
        self._xs = (self.bins[:-1] + self.bins[1:]) / 2
        self._interval = (self.bins[1] - self.bins[0]) / self.sample
        self._bwds = 1 - int(-self.bins[0] // self._interval)
        self.xsp = np.linspace(
            self.bins[0] - self._bwds * self._interval,
            self.bins[-1],
            num=len(self.hist) * self.sample + self._bwds + 1,
            endpoint=True,
        )
        self._xsp_width = self.xsp[1] - self.xsp[0]

        self.dof = len(init)
        self.C = self._log_l_C()
        self._start = time.time()

    # ------------
    # staticmethod
    # ------------

    def composite_simpson(self, pdf_slice, interval, sample):
        """Use composite Simpson to integrate pdf.

        Parameters
        ----------
        pdf_slice : ArrayLike
            a pdf list/array/...
        """
        result = pdf_slice[0] + pdf_slice[-1]
        odd_sum = sum(pdf_slice[i] for i in range(1, sample, 2))
        even_sum = sum(pdf_slice[i] for i in range(2, sample, 2))
        result += 4 * odd_sum + 2 * even_sum
        result *= interval / 3
        return result

    def isInBound(self, param: float | int, bound: tuple[None | float | int]) -> bool:
        assert len(bound) == 2, "Illegal bound!"
        if None not in bound:
            lower, upper = bound
            assert lower <= upper, "Illegal order of bound!"
            return (param >= lower) & (param <= upper)
        elif bound == (None, None):
            return True
        elif bound[0] is None:
            return param < bound[1]
        else:
            return param > bound[0]

    def isParamsInBound(self, params, bounds):
        flag = True
        for p, b in zip(params, bounds):
            flag = flag & self.isInBound(param=p, bound=b)
            if flag == False:
                return False
        return flag

    def merge_bins(self, hist, y, threshold=5):
        hist_ = hist.copy()
        y_ = y.copy()

        while True:
            peak_idx = np.argmax(hist_)
            idx = np.where(hist_ <= threshold)[0]
            if idx.size == 0:
                break

            idx = idx[0]
            merged = False

            if idx < peak_idx:
                hist_tmp = np.append(hist_[:idx], sum(hist_[idx : idx + 2]))
                hist_ = np.append(hist_tmp, hist_[idx + 2 :])

                y_tmp = np.append(y_[:idx], sum(y_[idx : idx + 2]))
                y_ = np.append(y_tmp, y_[idx + 2 :])

                merged = True
            elif idx > peak_idx:
                idx = np.where(hist_ <= threshold)[0][-1]
                hist_tmp = np.append(hist_[: idx - 1], sum(hist_[idx - 1 : idx + 1]))
                hist_ = np.append(hist_tmp, hist_[idx + 1 :])

                y_tmp = np.append(y_[: idx - 1], sum(y_[idx - 1 : idx + 1]))
                y_ = np.append(y_tmp, y_[idx + 1 :])

                merged = True
            if not merged:
                break

        return hist_, y_

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

    # -----------
    # classmethod
    # -----------

    def _pdf(self, args):
        pass

    def _zero(self, args):
        return 1 - args[-1]

    def const(self, args):
        return 0

    def get_gain(self, args):
        pass

    def _pdf_sr(self, args):
        """Applying DFT & IDFT to estimate pdf.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ)
        """
        ser_args = args[: self.dof]
        s_sp = fft(self._pdf(ser_args)) * self._xsp_width + self.const(ser_args)
        sr_sp = np.exp(-np.log(1 - args[self.dof]) * (s_sp - 1))
        sr_sp = np.real(ifft(sr_sp)) / self._xsp_width
        return sr_sp

    def _pdf_sr_n(self, args, n):
        """Return n-order pdf.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ)
        n : int
            nPE

        Notes
        -----
        nPE contributes exp(-mu) / k! * [mu * s_sp]^k
        """
        ser_args = args[: self.dof]
        s_sp = fft(self._pdf(ser_args)) * self._xsp_width + self.const(ser_args)
        mu = -np.log(1 - args[self.dof])
        sr_sp_n = np.exp(-mu) * (mu * s_sp) ** n / np.prod(range(1, n + 1))
        sr_sp_n = np.real(ifft(sr_sp_n)) / self._xsp_width
        return sr_sp_n

    def _estimate_smooth(self, args):
        return self.A * self._bin_width * self._pdf_sr(args=args)

    def estimate_smooth_n(self, args, n):
        return self.A * self._bin_width * self._pdf_sr_n(args=args, n=n)

    def _estimate_count(self, args) -> tuple:
        """Estimate counts of every bin.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ)

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
                    (self._bwds + self.sample * i) : (
                        self._bwds + self.sample * (i + 1) + 1
                    )
                ]
                for i in range(len(self.hist))
            ]
        )
        y_est = np.apply_along_axis(
            self.composite_simpson, 1, y_sp_slice, self._interval, self.sample
        )
        # nonegative pdf set
        y_est[y_est <= 0] = 1e-16
        z_est = self.A * self._zero(args)
        return y_est, z_est

    def log_l(self, args) -> float:
        """log likelihood of given args.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occ)
        """
        # make sure args are in range
        # otherwise an infinite "well"
        if self.isParamsInBound(args, self.bounds):
            y, z = self._estimate_count(args)
            return self.zero * np.log(z) + np.sum(self.hist * np.log(y)) - self.C
        else:
            return -np.inf

    def get_chi_sq(self, args) -> float:
        """Chi square.

        Parameters
        ----------
        ser_args : ArrayLike
        occs : ArrayLike
        """
        y, z = self._estimate_count(args)
        hist_reg, y_reg = self.merge_bins(self.hist, y)
        self.ndf = len(hist_reg) - self.dof
        return sum((y_reg - hist_reg) ** 2 / y_reg) + (z - self.zero) ** 2 / z

    def fit(
        self,
        nwalkers: int = 32,
        burn_in: int = 50,
        step: int = 200,
        seed: int = None,
        track: int = 1,
        step_length: dict[str, float] = None,
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
        # select the max log-likelihood (last 1000 steps) chain
        ind = np.argmax(np.mean(self.log_l_track[-1000:, :], axis=0))
        self.samples_track = sampler.get_chain(discard=burn_in)[
            :, ind, :
        ]  # (step, ndim)
        self.ser_args = np.mean(self.samples_track[:, : self.dof], axis=0)
        self.ser_args_std = np.std(self.samples_track[:, : self.dof], axis=0)
        self.occ = np.mean(self.samples_track[:, -1], axis=0)
        self.occ_std = np.std(self.samples_track[:, -1], axis=0)
        args_complete = np.append(self.ser_args, self.occ)

        self.gps = np.apply_along_axis(
            self.get_gain, axis=1, arr=self.samples_track, gain="gp"
        )
        self.gms = np.apply_along_axis(
            self.get_gain, axis=1, arr=self.samples_track, gain="gm"
        )

        print("----------")
        # print(f'Mean autocorrelation time: {autocorr_time} steps')
        print(f"Current burn-in: {burn_in} steps")
        print(f"Mean acceptance fraction: {np.mean(acceptance):.3f}")
        print(f"Acceptance percentile: {np.percentile(acceptance, [25, 50, 75])}")
        print("----------")
        print("Init params: " + ", ".join([f"{e:.4g}" for e in self.init]))
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


class MCP_Fitter(PMT_Fitter):
    """A class to fit MCP-PMT charge spectrum.

    Parameters
    ----------
    charge : ArrayLike
        charge dataset input
    A : ArrayLike
        total charge count
    occ_init : ArrayLike
        initial occupancy
    sample : int
        the number of sample intervals between bins
    cut : float or tuple[float]
        the upper cut (lower cut optional), expressed with the ratio divided by main peak
    init : ArrayLike
        initial params of SER charge model, in the order of
        "main peak ratio, main peak k/shape, main peak theta/rate,
        secondary electron number, secondary alpha/shape, secondary beta/rate"
    bounds : ArrayLike
        initial bounds of SER charge model, secondary within 3 sigma
    seterr : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        see https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
    """

    def __init__(
        self,
        hist,
        bins,
        A,
        occ_init=None,
        sample=None,
        seterr: str = "warn",
        init=[
            0.65,  # main peak ratio
            400,  # main peak mean
            100,  # main peak sigma
            4.0,  # secondary electron number
            0.60,  # secondary electron mean / Q1
            0.15,  # secondary electron std variance / Q1
        ],
        bounds=[
            (0.35, 1),
            (0, None),  # mean
            (0, None),  # sigma
            (3, 7),  # secondary
            (0.3, 0.7),  # mean / Q1 based on Jun's work
            (0.05, 0.3),  # std variance / Q1 based on Jun's work
        ],
    ):
        super().__init__(hist, bins, A, occ_init, sample, seterr, init, bounds)

    # ------------
    # staticmethod
    # ------------

    def const(self, args):
        return (1 - args[0]) * np.exp(-args[3])

    def get_gain(self, args, gain: str = "gm"):
        """Return gain of MCP.

        Parameters
        ----------
        ser_args : ArrayLike.
            Elements:
                frac : float
                    ratio of peak
                mean : float
                    main peak, mean in Gamma distribution
                sigma : float
                    main peak, sigma in Gamma distribution
                lam : float
                    mean of secondary electron numbers
                mean_t : float
                    calibrated on 8-inches, the mean of secondary Gamma gain / Q1
                sigma_t : float
                    calibrated on 8-inches, the std variance of secondary Gamma gain / Q1

        Return
        ------
        gain : float
            Gain of MCP.

        Notes
        -----
        Return mean of SPE distribution (Gm) by default.
        """
        if gain == "gp":
            return args[1]
        elif gain == "gm":
            return args[0] * args[1] + (1 - args[0]) * args[1] * args[3] * args[4]
        else:
            raise NameError(f"{gain} is not a illegal parameter!")

    def _map_args(self, args) -> tuple:
        """Map MCP-PMT SER/SPE charge model parameters.

        Parameters
        ----------
        args : ArrayLike.
            Elements:
                frac : float
                    ratio of peak
                mean : float
                    main peak, mean in Gamma distribution
                sigma : float
                    main peak, sigma in Gamma distribution
                lam : float
                    mean of secondary electron numbers
                mean_t : float
                    calibrated on 8-inches, the mean of secondary Gamma gain / Q1
                sigma_t : float
                    calibrated on 8-inches, the std variance of secondary Gamma gain / Q1

        Return
        ------
        result : tuple
            Elements:
                frac : float
                    ratio of peak
                shape : float
                    main peak, shape in Gamma distribution
                scale : float
                    main peak, scale in Gamma distribution
                mu : float
                    mean of tweedie (secondary electron charge)
                p : float
                    power of tweedie (1 < p < 2)
                phi : float
                    variance param of tweedie (var = phi * mu^p)

        Notes
        -----
        True secondary (ts) `q/q1 ~ GA(alpha_ts, beta_ts)` -> `q ~ GA(alpha_ts, beta_ts / q1)`;
        True secondary parameters mapping see:
        https://en.wikipedia.org/wiki/Compound_Poisson_distribution
        """
        frac, mean, sigma, lam, mean_t, sigma_t = args
        alpha_ts = (mean_t**2) / (sigma_t**2)
        beta_ts = mean_t / (sigma_t**2)
        k = (mean / sigma) ** 2
        theta = mean / k
        mu = lam * alpha_ts * mean / beta_ts
        p = 1 + 1 / (alpha_ts + 1)
        phi = (alpha_ts + 1) * pow(lam * alpha_ts, 1 - p) / pow(beta_ts / mean, 2 - p)
        return (frac, k, theta, mu, p, phi)

    def _pdf_gm(self, x, frac, k, theta):
        return frac * gamma.pdf(x, a=k, scale=theta)

    def _pdf_tw(self, x, frac, mu, p, phi):
        inreg = sum(x <= 0)
        pdf = (1 - frac) * tweedie_reckon(
            x[inreg:], p=p, mu=mu, phi=phi, dlambda=False
        )[0]
        for _ in range(inreg):
            pdf = np.insert(pdf, 0, 0)
        return pdf

    # --------
    # property
    # --------

    def Gms(self, args, occ):
        frac, mean, sigma = args[:3]
        k = (mean / sigma) ** 2
        theta = mean / k
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_gm(self.xsp, frac=frac, k=k, theta=theta)
        )

    def Tws(self, args, occ):
        frac, _, _, mu, p, phi = self._map_args(args)
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_tw(self.xsp, frac=frac, mu=mu, p=p, phi=phi)
        )

    # -----------
    # classmethod
    # -----------

    def _pdf(self, args):
        frac, k, theta, mu, p, phi = self._map_args(args)
        return self._pdf_gm(self.xsp, frac, k, theta) + self._pdf_tw(
            self.xsp, frac, mu, p, phi
        )

    def _zero(self, args):
        """MCP-PMT SER/SPE charge model zero charge count.

        Parameters
        ----------
        args : ArrayLike
            (frac, k, theta, lam, mean, std_variance, occupancy)
        """
        frac, _, _, lam, _, _ = args[: self.dof]
        mu = -np.log(1 - args[self.dof])
        return np.exp(mu * ((1 - frac) * np.exp(-lam) - 1))


class Dynode_Fitter(PMT_Fitter):
    """A class to fit Dynode PMT charge spectrum.

    Parameters
    ----------
    charge : ArrayLike
        charge dataset input
    A : ArrayLike
        total charge count
    occ_init : ArrayLike
        initial occupancy
    sample : int
        the number of sample intervals between bins
    cut : float or tuple[float]
        the upper cut (lower cut optional), expressed with the ratio divided by main peak
    init : ArrayLike
        initial params of SER charge model, in the order of
        "peak shape, peak rate"
    bounds : ArrayLike
        initial bounds of SER charge model, secondary within 3 sigma
    seterr : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        see https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
    """

    def __init__(
        self,
        hist,
        bins,
        A,
        occ_init,
        sample=None,
        seterr: str = "warn",
        init=[
            600,  # peak mean
            40,  # peak sigma
        ],
        bounds=[
            (0, None),
            (0, None),
        ],
    ):
        super().__init__(hist, bins, A, occ_init, sample, seterr, init, bounds)

    # ------------
    # staticmethod
    # ------------

    def get_gain(self, args, gain: str = "gm"):
        """Return gain of MCP.

        Parameters
        ----------
        ser_args : ArrayLike.
            Elements:
                shape : float
                    peak shape in Gamma distribution
                scale : float
                    peak scale in Gamma distribution

        Return
        ------
        gain : float
            Gain of dynode PMT.

        Notes
        -----
        Return mean of SPE distribution (Gm) by default.
        """
        if gain == "gp" or gain == "gm":
            return args[0]
        else:
            raise NameError(f"{gain} is not a illegal parameter!")

    # -----------
    # classmethod
    # -----------

    def _pdf(self, args):
        mean, sigma = args
        k = (mean / sigma) ** 2
        theta = mean / k
        return gamma.pdf(self.xsp, a=k, scale=theta)


class Dynode_Fitter_Opt(MCP_Fitter):
    """A class to fit Dynode PMT charge spectrum.

    Parameters
    ----------
    charge : ArrayLike
        charge dataset input
    A : ArrayLike
        total charge count
    occ_init : ArrayLike
        initial occupancy
    sample : int
        the number of sample intervals between bins
    cut : float or tuple[float]
        the upper cut (lower cut optional), expressed with the ratio divided by main peak
    init : ArrayLike
        initial params of SER charge model, in the order of
        "peak shape, peak rate"
    bounds : ArrayLike
        initial bounds of SER charge model, secondary within 3 sigma
    seterr : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        see https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
    """

    def __init__(
        self,
        hist,
        bins,
        A,
        occ_init,
        sample=None,
        seterr: str = "warn",
        init=[
            0.9,  # normal applified ratio
            600,  # peak mean
            40,  # peak sigma
            0.6,  # mis-applified mean ratio
            0.6,  # mis-applified sigma ratio
        ],
        bounds=[
            (0, 1),
            (0, None),
            (0, None),
            (0, 1),
            (0, 1),
        ],
    ):
        super().__init__(hist, bins, A, occ_init, sample, seterr, init, bounds)

    # ------------
    # staticmethod
    # ------------

    def get_gain(self, args, gain: str = "gm"):
        """Return gain of MCP.

        Parameters
        ----------
        ser_args : ArrayLike.
            Elements:
                ratio : float
                    mis-applified ratio
                mean : float
                    peak mean in Gamma distribution
                sigma : float
                    peak sigma in Gamma distribution
                mean_t : float
                    mis-applified mean
                sigma_t : float
                    mis-applified sigma

        Return
        ------
        gain : float
            Gain of dynode PMT.

        Notes
        -----
        Return mean of SPE distribution (Gm) by default.
        """
        if gain == "gp":
            return args[1]
        elif gain == "gm":
            return args[0] * args[1] * args[3] + (1 - args[0]) * args[1]
        else:
            raise NameError(f"{gain} is not a illegal parameter!")

    def _zero(self, args):
        return 1 - args[-1]

    # -----------
    # classmethod
    # -----------

    def _map_args(self, args) -> tuple:
        """
        Parameters
        ----------
        ser_args : ArrayLike.
            Elements:
                ratio : float
                    normal applified ratio
                mean : float
                    peak mean in Gamma distribution
                sigma : float
                    peak sigma in Gamma distribution
                mean_t : float
                    mis-applified mean
                sigma_t : float
                    mis-applified sigma

        Notes
        -----
        Insert lam=1 at the right position for MCP_Fitter.
        """
        args_with_lam = args.tolist()
        args_with_lam.insert(3, 1.0)  # lam = 1
        return super()._map_args(args_with_lam)


class Mixture_Fitter(PMT_Fitter):
    """A class to fit Dynode PMT charge spectrum.

    Parameters
    ----------
    charge : ArrayLike
        charge dataset input
    A : ArrayLike
        total charge count
    occ_init : ArrayLike
        initial occupancy
    sample : int
        the number of sample intervals between bins
    cut : float or tuple[float]
        the upper cut (lower cut optional), expressed with the ratio divided by main peak
    init : ArrayLike
        initial params of SER charge model, in the order of
        "peak shape, peak rate"
    bounds : ArrayLike
        initial bounds of SER charge model, secondary within 3 sigma
    seterr : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        see https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
    """

    def __init__(
        self,
        hist,
        bins,
        A,
        occ_init,
        sample=None,
        cut: float | tuple[float] = (0.2, 5),
        seterr: str = "warn",
        init=[
            5e-04,
            800,
            200,
        ],
        bounds=[
            (1e-04, 3e-03),
            (600, 1200),
            (0, 400),
        ],
    ):
        super().__init__(hist, bins, A, occ_init, sample, cut, seterr, init, bounds)

    # ------------
    # staticmethod
    # ------------

    def get_gain(self, args, gain: str = "gm"):
        """Return gain of MCP.

        Parameters
        ----------
        ser_args : ArrayLike.
            Elements:
                p : float
                    propotion of exponential
                G : float
                    gain
                sigma : float
                    sigma of gain

        Return
        ------
        gain : float
            Gain of dynode PMT.

        Notes
        -----
        Return mean of SPE distribution (Gm) by default.
        """
        if gain == "gp":
            return args[1]
        elif gain == "gm":
            return args[0] * args[1] + (1 - args[0]) * args[1] * 2.3
        else:
            raise NameError(f"{gain} is not a illegal parameter!")

    # -----------
    # classmethod
    # -----------

    def _pdf(self, args):
        p, G, sigma = args
        return p * expon.pdf(self.xsp / G, loc=0.1, scale=2.2) + (1 - p) * norm.pdf(
            self.xsp, loc=G, scale=sigma
        )

    def exps(self, args, A):
        p, G, sigma = args
        return (
            A
            * self._bin_width
            * p
            * expon.pdf(self.xsp / G, loc=0.1, scale=2.2)
            * self._xsp_width
        )

    def norms(self, args, A):
        p, G, sigma = args
        return (
            A
            * self._bin_width
            * (1 - p)
            * norm.pdf(self.xsp, loc=G, scale=sigma)
            * self._xsp_width
        )
