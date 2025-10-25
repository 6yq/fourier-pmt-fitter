import numpy as np
from scipy.stats import gamma
from scipy.special import lambertw

from math import exp, log
from ..core.base import PMT_Fitter
from ..core.utils import compute_init
from ..core.fft_utils import roll_and_pad


class Recursive_Fitter(PMT_Fitter):
    def __init__(
        self,
        hist,
        bins,
        isWholeSpectrum=False,
        A=None,
        occ_init=None,
        sample=None,
        init=[
            0.40,  # channel-mode ratio
            0.6,  # channel-mode mean
            0.25,  # channel-mode sigma
            4.5,  # channel-mode sey
            1.0,  # recursive sey
            0.60,  # recursive mean
            0.20,  # recursive sigma
        ],
        bounds=[
            (0.0, 1.0),
            (0, None),
            (0, None),
            (1, None),
            (0, None),
            (0, 1),
            (0, 1),
        ],
        constraints=[
            {"coeffs": [(1, 1), (2, -1)], "threshold": 0, "op": ">"},
        ],
        threshold: str = None,
        auto_init: bool = False,
        seterr: str = "warn",
        fit_total: bool = True,
        **peak_kwargs,
    ):
        super().__init__(
            hist,
            bins,
            isWholeSpectrum,
            A,
            occ_init,
            sample,
            init,
            bounds,
            constraints,
            threshold,
            auto_init,
            seterr,
            fit_total,
            **peak_kwargs,
        )

    def const_r(self, ser_args):
        """p_s"""
        frac, _, _, _, lam_r, _, _ = ser_args
        return -lambertw(lam_r * (frac - 1) * exp(-lam_r)).real / lam_r

    def const(self, ser_args):
        """p_S"""
        frac, _, _, lam, _, _, _ = ser_args
        const_r = self.const_r(ser_args)
        return (1 - frac) * exp(lam * (const_r - 1))

    def _map_args(self, args):
        frac, mean, sigma, lam, lam_r, mean_r, sigma_r = args
        k = (mean / sigma) ** 2
        theta = mean / k
        k_r = (mean_r / sigma_r) ** 2
        theta_r = mean * (sigma_r**2) / mean_r
        return (frac, k, theta, lam, lam_r, k_r, theta_r)

    def _ft_gamma(self, freq, k, theta):
        """Wow, Gamma FFT is analytic!

        Parameters
        ----------
        freq : ndarray
        k : float
        theta : float
        """
        return (1 + 1j * theta * freq) ** (-k)

    def _ser_ft_r(self, freq, ser_args):
        """s"""
        frac, k, theta, lam, lam_r, k_r, theta_r = self._map_args(ser_args)
        ft_r = self._ft_gamma(freq, k_r, theta_r)
        return (
            frac * ft_r
            - lambertw(lam_r * (frac - 1) * np.exp(lam_r * (frac * ft_r - 1))) / lam_r
        )

    def _ser_ft(self, freq, ser_args):
        """S"""
        frac, k, theta, lam, _, _, _ = self._map_args(ser_args)
        ft_g = self._ft_gamma(freq, k, theta)
        ft_r = self._ser_ft_r(freq, ser_args)
        return (
            frac * ft_g + (1 - frac) * np.exp(lam * (ft_r - 1)) - self.const(ser_args)
        )

    def Gms(self, args, occ):
        frac, mean, sigma = args[:3]
        k = (mean / sigma) ** 2
        theta = mean / k
        return self.A * self._bin_width * self._pdf_gm(frac, k, theta, occ)

    def _pdf_gm(self, frac, k, theta, occ):
        fft_input = self._ft_gamma(self._freq, k, theta)
        s_sp = self._nPE_processor(occ, 1)(fft_input)
        ifft_pdf = self._ifft_pipeline(s_sp)
        return frac * ifft_pdf

    def Recurs(self, args, occ):
        frac, _, _, lam, _, _, _ = args
        ft_r = self._ser_ft_r(self._freq, args)
        fft_input = np.exp(lam * (ft_r - 1))
        s_sp = self._nPE_processor(occ, 1)(fft_input)
        ifft_pdf = self._ifft_pipeline(s_sp)
        return self.A * self._bin_width * (1 - frac) * ifft_pdf

    def get_gain(self, args, gain: str = "gm"):
        if gain == "gp":
            _, mean, sigma, _, _, _, _ = args
            k = (mean / sigma) ** 2
            theta = mean / k
            return (k - 1) * theta
        elif gain == "gm":
            frac, mean, sigma, lam, lam_r, mean_r, sigma_r = args
            mu1 = mean
            mu2 = mean * mean_r
            denom = 1 - (1 - frac) * lam_r
            Es = frac * mu2 / denom
            ES = frac * mu1 + (1 - frac) * lam_r * Es
            # E[S | S > 0] = E[S] / (1 - p_S)
            pS = self.const(args)
            return ES / (1 - pS)
        else:
            raise NameError(f"{gain} is not a legal gain type")

    def _zero(self, args):
        mu = -log(1 - args[-1])
        return exp(mu * (self.const(args[:-1]) - 1))

    def _replace_spe_params(self, gp_init, sigma_init, occ=0):
        # ha, some magic to correct sigma under different occupancy
        coef = 1 + np.log(1 - occ) / 4
        self._init[1] = gp_init
        self._init[2] = 0.6 * coef * sigma_init

    def _replace_spe_bounds(self, gp_bound, sigma_bound, occ=0):
        coef = 1 + np.log(1 - occ) / 4
        self.bounds[1] = (0.5 * gp_bound, 1.5 * gp_bound)
        self.bounds[2] = (0.05 * coef * sigma_bound, 3 * coef * sigma_bound)
