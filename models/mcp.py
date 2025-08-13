import numpy as np
from scipy.stats import gamma

from ..core.base import PMT_Fitter
from ..core.utils import compute_init
from ..core.fft_utils import roll_and_pad


class MCP_Fitter(PMT_Fitter):
    def __init__(
        self,
        hist,
        bins,
        isWholeSpectrum=False,
        A=None,
        occ_init=None,
        sample=None,
        init=[0.60, 0.6, 0.25, 5.0, 0.40, 0.65],
        bounds=[
            (0.0, 1.0),
            (0, None),
            (0, None),
            (1, None),
            (0, 1),
            (0, 1),
        ],
        constraints=[
            {"coeffs": [(1, 1), (2, -1)], "threshold": 0, "op": ">"},
        ],
        threshold=None,
        auto_init=False,
        seterr: str = "warn",
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
            **peak_kwargs,
        )

    def const(self, ser_args):
        frac, k, theta, lam, k_ts, theta_ts = ser_args
        return (1 - frac) * np.exp(-lam)

    def _map_args(self, args):
        frac, mean, sigma, lam, mean_t, sigma_t = args
        k = (mean / sigma) ** 2
        theta = mean / k
        k_ts = (mean_t / sigma_t) ** 2
        theta_ts = mean * (sigma_t**2) / mean_t
        return (frac, k, theta, lam, k_ts, theta_ts)

    def _ft_gamma(self, freq, k, theta):
        """Wow, Gamma FFT is analytic!

        Parameters
        ----------
        freq : ndarray
        k : float
        theta : float
        """
        return (1 + 1j * theta * freq) ** (-k)

    def _ser_ft(self, freq, ser_args):
        frac, k, theta, lam, k_ts, theta_ts = self._map_args(ser_args)
        ft_g = self._ft_gamma(freq, k, theta)
        ft_ts0 = self._ft_gamma(freq, k_ts, theta_ts)
        ft_tw = np.exp(-lam) * (np.exp(lam * ft_ts0) - 1)
        return frac * ft_g + (1 - frac) * ft_tw

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

    def Tws(self, args, occ):
        frac, _, _, lam, k_ts, theta_ts = self._map_args(args)
        return self.A * self._bin_width * self._pdf_tw(frac, lam, k_ts, theta_ts, occ)

    def _pdf_tw(self, frac, lam, k_ts, theta_ts, occ):
        const = np.exp(-lam)
        ft_ts0 = self._ft_gamma(self._freq, k_ts, theta_ts)
        ft_tw_nz = np.exp(-lam) * (np.exp(lam * ft_ts0) - 1) / (1 - np.exp(-lam))
        fft_input = (1 - const) * ft_tw_nz + const
        s_sp = self._nPE_processor(occ, 1)(fft_input)
        ifft_pdf = self._ifft_pipeline(s_sp)
        return (1 - frac) * ifft_pdf

    def get_gain(self, args, gain: str = "gm"):
        if gain == "gp":
            _, mean, sigma, _, _, _ = args
            k = (mean / sigma) ** 2
            theta = mean / k
            return (k - 1) * theta
        elif gain == "gm":
            frac, mean, sigma, lam, mean_t, sigma_t = args
            fracReNorm = frac / (1 - (1 - frac) * np.exp(-lam))
            return fracReNorm * mean + (1 - fracReNorm) * mean * mean_t * lam
        else:
            raise NameError(f"{gain} is not a legal gain type")

    def _zero(self, args):
        frac, _, _, lam, _, _, occ = args
        mu = -np.log(1 - occ)
        return np.exp(mu * ((1 - frac) * np.exp(-lam) - 1))

    def _replace_spe_params(self, gp_init, sigma_init, occ=0):
        # ha, some magic to correct sigma under different occupancy
        coef = 1 + np.log(1 - occ) / 4
        self._init[1] = gp_init
        self._init[2] = 0.6 * coef * sigma_init

    def _replace_spe_bounds(self, gp_bound, sigma_bound, occ=0):
        coef = 1 + np.log(1 - occ) / 4
        self.bounds[1] = (0.5 * gp_bound, 1.5 * gp_bound)
        self.bounds[2] = (0.05 * coef * sigma_bound, 3 * coef * sigma_bound)
