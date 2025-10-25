import numpy as np
from scipy.stats import gamma
from ..core.base import PMT_Fitter


class BiPolya_Fitter(PMT_Fitter):
    def __init__(
        self,
        hist,
        bins,
        isWholeSpectrum=False,
        A=None,
        occ_init=None,
        sample=None,
        init=[
            0.10,  # P(missing 1st dynode)
            0.8,  # Normal mean
            0.25,  # Normal sigma
            0.5,  # Missing mean / Q0
            0.1,  # Missing sigma / Q0
        ],
        bounds=[(0, 1), (1, None), (0, None), (0, 1), (0, 1)],
        constraints=None,
        threshold=None,
        auto_init=False,
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

    def _ft_gamma(self, freq, k, theta):
        return (1 + 1j * theta * freq) ** (-k)

    def Missing(self, args, occ):
        frac, mean, sigma, mean_t, sigma_t = args
        k_ts = (mean_t / sigma_t) ** 2
        theta_ts = mean * (sigma_t**2) / mean_t
        return self.A * self._bin_width * self._pdf_gm(frac, k_ts, theta_ts, occ)

    def Normal(self, args, occ):
        frac, mean, sigma, mean_t, sigma_t = args
        k = (mean / sigma) ** 2
        theta = mean / k
        return self.A * self._bin_width * self._pdf_gm(1 - frac, k, theta, occ)

    def _pdf_gm(self, frac, k, theta, occ):
        fft_input = self._ft_gamma(self._freq, k, theta)
        s_sp = self._nPE_processor(occ, 1)(fft_input)
        ifft_pdf = self._ifft_pipeline(s_sp)
        return frac * ifft_pdf

    def _ser_ft(self, freq, ser_args):
        frac, mean, sigma, mean_t, sigma_t = ser_args
        k = (mean / sigma) ** 2
        theta = mean / k
        k_ts = (mean_t / sigma_t) ** 2
        theta_ts = mean * (sigma_t**2) / mean_t
        ft_g = self._ft_gamma(freq, k, theta)
        ft_ts0 = self._ft_gamma(freq, k_ts, theta_ts)
        return (1 - frac) * ft_g + frac * ft_ts0

    def get_gain(self, args, gain: str = "gm"):
        if gain == "gp":
            frac, mean, sigma, mean_t, sigma_t = args
            k = (mean / sigma) ** 2
            theta = mean / k
            return (k - 1) * theta
        elif gain == "gm":
            frac, mean, sigma, mean_t, sigma_t = args
            return (1 - frac) * mean + frac * mean * mean_t
        else:
            raise NameError(f"{gain} is not a legal gain type")

    def _replace_spe_params(self, gp_init, sigma_init, occ=0):
        # ha, some magic to correct sigma under different occupancy
        coef = 1 + np.log(1 - occ) / 4
        self._init[1] = gp_init
        self._init[2] = coef * sigma_init

    def _replace_spe_bounds(self, gp_bound, sigma_bound, occ=0):
        coef = 1 + np.log(1 - occ) / 4
        self.bounds[1] = (0.5 * gp_bound, 1.5 * gp_bound)
        self.bounds[2] = (0.2 * coef * sigma_bound, 2.0 * coef * sigma_bound)
