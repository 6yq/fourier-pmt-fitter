import numpy as np
from scipy.stats import norm
from ..core.base import PMT_Fitter


class BiGauss_Fitter(PMT_Fitter):
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

    def _pdf_normal(self, x, ratio, mean, sigma):
        inv = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        return (1 - ratio) * inv * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def _pdf_missing(self, x, ratio, mean, mean_r, sigma_r):
        mean_ = mean * mean_r
        sigma_ = mean * sigma_r
        inv = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_)
        return ratio * inv * np.exp(-0.5 * ((x - mean_) / sigma_) ** 2)

    def _ser_pdf_time(self, args):
        ratio, mean, sigma, mean_r, sigma_r = args
        return self._pdf_normal(self.xsp, ratio, mean, sigma) + self._pdf_missing(
            self.xsp, ratio, mean, mean_r, sigma_r
        )

    def Normal(self, args, occ):
        ratio, mean, sigma, _, _ = args
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_normal(self.xsp, ratio, mean, sigma)
        )

    def Missing(self, args, occ):
        ratio, mean, _, mean_r, sigma_r = args
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_missing(self.xsp, ratio, mean, mean_r, sigma_r)
        )

    def get_gain(self, args, gain: str = "gm"):
        if gain == "gp":
            _, mean, _, _, _ = args
            return mean
        elif gain == "gm":
            ratio, mean, _, mean_r, _ = args
            return mean * (1 - ratio * (1 - mean_r))
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
