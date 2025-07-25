import numpy as np
from scipy.stats import gamma, norm
from ..core.base import PMT_Fitter


class Dynode_Fitter(PMT_Fitter):
    """
    Dynode Normal plus Missing 1st model.

    Notes
    -----
    - Introduced to JUNO by Zhangming, Junting et al;
    - First proposed by K. Lang, J. Day, S. Eilerts et al;
    - See JUNO-doc-13627, 14075.
    """

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
            2.55,  # P(multiplication missing)
            0.8,  # Normal mean
            0.25,  # Normal sigma
        ],
        bounds=[
            (0, 1),
            (1, None),
            (0, None),
            (0, None),
        ],
        constraints=None,
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

    def _pdf_normal(self, x, df, mean, sigma):
        inv = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        return (1 - df) * inv * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def _pdf_missing(self, x, df, ds, mean, sigma):
        mean_ = mean / ds
        sigma_ = sigma / ds
        inv = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_)
        return df * inv * np.exp(-0.5 * ((x - mean_) / sigma_) ** 2)

    def _pdf(self, args):
        df, ds, mean, sigma = args
        return self._pdf_normal(self.xsp, df, mean, sigma) + self._pdf_missing(
            self.xsp, df, ds, mean, sigma
        )

    def Normal(self, args, occ):
        df, _, mean, sigma = args
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_normal(self.xsp, df, mean, sigma)
        )

    def Missing(self, args, occ):
        df, ds, mean, sigma = args
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_missing(self.xsp, df, ds, mean, sigma)
        )

    def get_gain(self, args, gain: str = "gm"):
        if gain == "gp":
            _, _, mean, _ = args
            return mean
        elif gain == "gm":
            df, ds, mean, _ = args
            return df * mean / ds + (1 - df) * mean
        else:
            raise NameError(f"{gain} is not a legal gain type")

    def _replace_spe_params(self, gp_init, sigma_init, occ=0):
        # ha, some magic to correct sigma under different occupancy
        coef = 1 + np.log(1 - occ) / 4
        self._init[2] = gp_init
        self._init[3] = coef * sigma_init

    def _replace_spe_bounds(self, gp_bound, sigma_bound, occ=0):
        coef = 1 + np.log(1 - occ) / 4
        self.bounds[2] = (0.5 * gp_bound, 1.5 * gp_bound)
        self.bounds[3] = (0.2 * coef * sigma_bound, 2.0 * coef * sigma_bound)
