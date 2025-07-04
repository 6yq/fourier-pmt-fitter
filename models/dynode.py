import numpy as np
from scipy.stats import norm
from core.base import PMT_Fitter


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
        seterr: str = "warn",
        init=[
            0.14,  # P(missing 1st dynode)
            2.8,  # P(multiplication missing)
            667,  # Normal mean
            40,  # Normal sigma
        ],
        bounds=[
            (0, 1),
            (1, None),
            (0, None),
            (0, None),
        ],
        constraints=None,
        auto_init=False,
    ):
        super().__init__(
            hist,
            bins,
            isWholeSpectrum,
            A,
            occ_init,
            sample,
            seterr,
            init,
            bounds,
            constraints,
            auto_init,
        )

    def _pdf_normal(self, x, df, mean, sigma):
        return (1 - df) * norm.pdf(x, loc=mean, scale=sigma)

    def _pdf_missing(self, x, df, ds, mean, sigma):
        mean_ = mean / ds
        sigma_ = sigma / ds
        return df * norm.pdf(x, loc=mean_, scale=sigma_)

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

    def _replace_spe_params(self, gp_init, sigma_init):
        self._init[2] = gp_init
        self._init[3] = sigma_init

    def _replace_spe_bounds(self, gp_bound, sigma_bound):
        self.bounds[2] = (0.5 * gp_bound, 1.5 * gp_bound)
        self.bounds[3] = (0.5 * sigma_bound, 1.5 * sigma_bound)
