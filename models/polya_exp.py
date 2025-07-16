import numpy as np
from scipy.stats import gamma, expon
from ..core.base import PMT_Fitter


class Polya_Exp_Fitter(PMT_Fitter):
    """
    Polya Exponential model.

    Notes
    -----
    The polya part is actully a reparameterized Gamma distribution.

    `lambda` equals `1 / gain` and `theta` is a shaping parameter.
    If we are describing it with the traditional Gamma parameters (`alpha` for the shape and `beta` for the scale),
    then `lambda = 1 / (alpha * beta)` and `theta = 1 - alpha`.

    Here we want to parameterize the Gamma with mean and standard deviation,
    because they are also orthometric, and easier to give better initial guess.
    Equally, `mean = 1 / lambda` and `sigma = 1 / (lambda * sqrt(1 + theta))`.

    - Introduced by Marcos Dracos;
    - See JUNO-doc-14081.
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
            0.4,  # fraction of exponential
            0.8,  # mean of gamma
            0.2,  # sigma of gamma
            1.6,  # scale or 1 / lambda of exponential
        ],
        bounds=[
            (0, 1),
            (0, None),
            (0, None),
            (0, None),
        ],
        constraints=[
            {"coeffs": [(1, 1), (2, -1)], "threshold": 0, "op": ">"},
        ],
        threshold=None,
        auto_init=False,
        seterr: str = "warn",
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
        )

    def _pdf_polya(self, x, frac, mean, sigma):
        k = (mean / sigma) ** 2
        theta = mean / k
        return (1 - frac) * gamma.pdf(x, a=k, scale=theta)

    def _pdf_exp(self, x, frac, scale):
        return frac * expon.pdf(x, scale=scale)

    def _pdf(self, args):
        frac, mean, sigma, scale = args
        return self._pdf_polya(self.xsp, frac, mean, sigma) + self._pdf_exp(
            self.xsp, frac, scale
        )

    def Polya(self, args, occ):
        frac, mean, sigma, _ = args
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_polya(self.xsp, frac, mean, sigma)
        )

    def Exponential(self, args, occ):
        frac, _, _, scale = args
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_exp(self.xsp, frac, scale)
        )

    def get_gain(self, args, gain: str = "gm"):
        if gain == "gp":
            _, mean, sigma, _ = args
            return mean
        elif gain == "gm":
            frac, mean, _, scale = args
            return frac * scale + (1 - frac) * mean
        else:
            raise NameError(f"{gain} is not a legal gain type")

    def _replace_spe_params(self, gp_init, sigma_init):
        self._init[2] = gp_init
        self._init[3] = sigma_init

    def _replace_spe_bounds(self, gp_bound, sigma_bound):
        self.bounds[2] = (0.5 * gp_bound, 1.5 * gp_bound)
        self.bounds[3] = (0.5 * sigma_bound, 1.5 * sigma_bound)
