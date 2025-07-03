import numpy as np
from scipy.stats import gamma
from core.base import PMT_Fitter


class Dynode_Fitter(PMT_Fitter):
    def __init__(
        self,
        hist,
        bins,
        isWholeSpectrum=False,
        A=None,
        occ_init=None,
        sample=None,
        seterr: str = "warn",
        init=[600, 40],
        bounds=[(0, None), (0, None)],
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

    def get_gain(self, args, gain: str = "gm"):
        if gain == "gp" or gain == "gm":
            mean, sigma = args
            k = (mean / sigma) ** 2
            theta = mean / k
            return (k - 1) * theta
        else:
            raise NameError(f"{gain} is not a legal gain type")

    def _pdf(self, args):
        mean, sigma = args
        k = (mean / sigma) ** 2
        theta = mean / k
        return gamma.pdf(self.xsp, a=k, scale=theta)

    def _replace_spe_params(self, gp_init, sigma_init):
        self._init[0] = gp_init
        self._init[1] = sigma_init

    def _replace_spe_bounds(self, gp_bound, sigma_bound):
        self.bounds[0] = (0.5 * gp_bound, 1.5 * gp_bound)
        self.bounds[1] = (0.5 * sigma_bound, 1.5 * sigma_bound)
