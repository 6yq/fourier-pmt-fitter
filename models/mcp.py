import numpy as np
from scipy.stats import gamma
from ..core.base import PMT_Fitter
from ..core.utils import compute_init
from .tweedie_pdf import tweedie_reckon


class MCP_Fitter(PMT_Fitter):
    def __init__(
        self,
        hist,
        bins,
        isWholeSpectrum=False,
        A=None,
        occ_init=None,
        sample=None,
        seterr: str = "warn",
        init=[0.50, 400, 100, 3.0, 0.60, 0.15],
        bounds=[
            (0.0, 1.0),
            (0, None),
            (0, None),
            (1, 7),
            (0.3, 0.7),
            (0.05, 0.3),
        ],
        constraints=[
            {"coeffs": [(1, 1), (2, -1)], "threshold": 0, "op": ">"},
        ],
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

    def const(self, args):
        return (1 - args[0]) * np.exp(-args[3])

    def _map_args(self, args):
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
            * self._pdf_gm(self.xsp, frac, k, theta)
        )

    def Tws(self, args, occ):
        frac, _, _, mu, p, phi = self._map_args(args)
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_tw(self.xsp, frac, mu, p, phi)
        )

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

    def _pdf(self, args):
        frac, k, theta, mu, p, phi = self._map_args(args)
        return self._pdf_gm(self.xsp, frac, k, theta) + self._pdf_tw(
            self.xsp, frac, mu, p, phi
        )

    def _zero(self, args):
        frac, _, _, lam, _, _, occ = args
        mu = -np.log(1 - occ)
        return np.exp(mu * ((1 - frac) * np.exp(-lam) - 1))

    def _replace_spe_params(self, gp_init, sigma_init):
        self._init[1] = gp_init
        self._init[2] = 0.8 * sigma_init

    def _replace_spe_bounds(self, gp_bound, sigma_bound):
        self.bounds[1] = (0.5 * gp_bound, 1.5 * gp_bound)
        self.bounds[2] = (0.05 * sigma_bound, 3 * sigma_bound)
