import numpy as np

from ..core.base import PMT_Fitter
from ..core.utils import compute_init
from ..core.fft_utils import roll_and_pad


# ======================
#       Bi-Gauss
# ======================


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


# ======================
#      Linear Gauss
# ======================


class Linear_Gauss_Fitter(PMT_Fitter):
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

    def _pdf_normal(self, x, df, mean, sigma):
        inv = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        return (1 - df) * inv * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def _pdf_missing(self, x, df, ds, mean, sigma):
        mean_ = mean / ds
        sigma_ = sigma / ds
        inv = 1.0 / (np.sqrt(2.0 * np.pi) * sigma_)
        return df * inv * np.exp(-0.5 * ((x - mean_) / sigma_) ** 2)

    def _ser_pdf_time(self, args):
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


# ======================
#       Tri-Gauss
# ======================


class TriGauss_Fitter(PMT_Fitter):
    def __init__(
        self,
        hist,
        bins,
        isWholeSpectrum=False,
        A=None,
        occ_init=None,
        sample=None,
        init=[
            0.20,  # w1
            0.65,  # w2  (w3 = 1 - w1 - w2)
            300,  # m1
            200,  # d12 > 0  => m2 = m1 + d12
            300,  # d23 > 0  => m3 = m2 + d23
            30,  # s1 > 0
            100,  # s2 > 0
            250,  # s3 > 0
        ],
        bounds=[
            (0, 1),  # w1
            (0, 1),  # w2
            (0, None),  # m1
            (0, None),  # d12
            (0, None),  # d23
            (0, None),  # s1
            (0, None),  # s2
            (0, None),  # s3
        ],
        constraints=[
            {"coeffs": [(0, 1), (1, 1)], "threshold": 1, "op": "<"},
        ],
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

    def _gauss(self, x, mean, sigma):
        inv = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
        z = (x - mean) / sigma
        return inv * np.exp(-0.5 * z * z)

    def _means_from_args(self, m1, d12, d23):
        m2 = m1 + d12
        m3 = m2 + d23
        return m1, m2, m3

    def _weights_from_args(self, w1, w2):
        return w1, w2, 1.0 - w1 - w2

    def _ser_pdf_time(self, args):
        w1, w2, m1, d12, d23, s1, s2, s3 = args
        m1, m2, m3 = self._means_from_args(m1, d12, d23)
        w1, w2, w3 = self._weights_from_args(w1, w2)

        return (
            w1 * self._gauss(self.xsp, m1, s1)
            + w2 * self._gauss(self.xsp, m2, s2)
            + w3 * self._gauss(self.xsp, m3, s3)
        )

    def G1(self, args, occ):
        w1, w2, m1, d12, d23, s1, _, _ = args
        m1, _, _ = self._means_from_args(m1, d12, d23)
        w1, _, w3 = self._weights_from_args(w1, w2)
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * w1
            * self._gauss(self.xsp, m1, s1)
        )

    def G2(self, args, occ):
        w1, w2, m1, d12, d23, _, s2, _ = args
        _, m2, _ = self._means_from_args(m1, d12, d23)
        _, w2, _ = self._weights_from_args(w1, w2)
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * w2
            * self._gauss(self.xsp, m2, s2)
        )

    def G3(self, args, occ):
        w1, w2, m1, d12, d23, _, _, s3 = args
        _, _, m3 = self._means_from_args(m1, d12, d23)
        _, _, w3 = self._weights_from_args(w1, w2)
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * w3
            * self._gauss(self.xsp, m3, s3)
        )

    def get_gain(self, args, gain: str = "gm"):
        w1, w2, m1, d12, d23, *_ = args
        m1, m2, m3 = self._means_from_args(m1, d12, d23)
        w1, w2, w3 = self._weights_from_args(w1, w2)

        if gain == "gm":
            return w1 * m1 + w2 * m2 + w3 * m3
        elif gain == "gp":
            idx = int(np.argmax([w1, w2, w3]))
            return [m1, m2, m3][idx]
        else:
            raise NameError(f"{gain} is not a legal gain type")


# ======================
#     Gauss Compound
# ======================


class Gauss_Compound_Fitter(PMT_Fitter):
    def __init__(
        self,
        hist,
        bins,
        isWholeSpectrum=False,
        A=None,
        occ_init=None,
        sample=None,
        init=[0.60, 0.6, 0.25, 5.0, 0.6, 0.2],
        bounds=[
            (0.3, 1.0),
            (0, None),
            (0, None),
            (1, None),
            (0, 1),
            (0, 1),
        ],
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

    def const(self, ser_args):
        frac, mu, sigma, lam, mu_ts, sigma_ts = ser_args
        return (1 - frac) * np.exp(-lam)

    def _map_args(self, args):
        frac, mu, sigma, lam, mu_ts, sigma_ts = args
        mu_prime = mu * mu_ts
        sigma_prime = sigma * sigma_ts
        return (frac, mu, sigma, lam, mu_prime, sigma_prime)

    def _ft_gauss(self, freq, mu, sigma):
        return np.exp(-1j * mu * freq) * np.exp(-((sigma * freq) ** 2) / 2)

    def _ser_ft(self, freq, ser_args):
        frac, mu, sigma, lam, mu_prime, sigma_prime = self._map_args(ser_args)
        ft_g = self._ft_gauss(freq, mu, sigma)
        ft_ts0 = self._ft_gauss(freq, mu_prime, sigma_prime)
        ft_tw = np.exp(-lam) * (np.exp(lam * ft_ts0) - 1)
        return frac * ft_g + (1 - frac) * ft_tw

    def Gms(self, args, occ):
        frac, mu, sigma = args[:3]
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * frac
            * np.exp(-0.5 * ((self.xsp - mu) / sigma) ** 2)
            / np.sqrt(2 * np.pi * sigma**2)
        )

    def Comps(self, args, occ):
        frac, mu, sigma, lam, mu_prime, sigma_prime = self._map_args(args)
        return (
            self.A
            * self._bin_width
            * self._pdf_comps(frac, lam, mu_prime, sigma_prime, occ)
        )

    def _pdf_comps(self, frac, lam, mu_prime, sigma_prime, occ):
        const = np.exp(-lam)
        ft_ts0 = self._ft_gauss(self._freq, mu_prime, sigma_prime)
        ft_tw_nz = np.exp(-lam) * (np.exp(lam * ft_ts0) - 1) / (1 - np.exp(-lam))
        fft_input = (1 - const) * ft_tw_nz + const
        s_sp = self._nPE_processor(occ, 1)(fft_input)
        ifft_pdf = self._ifft_pipeline(s_sp)
        return (1 - frac) * ifft_pdf

    def get_gain(self, args, gain: str = "gm"):
        if gain == "gp":
            frac, mu, sigma, lam, mu_ts, sigma_ts = args
            return mu
        elif gain == "gm":
            frac, mu, sigma, lam, mu_ts, sigma_ts = args
            fracReNorm = frac / (1 - (1 - frac) * np.exp(-lam))
            return fracReNorm * mu + (1 - fracReNorm) * mu * mu_ts * lam
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
