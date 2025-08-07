import numpy as np
from scipy.stats import gamma
from scipy.fft import fft, ifft

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

    def const(self, args):
        return (1 - args[0]) * np.exp(-args[3])

    def _map_args(self, args):
        frac, mean, sigma, lam, mean_t, sigma_t = args
        k = (mean / sigma) ** 2
        theta = mean / k
        k_ts = (mean_t / sigma_t) ** 2
        theta_ts = mean * (sigma_t**2) / mean_t
        return (frac, k, theta, lam, k_ts, theta_ts)

    def _pdf_gm(self, frac, k, theta, occ):
        n_full = len(self.xsp) + self._pad_safe
        # omega_j
        freq = 2 * np.pi * np.fft.fftfreq(n_full, d=self._xsp_width)
        fft_pdf = self._ft_gamma(freq, k, theta)
        fft_processed = self._nPE_processor(occ, 1)(fft_pdf)
        shift_padded = 2 * self._shift if self._shift < 0 else 0
        ifft_pdf = np.roll(
            np.real(ifft(fft_processed)) / self._xsp_width, -shift_padded
        )[: len(self.xsp)]
        return frac * ifft_pdf

    def _pdf_tw(self, frac, lam, k_ts, theta_ts, occ, const):
        n_full = len(self.xsp) + self._pad_safe
        # omega_j
        freq = 2 * np.pi * np.fft.fftfreq(n_full, d=self._xsp_width)
        ft_gamma_ts = self._ft_gamma(freq, k_ts, theta_ts)
        ft_tweedie_nz = np.exp(-lam) * (np.exp(lam * ft_gamma_ts) - 1)

        denom = 1 - np.exp(-lam)
        ft_cont = 1 / denom * ft_tweedie_nz

        fft_pdf = (1 - const) * ft_cont + const
        fft_processed = self._nPE_processor(occ, 1)(fft_pdf)
        shift_padded = 2 * self._shift if self._shift < 0 else 0
        ifft_pdf = np.roll(
            np.real(ifft(fft_processed)) / self._xsp_width, -shift_padded
        )[: len(self.xsp)]
        return (1 - frac) * ifft_pdf

    def _ft_gamma(self, freq, k, theta):
        """Wow, Gamma FFT is analytic!

        Parameters
        ----------
        freq : ndarray
        k : float
        theta : float
        """
        return (1 + 1j * theta * freq) ** (-k)

    def _pdf_sr(self, args):
        frac, k, theta, lam, k_ts, theta_ts = self._map_args(args[self._start_idx : -1])

        const = self.const(args[self._start_idx : -1])
        n_full = len(self.xsp) + self._pad_safe
        # omega_j
        freq = 2 * np.pi * np.fft.fftfreq(n_full, d=self._xsp_width)
        ft_gamma_g = self._ft_gamma(freq, k, theta)
        ft_gamma_ts = self._ft_gamma(freq, k_ts, theta_ts)
        ft_tweedie_nz = np.exp(-lam) * (np.exp(lam * ft_gamma_ts) - 1)

        denom = 1 - np.exp(-lam)
        ft_cont = frac * ft_gamma_g + (1 - frac) / denom * ft_tweedie_nz
        fft_pdf = (1 - const) * ft_cont + const
        shift_padded = 2 * self._shift if self._shift < 0 else 0

        b_sp = self._b_sp(args)
        pass_threshold = self._efficiency(self.xsp, *args[: self._start_idx])
        # fft_processed = self._all_PE_processor(args[-1], b_sp)(fft_pdf)
        fft_processed = self._nPE_processor(args[-1], 2)(fft_pdf)
        ifft_pdf = np.roll(
            np.real(ifft(fft_processed)) / self._xsp_width, -shift_padded
        )[: len(self.xsp)]
        return pass_threshold * ifft_pdf

    def _pdf_sr_n(self, args, n):
        if n == 0:
            return self._pdf_ped(args[: self._start_idx])
        else:
            frac, k, theta, lam, k_ts, theta_ts = self._map_args(
                args[self._start_idx : -1]
            )

            const = self.const(args[self._start_idx : -1])
            n_full = len(self.xsp) + self._pad_safe
            # omega_j
            freq = 2 * np.pi * np.fft.fftfreq(n_full, d=self._xsp_width)
            ft_gamma_g = self._ft_gamma(freq, k, theta)
            ft_gamma_ts = self._ft_gamma(freq, k_ts, theta_ts)
            ft_tweedie_nz = np.exp(-lam) * (np.exp(lam * ft_gamma_ts) - 1)

            denom = 1 - np.exp(-lam)
            ft_cont = frac * ft_gamma_g + (1 - frac) / denom * ft_tweedie_nz
            fft_pdf = (1 - const) * ft_cont + const
            fft_processed = self._nPE_processor(args[-1], n)(fft_pdf)
            shift_padded = 2 * self._shift if self._shift < 0 else 0
            ifft_pdf = np.roll(
                np.real(ifft(fft_processed)) / self._xsp_width, -shift_padded
            )[: len(self.xsp)]
            return ifft_pdf

    def _produce_pdf_sr_n(self):
        return self._pdf_sr_n

    def Gms(self, args, occ):
        frac, mean, sigma = args[:3]
        k = (mean / sigma) ** 2
        theta = mean / k
        mu_l = -np.log(1 - occ)
        return self.A * self._bin_width * self._pdf_gm(frac, k, theta, occ)

    def Tws(self, args, occ):
        frac, _, _, lam, k_ts, theta_ts = self._map_args(args)
        mu_l = -np.log(1 - occ)
        return (
            self.A
            * mu_l
            * np.exp(-mu_l)
            * self._bin_width
            * self._pdf_tw(frac, lam, k_ts, theta_ts, occ, self.const(args))
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

    # def _pdf(self, args):
    #     frac, k, theta, mu, p, phi = self._map_args(args)
    #     return self._pdf_gm(self.xsp, frac, k, theta) + self._pdf_tw(
    #         self.xsp, frac, mu, p, phi
    #     )

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
