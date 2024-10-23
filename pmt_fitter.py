import time
import numpy as np

from typing import Callable
from tweedie import tweedie
from scipy.fft import fft, ifft
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import gamma, expon, norm
from abc import ABCMeta


class PMT_Fitter(metaclass=ABCMeta):
    def __init__(
        self, 
        charge, 
        A, 
        occ_init, 
        sample = None, 
        cut:float | tuple[float] = (0.2, 5), 
        seterr:str = 'warn', 
        init = None, 
        bounds = None
    ):
        np.seterr(all=seterr)
        
        self.A = A
        c = charge if isinstance(charge, np.ndarray) else np.array(charge)
        self.bounds = bounds.tolist() if isinstance(bounds, np.ndarray) else list(bounds)
        self._init = init if isinstance(init, np.ndarray) else np.array(init)
        self._occ_init = occ_init if isinstance(occ_init, np.ndarray) else np.array(occ_init)
        # see https://gitlab.airelinux.org/juno/OSIRIS/production/-/issues/48#note_42922
        self.sample = 16 * int(np.exp(-0.673313 * np.log(1 - np.max(self._occ_init)))) if sample is None else sample

        self.init = np.append(self._init, self._occ_init)
        self.bounds.append((0, 1))
        self.bounds = tuple(self.bounds)
                
        if isinstance(cut, tuple):
            lower, upper = cut
            if lower is None:
                lower = 0
            if upper is None:
                upper = 1000
        else:
            # default lower cut
            lower = 0.2     
            upper = cut
            
        hist, bins = np.histogram(c, bins='fd')
        peaks, _ = find_peaks(hist, distance=np.ptp(c))
        c = c[c >= lower * bins[peaks[0]]]
        c = c[c <= upper * bins[peaks[0]]]
                
        self.hist, self.bins = np.histogram(c, bins='fd')
        self.zero = A - sum(self.hist)
        # there is no need to check these variables outside the class
        self._bin_width = self.bins[1] - self.bins[0]
        self._xs = (self.bins[:-1] + self.bins[1:]) / 2
        self._interval = (self.bins[1] - self.bins[0]) / self.sample
        self._bwds = 1 - int(-self.bins[0] // self._interval)
        self.xsp = np.linspace(self.bins[0] - self._bwds * self._interval, 
                               self.bins[-1], 
                               num=len(self.hist) * self.sample + self._bwds + 1, 
                               endpoint=True)
        self._xsp_width = self.xsp[1] - self.xsp[0]

        self.dof = len(init)
        self.ndf = len(self.hist) - self.dof
        self.C = self._log_l_C()
        self._start = time.time()
        
    # ------------
    # staticmethod
    # ------------

    def composite_simpson(self, pdf_slice, interval, sample):
        """ Use composite Simpson to integrate pdf.
                
        Parameters
        ----------
        pdf_slice : ArrayLike
            a pdf list/array/...
        """
        result = pdf_slice[0] + pdf_slice[-1]
        odd_sum = sum(pdf_slice[i] for i in range(1, sample, 2))
        even_sum = sum(pdf_slice[i] for i in range(2, sample, 2))
        result += 4 * odd_sum + 2 * even_sum
        result *= interval / 3
        return result
    
    def isInBound(self, param:float | int, bound:tuple[None | float | int]) -> bool:
        assert len(bound) == 2, "Illegal bound!"
        if None not in bound:
            lower, upper = bound
            assert lower <= upper, "Illegal order of bound!"
            return (param >= lower) & (param <= upper)
        elif bound == (None, None):
            return True
        elif bound[0] is None:
            return (param < bound[1])
        else:
            return (param > bound[0])
        
    def isParamsInBound(self, params, bounds):
        flag = True
        for p, b in zip(params, bounds):
            flag = flag & self.isInBound(param=p, bound=b)
            if flag == False:
                return False
        return flag

    def get_gain(self, args):
        pass
    
    # --------
    # property
    # --------
    
    def _log_l_C(self):
        """ Return constant in log likelihood.
                
        Notes
        -----
        C = NlnN - ln(N!) + sum_j(ln(nj!))
        """
        N = sum(self.hist) + self.zero
        N_part = N * np.log(N) - sum(np.log(np.arange(1, N + 1)))
        n_part = sum([sum(np.log(np.arange(1, n + 1))) 
                      for n in self.hist]) + sum(np.log(np.arange(1, self.zero + 1)))
        return N_part + n_part
    
    # -----------
    # classmethod
    # -----------
    
    def _pdf(self, args):
        pass
    
    def _zero(self, args):
        pass

    def _const(self, args):
        return 0
    
    def _pdf_sr(self, args):
        """ Applying DFT & IDFT to estimate pdf.
                
        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occupancy)
        """
        ser_args = args[:self.dof]
        occupancy = args[self.dof]
        s_sp = fft(self._pdf(ser_args)) * self._xsp_width + self._const(ser_args)
        mu = -np.log(1 - occupancy)
        sr_sp = np.exp(mu * (s_sp - 1))
        sr_sp = np.real(ifft(sr_sp)) / self._xsp_width
        return sr_sp
    
    def _estimate_smooth(self, args):
        return self.A * self._pdf_sr(args=args) * self._bin_width
    
    def _estimate_count(self, args) -> tuple:
        """ Estimate counts of every bin.
                
        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occupancy)
            
        Return
        ------
        y_est : ArrayLike
            (entry_est_in_bin_1, ..., entry_est_in_bin_n)
        z_est : float
            Expected zero entries.
        """
        y_sp = self.A * self._pdf_sr(args=args)
        y_sp_slice = np.array([
            y_sp[(self._bwds + self.sample * i):(self._bwds + self.sample * (i + 1) + 1)] 
            for i in range(len(self.hist))])
        y_est = np.apply_along_axis(self.composite_simpson, 
                                    1, 
                                    y_sp_slice, 
                                    self._interval,
                                    self.sample)
        # nonegative pdf set
        y_est[y_est <= 0] = 1e-16
        z_est = self.A * self._zero(args)
        return y_est, z_est
    
    def log_l(self, args) -> float:
        """ log likelihood of given args.

        Parameters
        ----------
        args : ArrayLike
            (ser_args_1, ..., ser_args_(dof), occupancy)
        """
        # make sure args are in range
        # otherwise an infinite "well"
        if self.isParamsInBound(args, self.bounds):
            y, z = self._estimate_count(args)
            return self.zero * np.log(z) + np.sum(self.hist * np.log(y)) - self.C
        else:
            return -np.inf
        
    def get_chi_sq(self, args) -> float:
        """ Chi square.

        Parameters
        ----------
        ser_args : ArrayLike
        occs : ArrayLike
        """        
        y, z = self._estimate_count(args)
        return sum((y - self.hist) ** 2 / y) + (z - self.zero) ** 2 / z
    
    def fit(
        self, 
        burn_in:int=2500,
        mc_step:int=5000,
        seed:int=None,
        verbose:int=10,
        track:int=0, 
        step_length:dict[str, float]=None
    ):
        """ MCMC fit.
        
        Parameters
        ----------
        mc_step : int
            MCMC step.
        verbose : int
            If equals 0, no intermediate info would be printed.
            Otherwise, event verbose steps, info would be printed.
        track : int
            If equals 0, no intermediate value would be recorded.
            Otherwise, event track steps, log-l and parameters would be recorded.
        seed : int
            Seed for random.
        """
        steps_mc = np.array(list(step_length.values()))
        args = self.init
        args_mc = np.zeros((mc_step, self.dof + 1))
        
        if seed is not None:
            np.random.seed(seed)
        u_perturb = np.random.uniform(-1, 1, (mc_step, self.dof + 1))
        u_accept = np.random.uniform(0, 1, (mc_step, self.dof + 1))
        perturbation = steps_mc * u_perturb
        acceptance_log = np.log(u_accept)
        del u_perturb, u_accept
        
        if track != 0:
            self.track = track
            self.log_l_track = []
            self.ser_args_track = []  
        
        #--------------
        #-- sampling --
        #--------------
        for step in range(mc_step):
            likelihood = self.log_l(args)
            for i, arg in enumerate(args):
                args_new = np.copy(args)
                args_new[i] += perturbation[step, i]
                likelihood_new = self.log_l(args_new)
                index = likelihood_new - likelihood > acceptance_log[step, i]
                args = [args, args_new][index]
                likelihood = [likelihood, likelihood_new][index]
            args_mc[step, :] = args
            
            # print verbose
            if verbose != 0 and step % verbose == 0:
                print("----------")
                print(f'Step {step}, log likelihood: {likelihood}')
                print("Params: " + ', '.join([f'{e:.6g}' for e in args]))
                print(time.strftime("%H:%M:%S",time.gmtime(time.time() - self._start)))
                self._start = time.time()
                
            # record track
            if track != 0 and step % track == 0:
                self.log_l_track.append(likelihood)
                self.ser_args_track.append(args)
        
        # burn and print final result
        args_mc = args_mc[burn_in:, :]
        args = np.mean(args_mc, axis=0)
        args_std = np.std(args_mc, axis=0)
        self.ser_args = args[:self.dof]
        self.ser_args_std = args_std[:self.dof]
        self.occupancy = args[self.dof]
        self.occupancy_std = args_std[self.dof]
        self.likelihood = self.log_l(args)
        self.Gps = np.apply_along_axis(self.get_gain, axis=1, arr=args_mc, gain="gp")
        self.Gms = np.apply_along_axis(self.get_gain, axis=1, arr=args_mc, gain="gm")
        del args_mc
        
        print("----------")
        print(f'Reach {mc_step} steps.')
        print(f'log likelihood: {likelihood}')
        print("SER params: " + ', '.join([f'{e:.3g}$\pm${f:.3g}' for e, f in zip(self.ser_args, self.ser_args_std)]))
        print(f'Occupancy: {self.occupancy:.3g}$\pm${self.occupancy_std:.3g}')

        self.gains_mean = np.mean(self.Gms)
        self.gains_std = np.std(self.Gms)
        self.BIC = (self.dof + 1) * np.log(len(self.hist) + 1) - 2 * self.likelihood
        self.chi_sq = self.get_chi_sq(args)
        self.smooth = self._estimate_smooth(args)
        self.ys, self.zs = self._estimate_count(args)


class MCP_Fitter(PMT_Fitter):
    """ A class to fit MCP-PMT charge spectrum.
    
    Parameters
    ----------
    charge : ArrayLike
        charge dataset input
    A : ArrayLike
        total charge count
    occ_init : ArrayLike
        initial occupancy
    sample : int
        the number of sample intervals between bins
    cut : float or tuple[float]
        the upper cut (lower cut optional), expressed with the ratio divided by main peak
    init : ArrayLike
        initial params of SER charge model, in the order of
        "main peak ratio, main peak k/shape, main peak theta/rate, 
        secondary electron number, secondary alpha/shape, secondary beta/rate"
    seterr : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        see https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
    """
    def __init__(
        self, 
        charge, 
        A, 
        occ_init, 
        sample = None, 
        cut: float | tuple[float] = (0.2, 5), 
        seterr: str = 'warn', 
        init =
        [
            .40,            # main peak ratio
            15.00,          # main peak k/shape
            50.00,          # main peak theta/scale
            2.40,           # secondary electron number
            0.52,           # secondary electron mean / Q1
            0.16            # secondary electron std variance / Q1
        ], 
        bounds = 
        [
            (0, 1),         # ratio in (0, 1)
            (1, None),      # alpha > 1 to ensure peak
            (0, None),      # beta > 0
            (0, None),      # secondary > 0
            (0.3, 0.7),     # mean / Q1 based on Jun's work
            (0.05, 0.3)     # std variance / Q1 based on Jun's work
        ]
    ):
        super().__init__(charge, A, occ_init, sample, cut, seterr, init, bounds)
    
    # ------------
    # staticmethod
    # ------------
    
    def get_gain(self, args, gain:str="gm"):
        """ Return gain of MCP.
        
        Parameters
        ----------
        ser_args : ArrayLike.
            Elements:
                frac : float
                    ratio of peak
                shape : float
                    main peak, shape in Gamma distribution
                scale : float
                    main peak, scale in Gamma distribution
                lam : float
                    mean of secondary electron numbers
                mean : float
                    mean of secondary Gamma gain / Q1
                std_variance : float
                    std variance of secondary Gamma gain / Q1
                    
        Return
        ------
        gain : float
            Gain of MCP.
            
        Notes
        -----
        Return mean of SPE distribution (Gm) by default.
        """
        if gain == "gp":
            return args[1] * args[2]
        elif gain == "gm":
            return (args[0] * args[1] * args[2] + 
                    (1 - args[0]) * args[1] * args[2] * args[3] * args[4])
        else:
            raise NameError(f'{gain} is not a illegal parameter!')
    
    def _map_args(self, args) -> tuple:
        """ Map MCP-PMT SER/SPE charge model parameters.
        
        Parameters
        ----------
        args : ArrayLike.
            Elements:
                frac : float
                    ratio of peak
                shape : float
                    main peak, shape in Gamma distribution
                scale : float
                    main peak, scale in Gamma distribution
                lam : float
                    mean of secondary electron numbers
                mean : float
                    calibrated on 8-inches, the mean of secondary Gamma gain / Q1
                std_variance : float
                    calibrated on 8-inches, the std variance of secondary Gamma gain / Q1
                    
        Return
        ------
        result : tuple
            Elements:
                frac : float
                    ratio of peak
                shape : float
                    main peak, shape in Gamma distribution
                scale : float
                    main peak, scale in Gamma distribution
                mu : float
                    mean of tweedie (secondary electron charge)
                p : float
                    power of tweedie (1 < p < 2)
                phi : float
                    variance param of tweedie (var = phi * mu^p)
            
        Notes
        -----
        True secondary (ts) `q/q1 ~ GA(alpha_ts, beta_ts)` -> `q ~ GA(alpha_ts, beta_ts / q1)`;
        True secondary parameters mapping see: 
        https://en.wikipedia.org/wiki/Compound_Poisson_distribution
        """
        frac, k, theta, lam, mean, std_variance = args
        alpha_ts = (mean ** 2) / (std_variance ** 2)
        beta_ts = mean / (std_variance ** 2)
        Q1 = k * theta
        mu = lam * alpha_ts * Q1 / beta_ts
        p = 1 + 1 / (alpha_ts + 1)
        phi = (alpha_ts + 1) * pow(lam * alpha_ts, 1 - p) / pow(beta_ts / Q1, 2 - p)
        return (frac, k, theta, mu, p, phi)  
        
    def _pdf_gm(self, x, frac, k, theta):
        return frac * gamma.pdf(x, a=k, scale=theta)
        
    def _pdf_tw(self, x, frac, mu, p, phi):
        pdf = (1 - frac) * tweedie.pdf(x, mu=mu, p=p, phi=phi)
        if 0 in x:
            ind = np.where(x == 0)[0][0]
            pdf[ind] = 0
        return pdf
    
    # -----------
    # classmethod
    # -----------

    def _const(self, args):
        return (1 - args[0]) * np.exp(-args[3])
    
    def _pdf(self, args):
        frac, k, theta, mu, p, phi = self._map_args(args)
        return self._pdf_gm(self.xsp, frac, k, theta) + self._pdf_tw(self.xsp, frac, mu, p, phi)
    
    def _zero(self, args):
        """ MCP-PMT SER/SPE charge model zero charge count.
                
        Parameters
        ----------
        args : ArrayLike
            (frac, k, theta, lam, mean, std_variance, occupancy)
        """
        frac, _, _, lam, _, _ = args[:self.dof]
        occupancy = args[-1]
        mu = -np.log(1 - occupancy)
        return np.exp(mu * ((1 - frac) * np.exp(-lam) - 1))

    # ----------
    # components
    # ----------
    
    def gms(self, args):
        frac, k, theta = args[:3]
        return self.A * self._pdf_gm(self.xsp, frac=frac, k=k, theta=theta) * self._xsp_width
    
    def tws(self, args):
        frac, _, _, mu, p, phi = self._map_args(args)
        return self.A * self._pdf_tw(self.xsp, frac=frac, mu=mu, p=p, phi=phi) * self._xsp_width

    # --------------
    # default length
    # --------------
    step_length = {'frac'      : 0.05,
                   'k'         : 1.0,
                   'theta'     : 3.0,
                   'lambda'    : 0.15,
                   'mean'      : 0.05,
                   'std_var'   : 0.03,
                   'occupancy' : 0.05}

    def fit(
        self, 
        burn_in:int=2500,
        mc_step:int=5000,
        seed:int=None,
        verbose:int=10,
        track:int=0, 
        step_length:dict[str, float]=step_length
    ):
        """ MCMC fit.
        
        Additional Parameters
        ---------------------
        step_length : dict
            keys in the order of ('frac', 'k', 'theta', 'lambda', 'mean', 'std_var', 'occupancy'),
            values in float.
        """
        super().fit(burn_in, mc_step, seed, verbose, track, step_length)
        
        
class Dynode_Fitter(PMT_Fitter):
    """ A class to fit Dynode PMT charge spectrum.
    
    Parameters
    ----------
    charge : ArrayLike
        charge dataset input
    A : ArrayLike
        total charge count
    occ_init : ArrayLike
        initial occupancy
    sample : int
        the number of sample intervals between bins
    cut : float or tuple[float]
        the upper cut (lower cut optional), expressed with the ratio divided by main peak
    init : ArrayLike
        initial params of SER charge model, in the order of
        "peak shape, peak rate"
    bounds : ArrayLike
        initial bounds of SER charge model
    seterr : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        see https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
    """
    def __init__(
        self, 
        charge, 
        A, 
        occ_init, 
        sample = None, 
        cut: float | tuple[float] = (0.2, 5), 
        seterr: str = 'warn', 
        init =
        [
            20,        # peak k/shape
            40,        # peak theta/scale
        ], 
        bounds =
        [
            (1, None),  # k > 1 to ensure peak
            (0, None),  # theta > 0
        ]
    ):
        super().__init__(charge, A, occ_init, sample, cut, seterr, init, bounds)
    
    # ------------
    # staticmethod
    # ------------
    
    def get_gain(self, args, gain:str="gm"):
        """ Return gain of MCP.
        
        Parameters
        ----------
        ser_args : ArrayLike.
            Elements:
                shape : float
                    peak shape in Gamma distribution
                scale : float
                    peak scale in Gamma distribution
        
        Return
        ------
        gain : float
            Gain of dynode PMT.
            
        Notes
        -----
        Return mean of SPE distribution (Gm) by default.
        """
        if gain == "gp":
            return (args[0] - 1) * args[1]
        elif gain == "gm":
            return args[0] * args[1]
        else:
            raise NameError(f'{gain} is not a illegal parameter!')
    
    # -----------
    # classmethod
    # -----------
    
    def _pdf(self, args):
        p, G, sigma = args
        return gamma.pdf(self.xsp, a=args[0], scale=args[1])
    
    def _zero(self, args):
        """ MCP-PMT SER/SPE charge model zero charge count.
                
        Parameters
        ----------
        args : ArrayLike
            (frac, k, theta, lam, mean, std_variance, occupancy)
        """
        return 1 - args[-1]


class Simulation_Fitter(PMT_Fitter):
    """ A temporary class to fit MCP PMT charge spectrum with Exponential-Gaussian model.
    
    Parameters
    ----------
    charge : ArrayLike
        charge dataset input
    A : ArrayLike
        total charge count
    occ_init : ArrayLike
        initial occupancy
    sample : int
        the number of sample intervals between bins
    cut : float or tuple[float]
        the upper cut (lower cut optional), expressed with the ratio divided by main peak
    init : ArrayLike
        initial params of SER charge model, in the order of
        "exponential fraction, main peak mu, main peak sigma"
    bounds : ArrayLike
        initial bounds of SER charge model, secondary within 3 sigma
    seterr : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}
        see https://numpy.org/doc/stable/reference/generated/numpy.seterr.html
    """
    def __init__(
        self, 
        charge, 
        A, 
        occ_init, 
        sample = None, 
        cut: float | tuple[float] = (0.2, 5), 
        seterr: str = 'warn', 
        init =
        [
            0.0433,
            700,
            200,
        ], 
        bounds =
        [
            (0.035, 0.185),
            (600, 900),
            (0, 400),
        ],
        threshold: float = 0.5,
        limit: tuple = (0.8, 1.2),
    ):
        if isinstance(init, float):
            init = [init]
        super().__init__(charge, A, occ_init, sample, cut, seterr, init, bounds)

        # override init and bounds with main peak Gaussian fit
        lower, upper = limit
        ind_roi = np.where(self.hist >= max(self.hist) * threshold)[0]
        xs_roi = self._xs[ind_roi]
        ys_roi = self.hist[ind_roi]
        bounds = list(self.bounds)

        def func(x, a, b, c):
            return a * norm.pdf(x, loc=b, scale=c)

        popt, _ = curve_fit(func, xs_roi, ys_roi, p0=[self.A * self._occ_init * self._bin_width, *init[1:]])

        if len(init) == 1:
            self.init = np.concatenate((init, popt[1:], [self._occ_init]))
            self.dof = 3
            for i in range(1, self.dof):
                bounds.append((popt[i] * lower, popt[i] * upper))
            self.bounds = tuple(bounds)
        elif len(init) == 3:
            self.init[1:3] = popt[1:]
            for i in range(1, self.dof):
                bounds[i] = (popt[i] * 0.8, popt[i] * 1.2)
            self.bounds = tuple(bounds)
    
    # ------------
    # staticmethod
    # ------------
    
    def get_gain(self, args, gain:str="gm"):
        """ Return gain of MCP.
        
        Parameters
        ----------
        ser_args : ArrayLike.
            Elements:
                p : float
                    propotion of exponential
                G : float
                    gain
                sigma : float
                    sigma of gain
        
        Return
        ------
        gain : float
            Gain of dynode PMT.
            
        Notes
        -----
        Return mean of SPE distribution (Gm) by default.
        """
        if gain == "gp":
            return args[1]
        elif gain == "gm":
            return args[0] * args[1] + (1 - args[0]) * args[1] * 2.3
        else:
            raise NameError(f'{gain} is not a illegal parameter!')
    
    # -----------
    # classmethod
    # -----------
    
    def _pdf(self, args):
        p, G, sigma = args
        return p * expon.pdf(self.xsp, loc=0.1 * G, scale=G * 2.2) + (1 - p) * norm.pdf(self.xsp, loc=G, scale=sigma)
    
    def _zero(self, args):
        """ MCP-PMT SER/SPE charge model zero charge count.
                
        Parameters
        ----------
        args : ArrayLike
            (frac, k, theta, lam, mean, std_variance, occupancy)
        """
        return 1 - args[-1]

    def exps(self, args):
        p, G, sigma = args
        return self.A * p * expon.pdf(self.xsp, loc=0.1 * G, scale=G * 2.2) * self._xsp_width
    
    def norms(self, args):
        p, G, sigma = args
        return self.A * (1 - p) * norm.pdf(self.xsp, loc=G, scale=sigma) * self._xsp_width

    # --------------
    # default length
    # --------------
    step_length = {'p'         : 0.004,
                   'G'         : 10,
                   'sigma'     : 5,
                   'occupancy' : 0.05}

    def fit(
        self, 
        burn_in:int=2500,
        mc_step:int=5000,
        seed:int=None,
        verbose:int=10,
        track:int=0, 
        step_length:dict[str, float]=step_length
    ):
        """ MCMC fit.
        
        Additional Parameters
        ---------------------
        step_length : dict
            keys in the order of ('p', 'G', 'sigma', 'occupancy'),
            values in float.
        """
        super().fit(burn_in, mc_step, seed, verbose, track, step_length)
