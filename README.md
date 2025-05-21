# PMT-Fourier-Fitter

A modular and customizable PMT (Photomultiplier Tube) charge spectrum fitter based on FFT-based convolution.

This package is designed to model and fit PMT charge spectra by simulating the convolution of single photoelectron (PE) responses with Poisson-distributed occupancies using FFT, allowing for detailed error propagation, posterior sampling, and highly customizable physical modeling.

This work is inspired by Kalousis's fitter [here](https://github.com/kalousis/PMTCalib/).

---

## 🔧 Features

- ✨ **Customizable physical models**: define your own PE response shapes and likelihood logic
- 📈 **FFT-based convolution**: accurate and efficient modeling of nPE spectra
- 🧮 **MCMC posterior sampling** for uncertainty quantification (via `emcee`)
- 📏 **Chi-square, BIC, occupancy, gain statistics** available post-fit
- 🧩 Supports complex models: pedestal, compound response, δ-like peak, etc.
- ✅ **Constraint handling**: support for bounds and linear constraints

---

## 📦 Installation

Clone and install locally:

```
git clone https://github.com/6yq/fourier-pmt-fitter
cd fourier-fitter
pip install .
```

---

## 📚 Dependencies

This package requires:

```
numpy
scipy
emcee
```

They will be automatically installed via `pip`.

---

## 🛠 File Structure

```
fourier-fitter/
├── pmt_fitter.py       # Core fitter logic (base class)
├── tweedie_pdf.py      # Optional: compound Gamma-Tweedie model
└── setup.py            # Package metadata
```

---

## 🚀 Basic Usage

### 🔧 Quick Fit

```python
from pmt_fitter import MCP_Fitter

# Get histogram
hist, bins = np.histogram(charge_data, bins=..., range=...)

# Fit using auto-init (detects peaks automatically)
fitter = MCP_Fitter(hist, bins, auto_init=True)
fitter.fit()

# Access results
print(fitter.occ, fitter.occ_std)
print(fitter.ped_args, fitter.ped_args_std)
print(fitter.ser_args, fitter.ser_args_std)
print(fitter.chi_sq, fitter.ndf)
print(fitter.BIC)
```

**Note**:
- If `isWholeSpectrum=True`, the pedestal will be automatically modeled as a Gaussian and its parameters occupy the first two slots in the parameter array.
- The `fit()` method uses MCMC (`emcee`) and samples the posterior distribution. You can extract full trace via `samples_track` or `log_l_track`.

### 🔍 Checking MCMC Convergence

```python
import matplotlib.pyplot as plt
log_l_track = np.array(fitter.log_l_track)

for i in range(log_l_track.shape[1]):
    plt.plot(log_l_track[100:, i])  # discard burn-in if needed

plt.xscale("log")
plt.xlabel("Step")
plt.ylabel("Log Likelihood")
plt.title("MCMC chain stability")
plt.show()
```

---

## 🧩 Custom Model Design

To use a custom PE model, subclass `PMT_Fitter` and override the following:

- `_pdf(self, args)` – returns the single-PE PDF given model parameters
- `get_gain(self, args)` – estimate gain
- (optional) `_zero()` and `const()` – for δ-like models (e.g., Tweedie)
- `_replace_spe_params()` and `_replace_spe_bounds()` – for `auto_init=True` support

### ✅ Example: Custom Gamma Model

```python
from pmt_fitter import PMT_Fitter
from scipy.stats import gamma

class Custom_PMT_Fitter(PMT_Fitter):
    def __init__(
        self,
        hist,
        bins,
        isWholeSpectrum=False,
        A=None,
        occ_init=None,
        sample=None,
        seterr="warn",
        init=[5.0, 1.0],  # e.g., shape and scale for Gamma
        bounds=[(0.1, None), (0.1, None)],  # shape > 0, scale > 0
        constraints=None,
        auto_init=False,
    ):
        super().__init__(
            hist,
            bins,
            A,
            occ_init,
            sample,
            seterr,
            init,
            bounds,
            constraints,
            auto_init,
        )

    def _pdf(self, args):
        shape, scale = args
        return gamma.pdf(self.xsp, a=shape, scale=scale)

    def get_gain(self, args, gain: str = "gm"):
        if gain == "gp":
            return (args[0] - 1) * args[1]
        elif gain == "gm":
            return args[0] * args[1]
        else:
            raise NameError(f"{gain} is not a legal parameter!")

    # different models have different acceptable parameter regions
    # Caution: if `auto_init=True`, the initial values are always mean and std of the peaks
    def _replace_spe_params(self, gp_init, sigma_init):
        self._init[0] = gp_init
        self._init[1] = sigma_init

    def _replace_spe_bounds(self, gp_bound, sigma_bound):
        gp_bound_ = (0.5 * gp_bound, 1.5 * gp_bound)
        sigma_bound_ = (0.05 * sigma_bound, 3 * sigma_bound)
        self.bounds[0] = gp_bound_
        self.bounds[1] = sigma_bound_
```

## 📏 Chi-Square Calculation

To prevent instability due to sparse bins, chi-square is calculated by **merging bins** (lowest counts from edges inward) until each bin has ≥5 entries.

This avoids low-statistic bins skewing the χ².  
Different schemes (e.g., full merge on first low-count bin) are possible and can be customized.

---

## 📉 BIC: Bayesian Information Criterion

BIC is computed as:

```
BIC = k * log(N) - 2 * logL
```

Where:
- `k` = number of model parameters
- `N` = total number of observations
- `logL` = log-likelihood

This allows comparing models with different complexity (e.g., Tweedie vs. Gamma) under a consistent selection principle.

---

## ⚠ Tips and Cautions

- If using `auto_init=True`, the initial parameters are estimated from histogram peak shape using `compute_init()`. If your model uses different parameterization, be sure to map mean/std properly.
- When `isWholeSpectrum=True`, your custom model will receive pedestal parameters inserted before the PE-related args.
- A good rule: `nwalkers >= 2 × (number of parameters)` for MCMC convergence.
- Occupancy and gain are inferred along with all model parameters in the posterior.

---

## 📩 Contact

Maintainer: Yiqi Liu  
Email: liuyiqi24@mails.tsinghua.edu.cn

