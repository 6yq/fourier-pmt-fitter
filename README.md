# PMT-Fourier-Fitter

A modular and customizable PMT (Photomultiplier Tube) charge spectrum fitter based on FFT-based convolution.

This package is designed to model and fit PMT charge spectra by simulating the convolution of single photoelectron (PE) responses with Poisson-distributed occupancies using FFT, allowing for detailed error propagation, posterior sampling, and highly customizable physical modeling.

This work is inspired by Kalousis's fitter [here](https://github.com/kalousis/PMTCalib/).

---

## 🔧 Features

- **Customizable physical models**: define your own PE response shapes
- **FFT-based convolution**: accurate and efficient modeling of nPE spectra
- **Supports complex models**: pedestal, compound response, δ-like peak, etc.
- **Constraint handling**: support for bounds and linear constraints
- **Multi optimizer**: support for `pyROOT` and `emcee`

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
```

They will be automatically installed via `pip`.

You might also need these packages:

```
pyROOT (if you want to use `Minuit` optimizer)
emcee (if you want to use `emcee` optimizer)
```

---

## 🛠 File Structure

```
fourier-fitter/
├── __init__.py
├── core                # Core fitter logic (base class)
│   ├── __init__.py
│   ├── base.py         # Base class
│   ├── fft_utils.py    # FFT-based convolution
│   └── utils.py        # Helper functions
├── models              # PMT charge models
│   ├── __init__.py
│   ├── dynode.py       # dynode PMT fitter goes here
│   ├── mcp.py          # MCP PMT fitter goes here
|   |── polya_exp.py    # Polya-exponential fitter goes here
│   └── tweedie_pdf.py  # Helper function for MCP's Gamma-Tweedie model
├── README.md
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
# Use Minuit by default
fitter.fit(method="minuit")

# Access results
print(fitter.occ, fitter.occ_std)
print(fitter.ped_args, fitter.ped_args_std)
print(fitter.ser_args, fitter.ser_args_std)
print(fitter.chi_sq, fitter.ndf)
print(fitter.gp, fitter.gm)
print(fitter.likelihood)
```

**Note**:
- If `isWholeSpectrum=True`, the pedestal will be automatically modeled as a Gaussian and its parameters occupy the first two slots in the parameter array.
- If `isWholeSpectrum=False`, you might need threshold effect by giving `threshold="erf"` or `threshold="logistic"`.
- The `fit()` method using `Minuit` is much faster.
- The `fit()` method using MCMC (`emcee`) stores samples from the posterior distribution. You can extract full trace via `samples_track` or `log_l_track`.

#### 🔍 Checking MCMC Convergence

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
- (optional) `const()` – for δ-like models (e.g., Tweedie)
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
        init=[5.0, 1.0],  # e.g., mean and sigma for Gamma
        bounds=[(0, None), (0, None)],
        constraints=[
            {"coeffs": [(1, 1), (2, -1)], "threshold": 0, "op": ">"},
        ],  # ensure a peak
        threshold=None, # 
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
        mean, sigma = args
        k = (mean / sigma) ** 2
        theta = mean / k
        return gamma.pdf(self.xsp, a=k, scale=theta)

    def get_gain(self, args, gain: str = "gm"):
        mean, sigma = args
        k = (mean / sigma) ** 2
        theta = mean / k
        if gain == "gp":
            return (k - 1) * theta
        elif gain == "gm":
            return mean
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

    # Only necessary if SPE response contains δ component.
    # You might want to give the proportion here.
    def _const(self, args):
        return 0
```

## 📏 Chi-Square Calculation

You might want to go through `core/utils.py` to see the pre-defined chi-square functions:

| Function | Description | Fomula |
| --- | --- | --- |
| `modified_neyman_chi2_A` | Modified Neyman chi-square (A) | $E^O / O - O$ |
| `modified_neyman_chi2_B` | Modified Neyman chi-square (B) | $(E - O)^2 / O$
| `mighell_chi2` | Mighell chi-square | $(E + 1 - O) ^ 2 / (O + 1)$ |
| `merged_pearson_chi2` | Merged Pearson chi-square | $(E - O)^2 / E (E \ge 5)$ |

---

## ⚠ Tips and Cautions

- If using `auto_init=True`, the initial parameters are estimated from histogram peak shape using `compute_init()`. If your model uses different parameterization, be sure to map mean/std properly.

---

## 📩 Contact

Maintainer: Yiqi Liu  
Email: liuyiqi24@mails.tsinghua.edu.cn

