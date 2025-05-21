# PMT-Fourier-Fitter

A modular and customizable PMT (Photomultiplier Tube) charge spectrum fitter based on Fourier convolution.

This package is designed to model and fit PMT charge spectra by simulating the convolution of single photoelectron (PE) responses with Poisson-distributed occupancies using FFT, allowing for detailed error propagation and model flexibility.

This work is based on Kalousis's fitter [here](https://github.com/kalousis/PMTCalib/).

---

## 🔧 Features

- ✨ **Customizable physical model**: plug in your own PE response shape
- 📉 **nPE and all-PE modeling** with Poisson statistics
- 🧮 **Error analysis** via MCMC posterior estimation
- 🧩 Extendable to multi-component fits (e.g., pedestal + PE + afterpulse)

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

The following will be automatically installed:

```
[numpy](https://numpy.org/)
[scipy](https://scipy.org/)
[emcee](https://emcee.readthedocs.io/)
```

---

## 📁 File Structure

```
fourier-fitter/
├── pmt_fitter.py       # Core class (FFT + PDF + likelihood)
├── tweedie_pdf.py      # Custom compound distribution (optional)
└── setup.py            # Package installer
```

---

## 🚀 Example Usage

### 🎨 fitting

```python
from pmt_fitter import MCP_Fitter

# hist, bins = np.histogram(charge_data, ...)
# auto_init = True: automatically compute initial values based on the peaks on the histogram
# Caution: the initial values are always mean and std of the peaks
fitter = MCP_Fitter(hist, bins, auto_init=True)
fitter.fit()

# Access parameters
print(fitter.ser_args)
print(fitter.occ)            # Occupancy
print(fitter.chi_sq)         # Chi-square
print(fitter.samples_track)  # Posterior samples
```

Both **PE** and **without-threshold** spectra are supported.  
The first two parameters within the model would be substituded with the mean and the standard deviation of the pedestal (assumed to be Gaussian distribution), respectively.

### 🧩 Custom Model

The base class `PMT_Fitter` is designed to be easily extended for custom single-photoelectron (SPE) models.

You can override `_pdf()` to define your own SPE shape (e.g. Gamma, Gaussian mixture, etc.), and supply the corresponding initial parameters, parameter bounds, and optional constraints for optimization.

#### ✅ Supported Features

- **Customizable initial values (`init`)**:  
  List of model parameters to initialize the fitting.
  
- **Flexible parameter bounds (`bounds`)**:  
  Each parameter can have a bounded interval `(lower, upper)`, or be unbounded using `None`.

- **Support for linear constraints (`constraints`)**:  
  Expressed as dictionaries, like:

  ```python
  constraints=[
      {"coeffs": [(0, 1.0), (1, -0.5)], "threshold": 1.0, "op": ">="}  # param[0] - 0.5 * param[1] ≥ 1.0
  ]
  ```

  These constraints are enforced during likelihood evaluation to avoid unphysical regions.

- **Correction for gain and occupancy (`_gain()` and `_zero()`)**:  
  Override these methods to apply gain and zero-suppression effects to the model spectrum.

#### 🛠️ Example

```python
from pmt_fitter import PMT_Fitter
from scipy.stats import gamma

class Custom_PMT_Fitter(PMT_Fitter):
    def __init__(
        self,
        hist,
        bins,
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

You may also override `_pdf_ped()` if you want to customize the pedestal (default: Gaussian). The framework will automatically compute the full model spectrum using FFT-based convolution with proper phase correction and occupancy effects.

---

## 📩 Contact

Maintainer: Yiqi Liu  
Email: liuyiqi24@mails.tsinghua.edu.cn

