# fourier-fitter

A PMT charge spectrum fitter built on JAX.  The spectrum model is
assembled analytically in the Fourier domain, the likelihood is an
extended Poisson (binned or unbinned), and the maximization runs
L-BFGS-B on exact JAX gradients with Hessian-based errors.

Inspired by Kalousis's fitter (https://github.com/kalousis/PMTCalib/)
and sharing its core architecture with the SiPM charge fitter used by
the TAO reconstruction chain.

## Model

The number of photoelectrons per trigger is Poisson(lam).  A single PE
leaves charge with characteristic function f(w) (the SER model), which
may include a discrete probability mass p0 at exactly zero charge
(compound SER models).  The total PE charge spectrum in Fourier space is

    pgf(f(w); lam) = exp(lam * (f(w) - 1))

Four readout settings are supported, in any combination:

1. Pedestal on (whole spectrum): every trigger is recorded and the
   Gaussian pedestal noise g0 is convolved in,

       G(w) = g0(w) * exp(lam * (f(w) - 1))

2. Pedestal off (PE spectrum): only non-zero charges are histogrammed.
   Total charge is exactly zero when every PE leaves zero charge, with
   probability pgf(p0) (the n = 0 term gives exp(-lam)), so the
   recorded continuous density is

       G_c(w) = exp(lam * (f(w) - 1)) - exp(lam * (p0 - 1))

3. Threshold on: a hardware discriminator multiplies the recorded
   density by an efficiency eff(q) in charge space ("erf" or
   "logistic", two parameters: location and scale).

4. Pedestal and threshold together.

All settings share one bookkeeping rule: with A total triggers and
P_obs the probability of landing in the histogram window (after the
efficiency), the zero category (undetected, below threshold, or out of
window) has probability 1 - P_obs and enters the extended-Poisson
log-likelihood as an extra bin:

    log L = sum_k [ n_k log(A y_k) - A y_k ]
          + n_0 log(A (1 - P_obs)) - A (1 - P_obs) - log C

Without a threshold the bin integrals of G are computed analytically in
Fourier space (one matrix product over all bin edges).  With a
threshold, G(q) * eff(q) is evaluated by IFFT on a sub-sampled grid
aligned with the bin edges and integrated per bin with a composite
Simpson rule.  In unbinned mode the density at each recorded charge is
evaluated by a type-2 NUFFT, no binning involved.

## Models

Gaussian family (`models/gauss.py`):

| Class | SER | Parameters |
|---|---|---|
| `GaussFitter` | single Gaussian | mean, sigma |
| `BiGaussFitter` | normal + missing-first-dynode | ratio, mean, sigma, mean_r, sigma_r |
| `LinearGaussFitter` | normal + scaled-down copy | df, ds, mean, sigma |
| `TriGaussFitter` | three Gaussians, stick-breaking weights | v1, v2, m1, d12, d23, s1, s2, s3 |
| `GaussCompoundFitter` | Gaussian + compound Poisson of Gaussians | frac, mean, sigma, lam_c, mean_ts, sigma_ts |

Polya family (`models/polya.py`):

| Class | SER | Parameters |
|---|---|---|
| `PolyaFitter` | single Gamma | mean, sigma |
| `BiPolyaFitter` | Gamma + missing-first-dynode Gamma | frac, mean, sigma, mean_t, sigma_t |
| `PolyaExpFitter` | Gamma + exponential | frac, mean, sigma, exp_scale |
| `GammaTweedieFitter` | Gamma + compound Poisson of Gammas | frac, mean, sigma, lam_c, mean_t, sigma_t |
| `RecursivePolyaFitter` | MCP recursive secondary emission (Lambert-W) | frac, mean, sigma, lam, lam_r, mean_r, sigma_r |

The compound and recursive models carry a discrete mass at zero charge;
it is handled exactly through `p0` (see Model above).  Every fitter
exposes `get_gain(spe, "gm"/"gp")` and `spe_report(spe)`.

## Usage

```python
import numpy as np
from fourier_fitter import PolyaFitter, GaussFitter, CombinedFitter

# PE spectrum (pedestal subtracted), unbinned fit.
# A is the total trigger count including zero-PE events.
f = PolyaFitter(Q_raw=charges, A=n_triggers)
res = f.fit_mle()

print(res.converged, res.logl)
print(dict(zip(f.param_names, res.theta)))
spe, spe_err = res.block("spe")
print(f.get_gain(spe), f.occupancy(res.theta))

# Whole spectrum with pedestal, from a histogram.
hist, bins = np.histogram(charges, bins=200)
f = GaussFitter(hist=hist, bins=bins, pedestal=True)
res = f.fit_mle()

# PE spectrum with a hardware threshold.
f = GaussFitter(hist=hist, bins=bins, A=n_triggers, threshold="erf")
res = f.fit_mle()

# Joint fit of several intensities with shared SER (one lam each).
fitters = [PolyaFitter(hist=h, bins=b, A=n) for h, b, n in runs]
res = CombinedFitter(fitters).fit_mle()
spe, spe_err = res.spe()
lams, lam_errs = res.lams()
```

Initial values and bounds are derived from the data (occupancy from the
zero fraction, SPE charge scale from the spectrum mean, pedestal from
the tallest peak or the histogram left edge).  Pass `extra_block`,
`thres_block`, `spe_block` (`ParamBlock`) or `lam_init` / `scale` to
override.  Plotting helpers: `estimate_density`, `estimate_bin_counts`,
`estimate_component_counts(theta, n)` for the n-PE decomposition.

`fit_mle` runs a small curated multi-start when a threshold is enabled
(the truncated pedestal and the efficiency curve can trade against each
other and create separate likelihood basins); the best final likelihood
wins.

## Custom models

Subclass `PMTSpectrumFitter` and provide:

- `_model_callables()` returning `(ser_ft, p0_fn, count_pgf)`;
  `ser_ft(freq, spe)` is the SER characteristic function under the
  numpy.fft sign convention (a Gaussian is
  `exp(-1j * mean * freq - sigma**2 * freq**2 / 2)`), including any
  discrete mass at zero.  `p0_fn(spe)` returns that mass (or None for
  0).  `count_pgf` may be None (Poisson).
- `_default_spe_block()` returning a `ParamBlock`; `self.scale` holds
  the data-driven SPE charge scale.
- `get_gain(spe, kind)` and `spe_report(spe)`.

## File structure

```
fourier_fitter/
├── core/
│   ├── fft_grid.py     # uniform FFT grid aligned with bin edges
│   ├── likelihood.py   # spectrum assembly, bin integrals, log-likelihoods
│   ├── lambert_w.py    # JAX Lambert-W (recursive MCP model)
│   ├── base.py         # PMTSpectrumFitter (blocks, MLE, accessors)
│   └── combined.py     # joint multi-spectrum fitter
├── models/
│   ├── gauss.py        # Gaussian family
│   └── polya.py        # Polya family
└── tests/              # pytest suite (toy MC closure for all settings)
```

## Dependencies

numpy, scipy, jax.  Unbinned mode additionally needs jax-finufft.

## Tests

```
python -m pytest fourier_fitter/tests
```

The suite covers grid alignment, Lambert-W accuracy on its physical
domain, characteristic functions against toy MC samplers, both bin
integration paths, and full closure fits for the four readout settings,
the zero-mass model and the combined fit.

## Contact

Maintainer: Yiqi Liu
Email: liuyiqi24@mails.tsinghua.edu.cn
