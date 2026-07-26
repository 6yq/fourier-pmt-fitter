#!/usr/bin/env python3
"""MC closure test for GammaTweedieNBFitter (NegBin secondary count).

Generative process (exact model claim): N_pe ~ Poisson(mu) primaries; each
is a direct Gamma(mean, sigma) w.p. frac, else a NegBin(mean=lam_c,
dispersion=r_nb) sum of small Gammas(mean*mean_t, mean*sigma_t) — single
generation.  Zero-charge atom: p0 = (1-frac) * (1 + lam_c/r_nb)^-r_nb.

Checks (gain units, full spectrum, threshold=None):
  1. get_gain("gm") FORMULA at truth params vs MC E[q_pe | q_pe > 0]
     (the 2026-07-17 zero-truncation fix: old frac_nz form printed too)
  2. parameter recovery
  3. lam = detected-PE intensity mu*(1-p0); zero atom vs MC zero events
  4. fitted gain vs the same MC conditioned mean
Outputs: fitter/tests/out/mc_tweedie_nb.pdf (+ stdout table)
Run: /mnt/stage/liuyq/tao/venv/bin/python fitter/tests/mc_tweedie_nb.py
"""
import sys
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fitter import GammaTweedieNBFitter

OUT = os.path.join(os.path.dirname(__file__), "out")

# truth: campaign-realistic corner (lam_c ~ 2.2, strong overdispersion)
FRAC, MEAN, SIGMA, LAM_C, MEAN_T, SIGMA_T, R_NB = \
    0.70, 1.0, 0.30, 2.2, 0.35, 0.15, 2.0
MU, A = 0.3, 200_000
RNG = np.random.default_rng(20260717)


def gamma(rng, mean, sigma, size):
    k = (mean / sigma) ** 2
    return rng.gamma(k, mean / k, size=size)


def sample_pe(rng, n):
    """n PE charges from the exact NB-GT law (zero atom included)."""
    direct = rng.random(n) < FRAC
    q = np.zeros(n)
    q[direct] = gamma(rng, MEAN, SIGMA, direct.sum())
    idx_c = np.where(~direct)[0]
    n_sec = rng.negative_binomial(R_NB, R_NB / (R_NB + LAM_C), len(idx_c))
    owner = np.repeat(idx_c, n_sec)
    qs = gamma(rng, MEAN * MEAN_T, MEAN * SIGMA_T, len(owner))
    np.add.at(q, owner, qs)
    return q


def simulate():
    n_pe = RNG.poisson(MU, A)
    ev = np.repeat(np.arange(A), n_pe)
    q_pe = sample_pe(RNG, len(ev))
    tot = np.zeros(A)
    np.add.at(tot, ev, q_pe)
    return tot, q_pe


def main():
    os.makedirs(OUT, exist_ok=True)
    truth_spe = [FRAC, MEAN, SIGMA, LAM_C, MEAN_T, SIGMA_T, R_NB]
    p0 = (1.0 - FRAC) * (1.0 + LAM_C / R_NB) ** (-R_NB)
    lam_true = MU * (1.0 - p0)

    tot, q_pe = simulate()
    gm_mc = float(q_pe[q_pe > 0].mean())
    n_nz = int((q_pe > 0).sum())
    gm_err = float(q_pe[q_pe > 0].std() / np.sqrt(n_nz))

    f = GammaTweedieNBFitter(Q_raw=tot[tot > 0], A=A, mode="binned",
                             threshold=None)
    gm_new = f.get_gain(truth_spe, "gm")
    frac_nz = FRAC / (1.0 - p0)
    gm_old = frac_nz * MEAN + (1.0 - frac_nz) * MEAN * MEAN_T * LAM_C
    print(f"MC: {A} events, {len(q_pe)} PEs, p0={p0:.4f} "
          f"lam_detected={lam_true:.4f}")
    print(f"FORMULA CHECK  E[q|q>0]: MC {gm_mc:.4f} +- {gm_err:.4f}  "
          f"fixed {gm_new:.4f} ({100*(gm_new-gm_mc)/gm_mc:+.2f}%)  "
          f"old(buggy) {gm_old:.4f} ({100*(gm_old-gm_mc)/gm_mc:+.2f}%)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = f.fit_mle()
    th = np.asarray(res.theta)
    spe = th[f.layout["spe"]]
    lam = float(th[f.layout["lam"]][0])
    rep = f.spe_report(spe)
    print(f"\n== fit: converged={res.converged} logl={res.logl:.1f}")
    truth = dict(zip(
        ["frac", "spe_mean", "spe_sigma", "lam_c", "mean_t", "sigma_t",
         "r_nb"], truth_spe))
    truth["ts_mean"] = MEAN * MEAN_T
    truth["ts_sigma"] = MEAN * SIGMA_T
    truth["p0"] = p0
    for n in ("frac", "spe_mean", "spe_sigma", "lam_c", "r_nb", "ts_mean",
              "ts_sigma", "p0"):
        fv, tv = rep[n], truth[n]
        print(f"  {n:9s} fit {fv:9.4f}  truth {tv:9.4f}  "
              f"({100*(fv-tv)/tv:+.1f}%)")
    # atom-kept convention: lam = RAW primary intensity mu; detected
    # intensity = lam * (1 - p0)
    lam_det = lam * (1.0 - rep["p0"])
    print(f"  lam(raw)  fit {lam:9.4f}  truth {MU:9.4f}  "
          f"({100*(lam-MU)/MU:+.1f}%)")
    print(f"  lam(det)  fit {lam_det:9.4f}  truth {lam_true:9.4f}  "
          f"({100*(lam_det-lam_true)/lam_true:+.1f}%)")
    zc = f.estimate_zero_count(th)
    nz_mc = int((tot == 0).sum())
    print(f"  ZERO ATOM: est {zc:.0f}  MC {nz_mc}  "
          f"({100*(zc-nz_mc)/nz_mc:+.3f}%)")
    gm_fit = f.get_gain(spe, "gm")
    print(f"  gain(gm)  fit {gm_fit:9.4f}  MC E[q|q>0] {gm_mc:9.4f}  "
          f"({100*(gm_fit-gm_mc)/gm_mc:+.1f}%)")

    bins = np.asarray(f.bins)
    w = bins[1] - bins[0]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.stairs(np.asarray(f.hist), bins, fill=True, color="C0", alpha=0.2,
              label=f"MC {int(f.hist.sum())}")
    ax.plot(np.asarray(f.grid.xsp), np.asarray(f.estimate_density(th)) * w,
            color="red", lw=1.4, label="fit")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 2.5 * max(float(f.hist.max()), 1.0))
    ax.set_xlabel("q [gain units]")
    ax.set_ylabel(f"Entries [/ {w:.3g}]")
    ax.legend(frameon=False)
    with PdfPages(os.path.join(OUT, "mc_tweedie_nb.pdf")) as pp:
        pp.savefig(fig)
    plt.close(fig)
    print(f"\nsaved {OUT}/mc_tweedie_nb.pdf")


if __name__ == "__main__":
    main()
