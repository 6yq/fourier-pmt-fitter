#!/usr/bin/env python3
"""MC closure test for BiTruncGaussFitter (kup-gain default DYNODE model,
laser.yaml kDoubleGauss / TF1ConvTool::DoubleGauss).

Generative truth: N_pe ~ Poisson(mu) per trigger; each PE charge is a
zero-truncated Gaussian mixture -- w.p. w a TruncGauss(Q1, s1), else a
TruncGauss(f_Q*Q1, f_s*s1); a Gaussian pedestal N(mu0, sig0) is added to
every trigger.  Whole-spectrum fit (pedestal=True, threshold=None) must
recover the SER params, lam and gain (the truncated-Gaussian CF is a
discrete Fourier sum -- this validates it).

Run: /mnt/stage/liuyq/tao/venv/bin/python fitter/tests/mc_bitruncgauss.py
"""
import sys
import os
import warnings

import numpy as np
from scipy.stats import truncnorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fitter import BiTruncGaussFitter

OUT = os.path.join(os.path.dirname(__file__), "out")
# truth (gain units) — secondary at 0.27 +- 0.24 is genuinely truncated
Q1, S1, W, FQ, FS = 0.9, 0.3, 0.85, 0.35, 0.9
MU, MU0, SIG0, A = 0.8, 0.0, 0.08, 300_000
RNG = np.random.default_rng(20260725)


def tnorm(q, s, size):
    return truncnorm.rvs((0.0 - q) / s, np.inf, loc=q, scale=s, size=size,
                         random_state=RNG)


def simulate():
    n_pe = RNG.poisson(MU, A)
    ev = np.repeat(np.arange(A), n_pe)
    n = len(ev)
    direct = RNG.random(n) < W
    q = np.empty(n)
    q[direct] = tnorm(Q1, S1, int(direct.sum()))
    q[~direct] = tnorm(FQ * Q1, FS * S1, int((~direct).sum()))
    tot = RNG.normal(MU0, SIG0, A)          # pedestal noise on every trigger
    np.add.at(tot, ev, q)
    return tot, n_pe


def main():
    os.makedirs(OUT, exist_ok=True)
    tot, n_pe = simulate()
    lam_true = MU                            # continuous SER -> every PE detected
    print(f"MC: A={A} mu={MU} fired {(n_pe>0).mean():.4f} "
          f"(1-exp(-mu)={1-np.exp(-MU):.4f})")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = BiTruncGaussFitter(Q_raw=tot, A=A, mode="binned",
                               pedestal=True, threshold=None)
        r = f.fit_mle(maxiter=3000)
    th = np.asarray(r.theta)
    spe = th[f.layout["spe"]]; ex = th[f.layout["extra"]]; lam = float(th[f.layout["lam"]][0])
    rep = f.spe_report(spe)
    names = ["spe_mean", "spe_sigma", "w", "f_Q", "f_s"]
    truth = dict(zip(names, [Q1, S1, W, FQ, FS]))
    print(f"\n== closure: converged={r.converged} logl={r.logl:.1f}")
    ok = True
    for nm in names:
        fv, tv = rep[nm], truth[nm]
        dev = 100 * (fv - tv) / tv
        print(f"  {nm:9s} fit {fv:8.4f}  truth {tv:8.4f}  ({dev:+.1f}%)")
        if abs(dev) > 12:
            ok = False
    print(f"  ped_mean  fit {ex[0]:8.4f}  truth {MU0:8.4f}")
    print(f"  ped_sigma fit {ex[1]:8.4f}  truth {SIG0:8.4f}")
    print(f"  lam       fit {lam:8.4f}  truth {lam_true:8.4f}  "
          f"({100*(lam-lam_true)/lam_true:+.1f}%)")
    gain_true = W * (Q1) + (1 - W) * (FQ * Q1)   # approx (pre-truncation)
    print(f"  gain(gm)  fit {rep['gain']:8.4f}  (approx truth ~{gain_true:.3f})")
    print(f"  occupancy fit {f.occupancy(th):.4f}  MC fired {(n_pe>0).mean():.4f}")
    print("\nCLOSURE", "PASS" if (ok and r.converged and abs(lam-lam_true)/lam_true < 0.1)
          else "CHECK")

    bins = np.asarray(f.bins); ctr = 0.5 * (bins[:-1] + bins[1:]); wbin = bins[1] - bins[0]
    with PdfPages(os.path.join(OUT, "mc_bitruncgauss.pdf")) as pp:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.stairs(np.asarray(f.hist), bins, fill=True, color="C0", alpha=0.2,
                  label=f"MC {int(f.hist.sum())}")
        ax.plot(np.asarray(f.grid.xsp), np.asarray(f.estimate_density(th)) * wbin,
                color="red", lw=1.4, label="fit")
        for nn, col in ((1, "darkorange"), (2, "forestgreen"), (3, "#2166AC")):
            comp = np.asarray(f.estimate_component_counts(th, nn))
            ax.plot(ctr, comp, "--", color=col, lw=1.1, label=f"{nn} PE")
        ax.set_yscale("log"); ax.set_ylim(0.5, 3 * max(float(f.hist.max()), 1))
        ax.set_xlabel("Q [gain units]"); ax.set_ylabel(f"Entries [/ {wbin:.3g}]")
        ax.set_title(f"BiTruncGauss MC closure  A={A}  $\\mu$={MU}")
        ax.legend(frameon=False, ncol=2)
        pp.savefig(fig); plt.close(fig)
    print(f"saved {OUT}/mc_bitruncgauss.pdf")


if __name__ == "__main__":
    main()
