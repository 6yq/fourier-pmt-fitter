#!/usr/bin/env python3
"""MC closure test for GammaTweedieFitter (kup-aligned, atom-stripped).

Generative process (exact model claim): N_pe ~ Poisson(mu) primaries; each
is a direct Gamma(Q1, s1) w.p. w, else a Poisson(d1) sum of small
Gammas(Q2, s2) — single generation.  Zero-charge atom: pS = (1-w) e^-d1.

Checks (gain units, full spectrum, threshold=None):
  1. parameter recovery
  2. lam = detected-PE intensity mu*(1-pS); occupancy vs MC fired fraction
  3. ZERO-CHARGE ATOM: estimate_zero_count vs MC zero-charge events
     (the stripped-atom bookkeeping is right iff these match)
  4. components: estimate_component_counts(n) vs MC exactly-n-detected-PE
     sub-histograms
  5. hard truncation q >= 0.25 (keep A): atom + below-edge zero scheme
Outputs: fitter/tests/out/mc_tweedie.{log,pdf}
Run: /mnt/stage/liuyq/tao/venv/bin/python fitter/tests/mc_tweedie.py
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
from fitter import GammaTweedieFitter

OUT = os.path.join(os.path.dirname(__file__), "out")
COMP_COLORS = {1: "darkorange", 2: "forestgreen", 3: "#2166AC"}

# truth (inside the kup boxes)
W, Q1, S1, D1, Q2, S2 = 0.4, 1.0, 0.3, 3.0, 0.35, 0.15
MU, A = 0.3, 200_000
RNG = np.random.default_rng(20260710)


def gamma(rng, mean, sigma, size):
    k = (mean / sigma) ** 2
    return rng.gamma(k, mean / k, size=size)


def simulate():
    n_pe = RNG.poisson(MU, A)
    tot = np.zeros(A)
    npe_det = np.zeros(A, dtype=int)
    ev = np.repeat(np.arange(A), n_pe)
    n_prim = len(ev)
    direct = RNG.random(n_prim) < W
    q_prim = np.zeros(n_prim)
    q_prim[direct] = gamma(RNG, Q1, S1, direct.sum())
    idx_c = np.where(~direct)[0]
    n_sec = RNG.poisson(D1, len(idx_c))
    owner = np.repeat(idx_c, n_sec)
    qs = gamma(RNG, Q2, S2, len(owner))
    q_sec = np.zeros(n_prim)
    np.add.at(q_sec, owner, qs)
    q_prim = q_prim + q_sec
    np.add.at(tot, ev, q_prim)
    np.add.at(npe_det, ev[q_prim > 0], 1)
    return tot, npe_det


def fit(Q, label):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = GammaTweedieFitter(Q_raw=Q[Q > 0], A=A, mode="binned", threshold=None)
        res = f.fit_mle()
    th = np.asarray(res.theta)
    spe = th[f.layout["spe"]]
    lam = float(th[f.layout["lam"]][0])
    rep = f.spe_report(spe)
    print(f"\n== {label}: converged={res.converged} logl={res.logl:.1f}")
    names = ["w", "Q1", "sigma1", "delta1", "Q2", "sigma2"]
    truth = dict(zip(names, [W, Q1, S1, D1, Q2, S2]))
    for n in names:
        fv, tv = rep[n], truth[n]
        print(f"  {n:8s} fit {fv:9.4f}  truth {tv:9.4f}  ({100*(fv-tv)/tv:+.1f}%)")
    return f, th, lam, rep


def page(pp, f, th, tot, npe_det, label):
    bins = np.asarray(f.bins)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    w = bins[1] - bins[0]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.stairs(np.asarray(f.hist), bins, fill=True, color="C0", alpha=0.2,
              label=f"MC {int(f.hist.sum())}")
    ax.plot(np.asarray(f.grid.xsp), np.asarray(f.estimate_density(th)) * w,
            color="red", lw=1.4, label="fit")
    for n in (1, 2, 3):
        comp = np.asarray(f.estimate_component_counts(th, n))
        ax.plot(ctr, comp, "--", color=COMP_COLORS[n], lw=1.2, label=f"{n} PE fit")
        sel = tot[(npe_det == n) & (tot >= bins[0]) & (tot > 0)]
        hist, _ = np.histogram(sel, bins=bins)
        ax.plot(ctr, hist, "o", ms=3, mfc="none", mec=COMP_COLORS[n],
                label=f"{n} PE MC")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 2.5 * max(float(f.hist.max()), 1.0))
    ax.set_xlabel("q [gain units]")
    ax.set_ylabel(f"Entries [/ {w:.3g}]")
    ax.set_title(label)
    ax.legend(frameon=False, ncol=2)
    pp.savefig(fig); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    tot, npe_det = simulate()
    pS = (1.0 - W) * np.exp(-D1)
    lam_true = MU * (1.0 - pS)
    fired_mc = float((tot > 0).mean())
    print(f"MC: {A} events, fired {fired_mc:.4f}, pS={pS:.4f} -> "
          f"lam_detected={lam_true:.4f} (1-exp(-lam)={1-np.exp(-lam_true):.4f})")

    f, th, lam, rep = fit(tot, "full spectrum")
    print(f"  lam      fit {lam:9.4f}  truth {lam_true:9.4f} "
          f"({100*(lam-lam_true)/lam_true:+.1f}%)")
    print(f"  gain(gm) fit {rep['gain']:.4f}")
    print(f"  occupancy fit {f.occupancy(th):.4f}  MC fired {fired_mc:.4f}")
    zc = f.estimate_zero_count(th)
    print(f"  ZERO ATOM CHECK: est {zc:.0f}  MC {(tot == 0).sum()}  "
          f"({100*(zc-(tot==0).sum())/(tot==0).sum():+.3f}%)")
    bins = np.asarray(f.bins); ctr = 0.5 * (bins[:-1] + bins[1:])
    print("  components (exactly n detected PEs):")
    for n in (1, 2, 3):
        comp = np.asarray(f.estimate_component_counts(th, n))
        sel = tot[(npe_det == n) & (tot > 0)]
        hist, _ = np.histogram(sel, bins=bins)
        pk_fit = ctr[np.argmax(comp)]
        pk_mc = ctr[np.argmax(hist)] if hist.max() > 0 else np.nan
        print(f"    n={n}: peak fit {pk_fit:7.3f}  MC {pk_mc:7.3f}   "
              f"counts fit {comp.sum():8.1f}  MC {len(sel):6d}")

    CUT = 0.25
    tot_cut = np.where(tot >= CUT, tot, 0.0)
    f2, th2, *_ = fit(tot_cut, f"truncated q >= {CUT}")

    with PdfPages(os.path.join(OUT, "mc_tweedie.pdf")) as pp:
        page(pp, f, th, tot, npe_det, f"GT MC closure  A={A}  $\\mu$={MU}")
        page(pp, f2, th2, tot_cut, npe_det, f"GT MC closure  truncated q$\\geq${CUT}")
    print(f"\nsaved {OUT}/mc_tweedie.pdf")


if __name__ == "__main__":
    main()
