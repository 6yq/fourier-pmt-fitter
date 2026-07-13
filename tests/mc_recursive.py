#!/usr/bin/env python3
"""MC closure test for RecursivePolyaFitter (official kup-gain convention).

Generates toy events from the exact generative process the model claims:
  N_pe ~ Poisson(mu) primaries; each primary is a direct Gamma(Q1, s1)
  w.p. w, else a Poisson(d1) sum of cascade-electron charges; a cascade
  electron is a Gamma(Q2, s2) w.p. w, else a Poisson(d2) brood of further
  cascade electrons (branching process, subcritical for (1-w)*d2 < 1).

Checks (full spectrum, threshold=None):
  1. parameter recovery (fit vs truth)
  2. lam = detected-PE intensity: fit lam vs MC mu*(1 - P(S=0)), and
     occupancy vs MC fired fraction
  3. component curves estimate_component_counts(n) vs MC sub-histograms
     of events with exactly n DETECTED PEs (peak position + counts)
  4. zero category: estimate_zero_count vs MC zero-charge events
  5. hard-truncation robustness: drop all Q < 150 from the data (keep A);
     the atom + below-edge zero scheme should still recover the truth.

Outputs: fitter/tests/out/mc_recursive.{log,pdf} (one plot per page, no subplots).
Run: /mnt/stage/liuyq/tao/venv/bin/python fitter/tests/mc_recursive.py
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
from fitter import RecursivePolyaFitter
from fitter.models.polya import _cascade_extinction

OUT = os.path.join(os.path.dirname(__file__), "out")
COMP_COLORS = {1: "darkorange", 2: "forestgreen", 3: "#2166AC"}

# truth (subcritical: (1-w)*d2 = 0.48)
W, Q1, S1, D1, D2, Q2, S2 = 0.4, 666.67, 200.0, 3.0, 0.8, 333.3, 133.3
MU, A = 0.3, 200_000
RNG = np.random.default_rng(20260709)


def gamma(rng, mean, sigma, size):
    k = (mean / sigma) ** 2
    return rng.gamma(k, mean / k, size=size)


def cascade_charges(n):
    """Total charge of n independent cascade electrons (vector)."""
    out = np.zeros(n)
    active = np.arange(n)                      # event index of each live electron
    while len(active):
        direct = RNG.random(len(active)) < W
        q = gamma(RNG, Q2, S2, direct.sum())
        np.add.at(out, active[direct], q)
        # branch: each non-direct electron spawns Poisson(D2) children
        children = RNG.poisson(D2, (~direct).sum())
        active = np.repeat(active[~direct], children)
    return out


def simulate():
    n_pe = RNG.poisson(MU, A)
    tot = np.zeros(A)
    npe_detected = np.zeros(A, dtype=int)
    ev = np.repeat(np.arange(A), n_pe)         # event index per primary
    n_prim = len(ev)
    direct = RNG.random(n_prim) < W
    q_prim = np.zeros(n_prim)
    q_prim[direct] = gamma(RNG, Q1, S1, direct.sum())
    # cascade primaries: Poisson(D1) cascade electrons each
    idx_c = np.where(~direct)[0]
    n_casc = RNG.poisson(D1, len(idx_c))
    owner = np.repeat(idx_c, n_casc)           # primary index per cascade electron
    qc = cascade_charges(len(owner))
    np.add.at(q_prim, owner, 0.0)              # keep shape
    q_casc_sum = np.zeros(n_prim)
    np.add.at(q_casc_sum, owner, qc)
    q_prim = q_prim + q_casc_sum
    np.add.at(tot, ev, q_prim)
    np.add.at(npe_detected, ev[q_prim > 0], 1)
    return tot, npe_detected


def fit(Q, label):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = RecursivePolyaFitter(Q_raw=Q[Q > 0], A=A, mode="binned", threshold=None)
        res = f.fit_mle()
    th = np.asarray(res.theta)
    spe = th[f.layout["spe"]]
    lam = float(th[f.layout["lam"]][0])
    rep = f.spe_report(spe)
    print(f"\n== {label}: converged={res.converged} logl={res.logl:.1f}")
    names = ["w", "Q1", "sigma1", "delta1", "delta2", "Q2", "sigma2"]
    truth = dict(zip(names, [W, Q1, S1, D1, D2, Q2, S2]))
    for n in names:
        fitv, tv = rep[n], truth[n]
        print(f"  {n:8s} fit {fitv:9.3f}  truth {tv:9.3f}  ({100*(fitv-tv)/tv:+.1f}%)")
    return f, th, spe, lam, rep


def page(pp, f, th, tot, npe_det, label):
    """One page: MC histogram + fit density + per-n curves vs MC markers."""
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
    ax.set_xlabel("Q [ADC·ns]")
    ax.set_ylabel(f"Entries [/ {w:.3g} ADC·ns]")
    ax.set_title(label)
    ax.legend(frameon=False, ncol=2)
    pp.savefig(fig); plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    tot, npe_det = simulate()
    p_s0 = float(_cascade_extinction(W, D2))
    pS = (1.0 - W) * np.exp(D1 * (p_s0 - 1.0))
    lam_true = MU * (1.0 - pS)
    fired_mc = float((tot > 0).mean())
    print(f"MC: {A} events, fired {fired_mc:.4f}, "
          f"P(S=0)={pS:.4f} -> lam_detected={lam_true:.4f}  "
          f"(1-exp(-lam)={1-np.exp(-lam_true):.4f})")

    # ---- full-spectrum fit ----
    f, th, spe, lam, rep = fit(tot, "full spectrum")
    print(f"  lam      fit {lam:9.4f}  truth {lam_true:9.4f}  "
          f"({100*(lam-lam_true)/lam_true:+.1f}%)")
    print(f"  gain(gm) fit {rep['gain']:9.1f}")
    print(f"  occupancy fit {f.occupancy(th):.4f}  MC fired {fired_mc:.4f}")
    zc = f.estimate_zero_count(th)
    print(f"  zero: est {zc:.0f}  MC {(tot == 0).sum()}")

    # ---- component check ----
    bins = np.asarray(f.bins)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    print("  components (exactly n detected PEs):")
    for n in (1, 2, 3):
        comp = np.asarray(f.estimate_component_counts(th, n))
        sel = tot[(npe_det == n) & (tot > 0)]
        hist, _ = np.histogram(sel, bins=bins)
        pk_fit = ctr[np.argmax(comp)]
        pk_mc = ctr[np.argmax(hist)] if hist.max() > 0 else np.nan
        print(f"    n={n}: peak fit {pk_fit:7.0f}  MC {pk_mc:7.0f}   "
              f"counts fit {comp.sum():8.1f}  MC {len(sel):6d}")

    # ---- hard-truncation robustness (atom + below-edge zero scheme) ----
    CUT = 150.0
    tot_cut = np.where(tot >= CUT, tot, 0.0)
    f2, th2, *_ = fit(tot_cut, f"truncated Q >= {CUT}")

    with PdfPages(os.path.join(OUT, "mc_recursive.pdf")) as pp:
        page(pp, f, th, tot, npe_det, f"MC closure  A={A}  $\\mu$={MU}")
        page(pp, f2, th2, tot_cut, npe_det,
             f"MC closure  truncated Q$\\geq${CUT:.0f}")
    print(f"\nsaved {OUT}/mc_recursive.pdf")


if __name__ == "__main__":
    main()
