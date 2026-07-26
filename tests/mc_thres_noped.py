#!/usr/bin/env python3
"""MC closure test for a THRESHOLDED, pedestal-less spectrum.

The hardware-triggered corner of the (pedestal, threshold) matrix, which no
other test exercises: A triggers, compound-Poisson charge with a Polya SER,
and a soft (erf) discriminator that only records events above threshold, so
the pedestal is never histogrammed.  Fit is pedestal=False, threshold="erf".

Checks:
  1. recovery of lam, SER mean and the threshold (loc, scale) when the
     trigger count A is supplied
  2. that A is MANDATORY: with A defaulting to the recorded count the zero
     category vanishes and lam / gain / threshold are jointly unidentifiable

Caveat: generation uses the same erf the fitter assumes, so this validates
the machinery, not a discriminator-shape mismatch.
Run: /mnt/stage/liuyq/tao/venv/bin/python fitter/tests/mc_thres_noped.py
"""
import sys
import os
import warnings

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fitter import PolyaFitter

# truth: lam ~ 1, SER shape 1.4, threshold well above the pedestal
LAM, GAIN, THETA = 1.2, 1.0, 0.4
T_LOC, T_SCALE = 0.35, 0.08
A = 400_000
RNG = np.random.default_rng(20260726)
TOL = 0.05          # fractional tolerance for the A-known closure


def simulate():
    """A triggers -> compound-Poisson charge -> soft discriminator."""
    n = RNG.poisson(LAM, A)
    tot = np.zeros(A)
    ev = np.repeat(np.arange(A), n)
    k = 1.0 + THETA
    np.add.at(tot, ev, RNG.gamma(k, GAIN / k, len(ev)))
    p_acc = 0.5 * (1.0 + erf((tot - T_LOC) / (T_SCALE * np.sqrt(2.0))))
    return tot, RNG.random(A) < p_acc


def fit(Q_rec, A_used, label):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = PolyaFitter(Q_raw=Q_rec, A=A_used, mode="binned",
                        pedestal=False, threshold="erf")
        res = f.fit_mle()
    th = np.asarray(res.theta)
    lam = float(th[f.layout["lam"]][0])
    thres = np.asarray(th[f.layout["thres"]])
    rep = f.spe_report(th[f.layout["spe"]])
    print(f"\n== {label}: converged={res.converged} logl={res.logl:.1f} "
          f"zero-category={f.grid.zero}")
    out = {}
    for name, fitv, tv in (("lam", lam, LAM),
                           ("spe_mean", rep["spe_mean"], GAIN),
                           ("thres_loc", float(thres[0]), T_LOC),
                           ("thres_scale", float(thres[1]), T_SCALE)):
        print(f"  {name:12s} fit {fitv:9.4f}  truth {tv:9.4f}  "
              f"({100 * (fitv - tv) / tv:+6.1f}%)")
        out[name] = (fitv, tv)
    return out


def main():
    tot, rec = simulate()
    Q_rec = tot[rec]
    print(f"A={A} triggers, {rec.sum()} above threshold ({100*rec.mean():.1f}%); "
          f"pedestal never recorded")

    known = fit(Q_rec, A, "A KNOWN (trigger count supplied)")
    bad = fit(Q_rec, None, "A UNKNOWN (defaults to recorded count)")

    off = {k: abs(v[0] - v[1]) / v[1] for k, v in known.items()}
    ok = all(v <= TOL for v in off.values())
    print(f"\nA known : max |bias| {100*max(off.values()):.1f}% "
          f"(tol {100*TOL:.0f}%) -> {'PASS' if ok else 'FAIL'}")
    lam_bias = abs(bad["lam"][0] - LAM) / LAM
    degen = lam_bias > 1.0 and bad["thres_loc"][0] < 0.5 * T_LOC
    print(f"A unknown: lam bias {100*lam_bias:+.0f}%, threshold collapses to "
          f"{bad['thres_loc'][0]:.4f} -> {'as expected' if degen else 'UNEXPECTED'}")
    print("\nverdict: thresholded pedestal-less fits work, but A is mandatory "
          "-- the zero category carries all below-threshold information.")
    return 0 if (ok and degen) else 1


if __name__ == "__main__":
    sys.exit(main())
