# ===========================================================================
# core/threshold.py
#
# Toy-derived detection-efficiency turn-on for the sparsify gate, and the
# pinned per-type constraint scheme (fsmp_efficiency toy, commit 7459798;
# numbers from note/pulls.pq, per-type not per-channel).
#
# The gate is a fixed charge clip -- log-normal in q, at q50 = kappa_t * Gm
# with dimensionless log-width s_t.  Gm is EACH MODEL's own non-zero-
# renormalized charge-response mean (PMTSpectrumFitter.gain_gm), never an
# external ser-file number, so the per-channel structure is pure Gm scaling
# and no per-channel table is needed.  Only two per-type nuisances float,
# under a Gaussian pull (chi2_pen):
#
#     thr_loc   = a_t * kappa_t * Gm(spe)     (location; data may pull ~ +/-3%)
#     thr_scale = b_t * s_t                   (log-width; toy-supplied +/-5%)
#     -logL pull = 0.5 * [ ((a_t - 1)/da)^2 + ((b_t - 1)/db)^2 ]
#
# A free per-channel threshold is occupancy-degenerate (corr +0.59); pinning
# the location to the model gain removes that.
# ===========================================================================

import numpy as np

from .base import ParamBlock

# Per PMT type (junoid pmt_type: 2 = MCP, 1 = Dynode).  The MCP/Dynode split
# is physical (different SER shape -> different deconvolution/charge response),
# not measurement scatter.  kappa from the 38-channel survey; s_log from the
# per-type toy fit (note/pulls.pq holds these two values, broadcast per type).
KAPPA = {"MCP": 0.211, "Dynode": 0.205}    # q50 / Gm
S_LOG = {"MCP": 0.280, "Dynode": 0.292}    # log-normal width (dimensionless)
PULL_DA = 0.03    # location nuisance sigma: SER-shape channel scatter (systematic)
PULL_DB = 0.05    # width nuisance sigma: toy-supplied (data cannot constrain it)


def pmt_type_name(pmt_type):
    """junoid pmt_type code (2 = MCP, else Dynode) -> scheme key."""
    if pmt_type in KAPPA:            # already a name
        return pmt_type
    return "MCP" if int(pmt_type) == 2 else "Dynode"


def lognormal_thres_block(pmt_type):
    """Log-normal threshold block for a PMT type, in the fitter's gain units.

    Initialised at (kappa_t, s_t): the gain unit q = Q_raw / spe_scale makes
    the model gain O(1), so thr_loc ~ kappa is a good start; the pull then
    moves it to kappa * gain_gm(spe) exactly.  The box is a generous +/-6
    sigma -- the Gaussian pull, not the bounds, does the constraining.
    """
    t = pmt_type_name(pmt_type)
    loc0, s0 = KAPPA[t], S_LOG[t]
    return ParamBlock(
        name="threshold",
        names=["thr_loc", "thr_scale"],
        init=np.array([loc0, s0]),
        bounds=[(loc0 * (1.0 - 6 * PULL_DA), loc0 * (1.0 + 6 * PULL_DA)),
                (s0 * (1.0 - 6 * PULL_DB), s0 * (1.0 + 6 * PULL_DB))],
    )


def lognormal_thres_penalty(pmt_type):
    """Gaussian pull toward the pin (negative-log prior, subtracted from logL).

    a_t = thr_loc / (kappa_t * Gm(spe)), b_t = thr_scale / s_t, with Gm the
    model's own non-zero-renormalized gain.  Returns a closure
    pen(thres, spe, gain_gm) matching PMTSpectrumFitter's thres_penalty
    contract; the fitter supplies its own gain_gm so thr_loc tracks the
    fitted gain live.
    """
    t = pmt_type_name(pmt_type)
    kappa, s0 = KAPPA[t], S_LOG[t]

    def pen(thres, spe, gain_gm):
        a = thres[0] / (kappa * gain_gm(spe))
        b = thres[1] / s0
        return 0.5 * ((a - 1.0) / PULL_DA) ** 2 + 0.5 * ((b - 1.0) / PULL_DB) ** 2

    return pen
