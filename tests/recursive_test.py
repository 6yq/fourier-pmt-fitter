import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

from ..models.recursive import Recursive_Fitter
from .toyMC import sample_from_pe_recursive
from .score import score_param_similarity
from .plot import plot_histogram_with_fit

mpl.rc_file("tests/matplotlibrc")

SEED = 42
SAMPLES = 2_000_000
INTENSITIES = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0])
OCCS = 1 - np.exp(-INTENSITIES)

TRUE = [
    0.40,  # w
    100.0,  # mean1
    15.0,  # sigma1
    2.0,  # lam1
    1.2,  # lam2
    0.60,  # mean2_rel = mean2/mean1
    0.20,  # sigma2_rel = sigma2/mean1
]

INIT = [
    -10.0,  # pedestal mean (unused)
    50.0,  # pedestal sigma (unused)
    *TRUE,
]

resDfs = []

with PdfPages("recursive_fit.pdf") as pp:
    for occ in tqdm(OCCS, desc="fitting recursive"):
        qs = sample_from_pe_recursive(SAMPLES, TRUE, occ, SEED)

        binw = 10.0
        bins = np.arange(binw, max(qs) + binw, binw)
        hist, _ = np.histogram(qs, bins=bins)

        fit = Recursive_Fitter(
            hist=hist,
            bins=bins,
            isWholeSpectrum=False,
            init=INIT[2:],
            occ_init=occ,
            A=SAMPLES,
            auto_init=False,
            seterr="ignore",
            fit_total=False,
        )

        fit.fit(
            method="minuit",
            tol=1e-06,
            max_calls=10000,
            print_level=0,
        )

        n = int(2.75 - 3 * np.log(1 - 0.9 * fit.occ))
        npe_comps = [
            fit.estimate_smooth_n(np.append(fit.ser_args, fit.occ), order)
            for order in range(1, n + 1)
        ]
        npe_labels = [f"{order} PE" for order in range(1, n + 1)]

        common_kwargs = dict(
            bins=bins,
            hist=hist,
            xsp=fit.xsp,
            smooth=fit.smooth,
            comps=npe_comps,
            labels=npe_labels,
            params=fit.ser_args,
            occ=fit.occ,
            occ_std=fit.occ_std,
            chiSq=fit.chi_sq_pearson,
            ndf=fit.ndf,
            zero=fit.zs,
            zeroReal=int(fit.zero),
            ys=fit.ys,
        )

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 0.15], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_resid = fig.add_subplot(gs[1], sharex=ax_main)

        plot_histogram_with_fit(
            **common_kwargs,
            logscale=False,
            ax_main=ax_main,
            ax_resid=ax_resid,
            fig=fig,
        )
        ax_main.set_ylim(ymax=1.2 * max(hist))

        pp.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.85, 0.15], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_resid = fig.add_subplot(gs[1], sharex=ax_main)

        plot_histogram_with_fit(
            **common_kwargs,
            logscale=True,
            ax_main=ax_main,
            ax_resid=ax_resid,
            fig=fig,
        )
        ax_main.set_ylim(ymin=1e-01, ymax=2 * max(hist))

        res = pd.DataFrame.from_dict(
            score_param_similarity(
                fitted=np.append(fit.ser_args, fit.occ),
                true=np.append(TRUE, occ),
                fitted_sigma=np.append(fit.ser_args_std, fit.occ_std),
            )
        )
        resDfs.append(res)

        pp.savefig(fig)
        plt.close(fig)

resDfs = pd.concat(resDfs)
resDfs.to_parquet("recursive_fit.pq")
