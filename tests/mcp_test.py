import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

from ..models.mcp import MCP_Fitter
from .toyMC import sample_from_pe
from .score import score_param_similarity
from .plot import plot_histogram_with_fit

mpl.rc_file("tests/matplotlibrc")

SEED = 42
SAMPLES = 10000000
INTENSITIES = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
OCCS = 1 - np.exp(-INTENSITIES)

INIT = [
    -10,  # pedestal mean (not used here)
    50,  # pedestal sigma (not used here)
    0.5,
    100,
    15,
    1.0,
    0.60,
    0.15,
]

resDfs = []

with PdfPages("mcp_fit.pdf") as pp:
    for occ in tqdm(OCCS):
        qs = sample_from_pe(SAMPLES, INIT, occ, SEED)
        bins = np.arange(10, max(qs) + 10, 10)
        hist, _ = np.histogram(qs, bins=bins)

        fit = MCP_Fitter(
            hist=hist,
            bins=bins,
            isWholeSpectrum=False,
            init=INIT[2:],
            occ_init=occ,
            A=SAMPLES,
            auto_init=False,
            seterr="ignore",
        )

        fit.fit(
            method="minuit",
            tol=1e-06,
            max_calls=10000,
            print_level=0,
        )

        # some magic regression to decide how many PEs would be plotted
        n = int(2.75 - 3 * np.log(1 - 0.9 * fit.occ))
        npe_comps = [
            fit.estimate_smooth_n(np.append(fit.ser_args, fit.occ), order)
            for order in range(1, n + 1)
        ]
        npe_labels = [f"{order} PE" for order in range(1, n + 1)]

        common_kwargs = dict(
            bins=bins,
            hist=hist,
            xsp=fit.xsp[abs(fit._shift) :],
            smooth=fit.smooth,
            comps=npe_comps,
            labels=npe_labels,
            params=fit.ser_args,
            occ=fit.occ,
            occ_std=fit.occ_std,
            chiSq=fit.chi_sq_neyman_B,
            ndf=fit.ndf,
            zero=fit.zs,
            zeroReal=int(fit.zero),
            ys=fit.ys,
        )

        # residual plot
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
                true=np.append(INIT[2:], occ),
                fitted_sigma=np.append(fit.ser_args_std, fit.occ_std),
            )
        )
        resDfs.append(res)

        pp.savefig(fig)
        plt.close(fig)

resDfs = pd.concat(resDfs)
resDfs.to_parquet("mcp_fit.pq")
