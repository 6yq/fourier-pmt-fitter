import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_histogram_with_fit(
    bins,
    hist,
    xsp,
    smooth,
    comps,
    labels,
    params,
    ch=None,
    gp=None,
    gp_std=None,
    gm=None,
    gm_std=None,
    occ=None,
    occ_std=None,
    chiSq=None,
    ndf=None,
    zero=None,
    zeroReal=None,
    ys=None,
    logscale=False,
    ax_main=None,
    ax_resid=None,
    fig=None,
):
    if ax_main is None:
        fig, ax_main = plt.subplots()

    bin_centers = (bins[:-1] + bins[1:]) / 2

    # If occupancy is very small (< 0.01), use scientific notation
    if occ < 0.01:
        exponent = int(np.floor(np.log10(occ)))
        mantissa = occ / 10**exponent
        mantissa_std = occ_std / 10**exponent
        occ_str = rf"({mantissa:.3f}$\pm${mantissa_std:.3f})$\times10^{exponent}$"
    else:
        occ_str = rf"{occ:.3f}$\pm${occ_std:.3f}"

    ax_main.stairs(
        hist,
        bins,
        fill=True,
        color="C0",
        alpha=0.2,
        label=f"occupancy: {occ_str}",
    )
    ax_main.errorbar(
        bin_centers,
        hist,
        yerr=np.sqrt(hist),
        marker="o",
        linestyle="none",
        ecolor="black",
        mec="black",
        mfc="red",
        ms=3,
    )
    ax_main.plot(
        xsp,
        smooth,
        color="red",
        label=(f"$\chi^2$/ndf: {chiSq:.2f}/{ndf}" if chiSq is not None else "fit"),
    )

    if len(params) == 6 and gp is not None:
        ax_main.axvline(
            gp,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=(
                f"Gp={gp:.2f}$\pm${gp_std:.2f}"
                if gp_std is not None
                else f"Gp={gp:.2f}"
            ),
        )
        ax_main.axvline(
            gm,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=(
                f"Gm={gm:.2f}$\pm${gm_std:.2f}"
                if gm_std is not None
                else f"Gm={gm:.2f}"
            ),
        )
    elif gm is not None:
        ax_main.axvline(
            gm,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label=(
                f"G={gm:.2f}$\pm${gm_std:.2f}" if gm_std is not None else f"G={gm:.2f}"
            ),
        )

    for comp, _ in zip(comps, labels):
        ax_main.plot(xsp, comp, color="black", alpha=0.6, linestyle="--")

    # xmin, xmax = ax_main.get_xlim()
    # ymin, ymax = ax_main.get_ylim()

    # if zero is not None:
    #     ax_main.scatter(0, zero, label=f"0 PE (fit): {zero:.0f}")
    # if zeroReal is not None:
    #     ax_main.scatter(0, zeroReal, label=f"0 PE (true): {zeroReal}")

    # ax_main.set_xlim(xmin, xmax)
    # ax_main.set_ylim(ymin, ymax)

    if logscale:
        ax_main.set_yscale("log")

    ax_main.grid(True, which="both", linestyle="--", alpha=0.3)
    ax_main.set_ylabel("Entries")
    ax_main.set_xlabel("Q")
    if ch is not None:
        ax_main.set_title(f"ch {ch}")
    ax_main.legend(frameon=False)

    # plot residual if ax_resid
    if ax_resid is not None:
        residuals = hist - ys
        ax_resid.axhline(0, color="gray", linewidth=1, linestyle="--")
        ax_resid.plot(bin_centers, residuals, "o", color="black", ms=3)
        ax_resid.grid(True, which="both", linestyle="--", alpha=0.3)
        ax_resid.set_ylabel("Residual")
        ax_resid.set_xlabel("Q")
        ax_main.label_outer()

    return fig, ax_main, ax_resid
