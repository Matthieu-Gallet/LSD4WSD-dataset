import matplotlib as mpl
import datetime, h5py
import numpy as np
import pandas as pd
from dataset_load import load_h5_II

mpl.use("pgf")

import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.texsystem": "pdflatex",
        "pgf.preamble": "\n".join(
            [
                r"\usepackage[utf8x]{inputenc}",
                r"\usepackage[T1]{fontenc}",
                r"\usepackage{cmbright}",
            ]
        ),
    }
)


def plot_bilan(d, save):
    M3 = [datetime.datetime.strptime(str(i), "%Y%m%d") for i in d[1]["metadata"][:, 0]]

    M = d[1]["metadata"][:, 1]
    data = np.unique(M, return_counts=True)
    massifs = data[0]
    values = 100 * (data[1] / data[1].sum())

    M2 = d[1]["metadata"][:, 2]
    data2 = np.unique(M2, return_counts=True)
    types = data2[0]
    values2 = 100 * (data2[1] / data2[1].sum())

    df = pd.DataFrame(
        d[0].reshape(-1, 9),
        columns=[
            "VV",
            "VH",
            "VV/VH",
            "Elevation (m)",
            "Orientation (°)",
            "Slope (°)",
            "VV/VVref",
            "VH/VHref",
            "VV*VHref/VH*VVref",
        ],
    )
    ddf = df.describe()
    ddf.drop(["count"], axis=0, inplace=True)
    fig, ax = plt.subplots(figsize=(20, 3))

    ############################################
    y_pos = np.arange(len(massifs))
    for i, v in enumerate(values):
        ax.barh(
            0,
            v,
            align="center",
            alpha=0.5,
            edgecolor="black",
            linewidth=1,
            left=values[:i].sum(),
        )
        ax.text(
            values[:i].sum() + v / 2,
            0,
            f"\n{massifs[i]}: {v.round(2)}%\n {data[1][i]}",
            color="black",
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
            fontsize=8,
        )
    ax.set_title("Number of samples per Massif")
    ax.axis("off")

    ############################################
    ax2 = fig.add_axes([0.14, -0.25, 0.75, 0.2])
    y_pos = np.arange(len(types))
    for i, v in enumerate(values2):
        ax2.barh(
            0,
            v,
            align="center",
            alpha=0.5,
            edgecolor="black",
            linewidth=1,
            left=values2[:i].sum(),
        )
        ax2.text(
            values2[:i].sum() + v / 2,
            0,
            f"\n{types[i]}: {v.round(2)}%\n {data2[1][i]}",
            color="black",
            fontweight="bold",
            va="center",
            ha="center",
            fontsize=10,
        )
    ax2.set_title("Number of samples per type of acquisition")
    ax2.axis("off")

    ############################################
    ax3 = fig.add_axes([0.155, -1.15, 0.7, 0.65])
    ax3.hist(M3, edgecolor="black", linewidth=1, bins=25, align="mid", alpha=0.85)
    ax3.set_title("Number of samples per date")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Number of samples")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.set_yscale("log")
    ax3.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.75)

    ############################################
    ax4 = fig.add_axes([0.155, -2.15, 0.7, 0.65])

    h = d[1]["topography"][:, 0]
    dd = np.unique(h, return_counts=True)
    ax4.bar(
        dd[0],
        dd[1],
        width=300,
        edgecolor="black",
        linewidth=1,
        color="tab:brown",
        align="center",
        alpha=0.85,
    )
    ax4.set_yscale("log")
    ax4.set_xlabel("Elevation (m)")
    ax4.set_xticks(dd[0])
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["left"].set_visible(False)
    ax4.set_yscale("log")
    ax4.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.75)
    ax4.set_title("Number of samples per step of elevation (300m)")

    ############################################
    ax5 = fig.add_axes([0.155, -3.15, 0.7, 0.65])
    h2 = d[1]["topography"][:, 2]
    dd = np.unique(h2, return_counts=True)
    ax5.bar(
        dd[0],
        dd[1],
        width=45,
        edgecolor="black",
        linewidth=1,
        color="tab:green",
        align="center",
        alpha=0.85,
    )
    ax5.set_yscale("log")
    ax5.set_xlabel("Orientation (°)")
    ax5.set_xticks(dd[0])
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)
    ax5.spines["left"].set_visible(False)
    ax5.set_yscale("log")
    ax5.grid(True, which="both", linestyle="-", linewidth=0.5, alpha=0.75)
    ax5.set_title("Number of samples per step of orientation (45°)")

    ############################################
    ax61 = fig.add_axes([0, -4.1, 1, 0.65])
    ax61.spines["top"].set_visible(False)
    ax61.spines["right"].set_visible(False)
    ax61.spines["left"].set_visible(False)
    ax61.spines["bottom"].set_visible(False)
    ax61.axis("off")
    ax61.set_title("Number of samples per physical parameter simulated by Crocus")

    ax6 = fig.add_axes([0.125, -4.15, 0.225, 0.65])
    ax7 = fig.add_axes([0.38, -4.15, 0.225, 0.65])
    ax8 = fig.add_axes([0.64, -4.15, 0.225, 0.65])

    h3 = d[1]["physics"][:, 2]
    ax6.hist(
        d[1]["physics"][:, 2],
        edgecolor="black",
        linewidth=1,
        bins=75,
        align="mid",
        alpha=0.85,
    )
    ax6.set_yscale("log")
    ax6.set_xlabel("Liquid water content of the snowpack ($kg/m^2$)")
    ax6.set_ylabel("Number of samples")
    # --------------------------------------------
    ax7.hist(
        d[1]["physics"][:, 1],
        edgecolor="black",
        linewidth=1,
        bins=75,
        align="mid",
        alpha=0.85,
    )
    ax7.set_yscale("log")
    ax7.set_xlabel("Height of snow (m)")
    ax7.set_ylabel("Number of samples")
    # --------------------------------------------
    ax8.hist(
        d[1]["physics"][:, 0],
        edgecolor="black",
        linewidth=1,
        bins=75,
        align="mid",
        alpha=0.85,
    )
    ax8.set_yscale("log")
    ax8.set_ylabel("Number of samples")
    ax8.set_xlabel("Temperature minimal of the snowpack (°C)")

    ############################################
    ax9 = fig.add_axes([0.125, -5.2, 0.85, 0.65])
    rowo = [i for i in ddf.index.values.astype(str)]
    colo = [i for i in ddf.columns.values.astype(str)[:3]]
    ax9.table(
        cellText=ddf.values.round(2).astype(str)[:, :3],
        rowLabels=rowo,
        colLabels=colo,
        colWidths=[0.1] * 3,
        loc="center",
        cellLoc="center",
        fontsize=48,
    )
    ax9.axis("off")
    ax9.set_title("Statistics of the dataset (VV,VH,VV/VH)")

    # --------------------------------------------
    ax10 = fig.add_axes([0.125, -6.1, 0.85, 0.65])
    # colo = ["\\textbf{" + i + "}" for i in ddf.columns.values.astype(str)[3:6]]
    colo = [i for i in ddf.columns.values.astype(str)[3:6]]
    ax10.table(
        cellText=ddf.values.round(2).astype(str)[:, 3:6],
        rowLabels=rowo,
        colLabels=colo,
        colWidths=[0.2] * 3,
        loc="center",
        cellLoc="center",
        fontsize=15,
    )
    ax10.axis("off")
    ax10.set_title("Statistics of the dataset (Elevation,Orientation,Slope)")

    # --------------------------------------------
    ax11 = fig.add_axes([0.125, -7.0, 0.85, 0.65])
    # colo = ["\\textbf{" + i + "}" for i in ddf.columns.values.astype(str)[6:9]]
    colo = [i for i in ddf.columns.values.astype(str)[6:9]]
    ax11.table(
        cellText=ddf.values.round(2).astype(str)[:, 6:9],
        rowLabels=rowo,
        colLabels=colo,
        colWidths=[0.2] * 3,
        loc="center",
        cellLoc="center",
        fontsize=48,
    )
    ax11.axis("off")
    ax11.set_title("Statistics of the dataset (VV/VVref,VH/VHref,VV*VHref/VH*VVref)")

    if save is not None:
        plt.savefig(save, bbox_inches="tight", dpi=300, backend="pgf")


if __name__ == "__main__":
    p = "../dataset_/Vfinal_V4_V5_15/dataset_AD_08200821_14Mas3Top3Phy_W15.h5"
    d = load_h5_II(p)

    save = "../dataset_/Vfinal_V4_V5_15/bilan_dataset.pdf"
    plot_bilan(d, save)
