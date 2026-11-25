import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from network import NODES

EXPORT_DPI = 700
FIGURE_MM = 180, 105
FONT_FAMILY = "DejaVu Sans"
COLORS = {
    "original": "#293F55",
    "csi300": "#8397AA",
    "allshare": "#548579",
    "row": "#293F55",
    "scalar": "#548579",
    "normalization": "#293F55",
    "representation": "#548579",
    "reference": "#595959",
    "text": "#333333",
    "grid": "#E7E7E7",
}
TCI_XLIM = "2014-01-01", "2025-12-31"
TCI_YLIM = 40, 99
INCREMENT_XLIM = 0, 25
OVERLAP_XLIM = 0, 4
SIM_RECALL_YLIM = 0, 112
SIM_MAGNITUDE_TOP_MARGIN = 0.05


def load_figure4(frames):
    grid = frames["figure4"]
    data = grid[grid.N.eq(24)]
    data = data.pivot(index="date", columns="representation", values="row_h10")
    data = data.reset_index()
    data = data.rename(
        columns={"raw": "raw24_h10", "hs300": "hs300_24_h10", "allA": "allA24_h10"}
    )
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date")
    data = data.reset_index(drop=True)
    return data


def load_figureD1(frames):
    source = frames["figureD1"]
    order = ["raw_row", "hs300_row", "allA_row"]
    data = source.loc[(source.period == "full") & source.spec.isin(order)].copy()
    data = data.set_index("spec")
    data = data.loc[order]
    data = data.reset_index()
    return data


PAIRS = [
    ("raw_row", "raw_scalar", "Original: Row vs Scalar"),
    ("allA_row", "allA_scalar", "All Share: Row vs Scalar"),
    ("hs300_row", "hs300_scalar", "CSI 300: Row vs Scalar"),
    ("raw_row", "allA_row", "Row: Original vs All Share"),
    ("raw_row", "hs300_row", "Row: Original vs CSI 300"),
    ("allA_row", "hs300_row", "Row: All Share vs CSI 300"),
    ("raw_scalar", "allA_scalar", "Scalar: Original vs All Share"),
    ("raw_scalar", "hs300_scalar", "Scalar: Original vs CSI 300"),
    ("allA_scalar", "hs300_scalar", "Scalar: All Share vs CSI 300"),
]


def load_figure2(frames):
    source = frames["figure2"]
    data = source.loc[source.k == 5].copy()
    indexed = data.set_index(["spec1", "spec2"])
    order = []
    for a, b, _ in PAIRS:
        order.append((a, b))
    data = indexed.loc[order]
    data = data.reset_index()
    return data


DGPS = ["factor_only", "lagged_only", "factor_plus_lagged"]
RECALL_METRICS = ["row_true_source_top3_recall", "scalar_true_source_top3_recall"]


def style():
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "normal",
            "axes.titlepad": 8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 9.5,
            "figure.titleweight": "normal",
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["reference"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "axes.unicode_minus": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "lines.linewidth": 1.15,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.45,
            "legend.frameon": False,
            "legend.handlelength": 2.3,
            "figure.figsize": tuple(mm / 25.4 for mm in FIGURE_MM),
            "figure.dpi": 120,
            "savefig.dpi": EXPORT_DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "path.simplify": False,
            "savefig.bbox": None,
        }
    )


def plot_figure4(frames):
    data = load_figure4(frames)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.11, right=0.975, bottom=0.16, top=0.86)
    for col, label, color, dash in [
        ("raw24_h10", "Original", "original", "-"),
        ("hs300_24_h10", "CSI 300 adjusted", "csi300", "--"),
        ("allA24_h10", "All Share adjusted", "allshare", "-."),
    ]:
        ax.plot(data.date, data[col], label=label, color=COLORS[color], linestyle=dash)
    ax.set(
        xlim=tuple(pd.Timestamp(x) for x in TCI_XLIM),
        ylim=TCI_YLIM,
        ylabel="Total connectedness index (%)",
        xlabel="Year",
    )
    ax.set_yticks(np.arange(40, 100, 10))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y")
    ax.axvline(pd.Timestamp("2024-01-01"), color=COLORS["reference"], lw=0.9, ls="--")
    ax.text(
        pd.Timestamp("2024-02-01"),
        98.5,
        "Mapping frozen\n2024–2025",
        fontsize=8,
        va="top",
        color=COLORS["reference"],
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.975),
        ncol=3,
    )
    return fig


def plot_figureD1(frames):
    data = load_figureD1(frames)
    values = 100 * data.NET_change_rel_L1.to_numpy()
    low = 100 * data.NET_change_rel_L1_ci95_low.to_numpy()
    high = 100 * data.NET_change_rel_L1_ci95_high.to_numpy()
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.27, right=0.975, bottom=0.19, top=0.95)
    y = np.arange(3)
    colors = []
    for name in ["original", "csi300", "allshare"]:
        colors.append(COLORS[name])
    error = np.vstack([values - low, high - values])
    ax.barh(
        y,
        values,
        height=0.52,
        color=colors,
        xerr=error,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9, "ecolor": COLORS["text"]},
    )
    ax.set_yticks(y, ["Original", "CSI 300 adjusted", "All Share adjusted"])
    ax.invert_yaxis()
    ax.set(xlim=INCREMENT_XLIM, xlabel="Mean relative L1 increment (%)")
    ax.set_xticks(np.arange(0, 26, 5))
    ax.grid(axis="x")
    for i, (v, hi) in enumerate(zip(values, high)):
        ax.text(hi + 0.55, i, f"{v:.2f}%", va="center", fontsize=8.5)
    fig.text(
        0.615,
        0.035,
        "H=1 to H=10; error bars: 95% circular-block bootstrap CI",
        ha="center",
        fontsize=8,
    )
    return fig


def plot_figure2(frames):
    data = load_figure2(frames)
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.39, right=0.975, bottom=0.16, top=0.86)
    y = np.arange(len(data))
    colors = []
    for name in data.family:
        colors.append(COLORS[name])
    ax.barh(y, data.mean_topk_overlap, height=0.66, color=colors)
    labels = []
    for _, _, label in PAIRS:
        labels.append(label)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    reference = float(data.random_independent_topk_expected.iloc[0])
    ax.axvline(reference, color=COLORS["reference"], ls="--", lw=0.9)
    ax.set(xlim=OVERLAP_XLIM, xlabel="Mean Top-5 overlap (of 24 industries)")
    ax.set_xticks(np.arange(0, 4.01, 0.5))
    ax.grid(axis="x")
    for i, value in enumerate(data.mean_topk_overlap):
        ax.text(value + 0.06, i, f"{value:.2f}", va="center", fontsize=8.5)
    handles = [
        Patch(facecolor=COLORS["normalization"], label="Normalization"),
        Patch(facecolor=COLORS["representation"], label="Representation"),
        Line2D(
            [0],
            [0],
            color=COLORS["reference"],
            ls="--",
            lw=0.9,
            label=f"Random Top-5: {reference:.2f}",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        columnspacing=1.2,
    )
    return fig


def plot_figure5(frames):
    data = frames["figure5"].set_index(["dgp", "metric"])

    def value(dgp, metric):
        return float(data.loc[(dgp, metric), "mean"])

    magnitude = []
    for d in DGPS:
        magnitude.append(value(d, "row_h1_mean_abs_NET"))
    magnitude = np.array(magnitude)
    row = []
    scalar = []
    for d in DGPS[1:]:
        row.append(value(d, RECALL_METRICS[0]))
    for d in DGPS[1:]:
        scalar.append(value(d, RECALL_METRICS[1]))
    row = 100 * np.array(row)
    scalar = 100 * np.array(scalar)
    fig, (left, right) = plt.subplots(1, 2)
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.18, top=0.8, wspace=0.4)
    left.bar(np.arange(3), magnitude, color=COLORS["row"], width=0.68)
    left.set_xticks(np.arange(3), ["Factor only", "Lagged only", "Both"])
    left.set_ylabel("Mean |NET| at H=1 (pp)")
    left.text(0, 1.21, "(a)", transform=left.transAxes, fontsize=10, fontweight="bold")
    left.set_ylim(0, float(magnitude.max()) * (1 + SIM_MAGNITUDE_TOP_MARGIN))
    left.set_yticks(np.arange(0, 16, 2))
    for i, v in enumerate(magnitude):
        left.annotate(
            f"{v:.2f}",
            (i, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    x, width = np.arange(2), 0.3
    right.bar(x - width / 2, row, width, color=COLORS["row"])
    right.bar(x + width / 2, scalar, width, color=COLORS["scalar"])
    right.axhline(100 * 3 / len(NODES), color=COLORS["reference"], ls="--", lw=0.9)
    right.set_xticks(x, ["Lagged only", "Factor + lagged"])
    right.set(ylabel="Planted-source Top-3 recall (%)", ylim=SIM_RECALL_YLIM)
    right.set_yticks(np.arange(0, 101, 20))
    right.text(
        0, 1.21, "(b)", transform=right.transAxes, fontsize=10, fontweight="bold"
    )
    for shift, values in [(-width / 2, row), (width / 2, scalar)]:
        for i, v in enumerate(values):
            right.annotate(
                f"{v:.1f}",
                (i + shift, v),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
    for ax in (left, right):
        ax.grid(axis="y")
    left.legend(
        handles=[Patch(facecolor=COLORS["row"], label="Row")],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    handles = [
        Patch(facecolor=COLORS["row"], label="Row"),
        Patch(facecolor=COLORS["scalar"], label="Scalar"),
        Line2D(
            [0],
            [0],
            color=COLORS["reference"],
            ls="--",
            lw=0.9,
            label="Random Top-3: 12.5%",
        ),
    ]
    comparison_legend = right.legend(
        handles=handles[:2],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        columnspacing=1.4,
    )
    right.add_artist(comparison_legend)
    right.legend(handles=handles[2:], loc="upper center", bbox_to_anchor=(0.5, 1.1))
    fig.text(
        0.54,
        0.045,
        f"{int(data['n'].iloc[0])} replications per scenario",
        ha="center",
        fontsize=8,
    )
    return fig


def plot_finance(frames):
    data = frames["finance_net"]
    j = data.loc[data.sector.eq("J")]
    j = j.sort_values("date")
    j = j.copy()
    j["date"] = pd.to_datetime(j.date)
    style()
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.11, right=0.975, bottom=0.16, top=0.86)
    ax.axvspan(
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2025-12-31"),
        color=COLORS["grid"],
        alpha=0.45,
        lw=0,
        zorder=0,
    )
    for column, label, color, dash in [
        ("NET_raw_row", "Original returns", "original", "-"),
        ("NET_allA_row", "All Share adjusted", "allshare", "-."),
    ]:
        ax.plot(
            j.date, j[column], label=label, color=COLORS[color], linestyle=dash, lw=1.15
        )
    ax.axhline(0, color=COLORS["reference"], lw=0.9, ls="--", zorder=1)
    lo = np.floor(j[["NET_raw_row", "NET_allA_row"]].min().min() / 10) * 10 - 5
    hi = np.ceil(j[["NET_raw_row", "NET_allA_row"]].max().max() / 10) * 10 + 5
    ax.set(
        xlim=tuple(pd.Timestamp(x) for x in TCI_XLIM),
        ylim=(lo, hi),
        ylabel="Financial-sector NET (pp)",
        xlabel="Year",
    )
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y")
    ax.axvline(pd.Timestamp("2024-01-01"), color=COLORS["reference"], lw=0.75, ls=":")
    ax.text(
        pd.Timestamp("2024-02-01"),
        hi - 4,
        "Mapping frozen\n2024–2025",
        fontsize=8,
        va="top",
        color=COLORS["reference"],
    )
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.54, 0.975),
        ncol=2,
    )
    return fig


def draw(frames, out):
    out.mkdir(parents=True, exist_ok=False)
    mpl.use("Agg")
    style()
    for name, plot in [
        ("2", plot_figure2),
        ("3", plot_finance),
        ("4", plot_figure4),
        ("5", plot_figure5),
        ("D1", plot_figureD1),
    ]:
        fig = plot(frames)
        fig.savefig(out / f"Figure_{name}.png", dpi=700, bbox_inches="tight")
        plt.close(fig)
