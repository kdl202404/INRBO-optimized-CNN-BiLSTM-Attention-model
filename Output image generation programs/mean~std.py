import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib import rcParams

# ===================== Font settings =====================
rcParams["font.sans-serif"] = ["SimHei"]
rcParams["axes.unicode_minus"] = False

font_path = r"C:\Windows\Fonts\simhei.ttf"
my_font = font_manager.FontProperties(fname=font_path)

# ===================== Data =====================
models = [
    "XGBoost",
    "CNN-LSTM",
    "CNN-LSTM-Attention",
    "CNN-GRU",
    "CNN-GRU-Attention",
    "CNN-BiLSTM",
    "CNN-BiLSTM-Attention",
    "HEOA-CNN-BiLSTM-Attention",
    "SCNGO-CNN-BiLSTM-Attention",
    "NRBO-CNN-BiLSTM-Attention",
    "INRBO-CNN-BiLSTM-Attention"
]

x = np.arange(len(models))

# RMSE
rmse_mean = np.array([
    8.1489,
    10.6491,
    9.2704,
    8.0359,
    7.5203,
    7.6091,
    7.3610,
    6.8123,
    6.7473,
    6.4684,
    5.6247
])

rmse_std = np.array([
    0.3159,
    0.5336,
    0.7316,
    0.4907,
    0.3916,
    0.1990,
    0.2700,
    0.2516,
    0.0918,
    0.1876,
    0.1065
])

# MAE
mae_mean = np.array([
    6.1267,
    8.7436,
    7.5899,
    6.5175,
    6.0907,
    6.1030,
    5.9976,
    5.4382,
    5.3555,
    5.0323,
    4.5428
])

mae_std = np.array([
    0.2784,
    0.4557,
    0.5757,
    0.3176,
    0.4151,
    0.2636,
    0.3146,
    0.2923,
    0.1169,
    0.1541,
    0.1016
])

# ===================== Colors =====================
colors = [
    "#A9AEB6",  # XGBoost
    "#C8B6E2",  # CNN-LSTM
    "#B39DDB",  # CNN-LSTM-Attention
    "#85C1E9",  # CNN-GRU
    "#5DADE2",  # CNN-GRU-Attention
    "#76D7C4",  # CNN-BiLSTM
    "#F2C811",  # CNN-BiLSTM-Attention
    "#D18479",  # HEOA
    "#7F9AF2",  # SCNGO
    "#7DB55A",  # NRBO
    "#F39A73"   # INRBO proposed
]


# ===================== Drawing function =====================
def plot_metric(mean_values, std_values, ylabel, title, save_path):
    fig, ax = plt.subplots(figsize=(15, 6.5))

    # Bar chart
    bars = ax.bar(
        x,
        mean_values,
        width=0.72,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        zorder=2
    )

    # Mean points and standard deviation error bars
    ax.errorbar(
        x,
        mean_values,
        yerr=std_values,
        fmt="o",
        color="black",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        markersize=4,
        zorder=3
    )

    # X-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(
        models,
        rotation=35,
        ha="right",
        fontsize=9
    )

    ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Ensure error bars are fully displayed
    upper_limit = np.max(mean_values + std_values) * 1.12
    ax.set_ylim(0, upper_limit)

    ax.tick_params(axis="y", labelsize=11)
    # ax.grid(
    #     axis="y",
    #     linestyle="--",
    #     linewidth=0.6,
    #     alpha=0.5,
    #     zorder=0
    # )

    # Reduce excess whitespace on both sides
    ax.set_xlim(-0.6, len(models) - 0.4)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    # plt.show()
    # plt.close()


# ===================== RMSE figure =====================
plot_metric(
    mean_values=rmse_mean,
    std_values=rmse_std,
    ylabel="RMSE",
    title="各模型的 RMSE 对比（均值 ± 标准差）",
    save_path=r"D:\Desktop\RMSE-mean_std.jpg"
)

# ===================== MAE figure =====================
plot_metric(
    mean_values=mae_mean,
    std_values=mae_std,
    ylabel="MAE",
    title="各模型的 MAE 对比（均值 ± 标准差）",
    save_path=r"D:\Desktop\MAE-mean_std.jpg"
)