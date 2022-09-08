#!/usr/bin/env python3

import matplotlib.pyplot as plt
from collections import defaultdict
import json
import sys
sys.path.append("src_comet_eval")
import fig_utils

# scp euler:/cluster/work/sachan/vilem/comet-marathon-22/logs/*.json logs/

data = {}
data["baseline"] = {
    "xticks": ["baseline"],
    "files": ["baseline"],
    "ch_acc": [0.611],
    "ch_tau": [0.223],
    "mqm_pearson": [0.303],
    # "mqm_spearman": [0.346],
    "mqm_kendall": [0.259],
}
data["threshold"] = {
    "xticks": ["0.5\nthreshold"],
    "files": ["aug_05"],
    "ch_acc": [0.606],
    "ch_tau": [0.212],
    "mqm_pearson": [0.276],
    # "mqm_spearman": [0.329],
    "mqm_kendall": [0.246],
}
data["margin"] = {
    "xticks": ["0.25\nmargin", "0.5\nmargin", "1\nmargin"],
    "files": ["ch025_e4", "ch05_e2", "ch1_e1"],
    "ch_acc": [0.594, 0.557, 0.534],
    "ch_tau": [0.194, 0.115, 0.068],
    "mqm_pearson": [0.053, 0.034, -0.006],
    # "mqm_spearman": [0.346, 0.329],
    "mqm_kendall": [0.027, 0.035, -0.004],
}

KWARGS = {}

plt.figure(figsize=(5.5, 4))


ax1 = plt.gca()
ax2 = ax1.twinx()

last_i = 0
xticks_all = []
for data_model_i, data_model in enumerate(data.values()):
    xticks = list(range(last_i, last_i + len(data_model["xticks"])))
    xticks_all += data_model["xticks"]
    last_i = xticks[-1] + 1

    data_local = defaultdict(list)
    for fname in data_model["files"]:
        with open("logs/acc_" + fname + ".json", "r") as f:
            data_tmp = json.load(f)
            for k, v in data_tmp.items():
                data_local["ch_" + k].append(v)
        with open("logs/corr_" + fname + ".json", "r") as f:
            data_tmp = json.load(f)
            for k, v in data_tmp["avg"].items():
                data_local["mqm_" + k].append(v)

    if "ch_avg_diff_pos" in data_local:
        print(
            "avg diff:",
            ",".join([f"{x:.3f}" for x in data_local["ch_avg_diff"]])
        )
        print(
            "avg diff (pos)",
            ", ".join([f"{x:.3f}" for x in data_local["ch_avg_diff_pos"]])
        )
        print(
            "avg diff (neg)",
            ", ".join([f"{x:.3f}" for x in data_local["ch_avg_diff_neg"]])
        )

    ax2.plot(
        xticks,
        data_local["ch_acc"],
        label="Challenge set accuracy" if data_model_i == 0 else None,
        color=fig_utils.COLORS[0],
        marker="s",
        **KWARGS
    )
    ax2.plot(
        xticks,
        data_local["ch_tau"],
        label="Challenge set kendall $\\tau$" if data_model_i == 0 else None,
        color=fig_utils.COLORS[0],
        linestyle=":",
        marker="v",
        **KWARGS
    )
    ax1.plot(
        xticks,
        data_local["mqm_pearson"],
        label="MQM correlation (Pearson)" if data_model_i == 0 else None,
        color=fig_utils.COLORS[1],
        marker="*",
        **KWARGS
    )
    # ax1.plot(
    #     XTICKS,
    #     data_local["mqm_spearman"],
    #     label="MQM correlation (Spearman)" if data_model_i == 0 else None,
    #     color=fig_utils.COLORS[1],
    #     linestyle=":",
    #     **KWARGS
    # )
    ax1.plot(
        xticks,
        data_local["mqm_kendall"],
        label="MQM correlation (Kendall)" if data_model_i == 0 else None,
        color=fig_utils.COLORS[1],
        linestyle="-.",
        marker="x",
        **KWARGS
    )

plt.xticks(range(len(xticks_all)), xticks_all)

ax2.set_ylabel("Performance (challenge set)")
ax1.set_ylabel("Correlation (MQM)")

plt.xlabel("$\\alpha$ in DA-$\\alpha$ augmentation for $h^-$")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

plt.legend(
    h1 + h2, l1 + l2,
    ncol=2, loc="upper left", bbox_to_anchor=(-0.09, 1.3)
)
plt.tight_layout(rect=(0.01, 0, 0.95, 0.99), pad=0.1)
plt.show()
