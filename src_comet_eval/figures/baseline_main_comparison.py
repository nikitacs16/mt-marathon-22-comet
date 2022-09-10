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
}
data["threshold"] = {
    "xticks": ["0.25\nthreshold", "0.5\nthreshold", "0.75\nthreshold", "dynamic\nthreshold", "static\nthreshold", "unpair dyn\nthreshold"],
    "files": ["threshold_025", "aug_05", "threshold_075", "threshold_dyn", "threshold_dyn_extra", "threshold_dyn_unpair"],
}
data["+reg"] = {
    "xticks": ["contrastive+da\nmulti", "0.1\nmargin+da", "0.25\nmargin+da"],
    "files": ["contrastive_multi_reg", "margin_01_reg", "margin_025_reg"],
}
data["margin"] = {
    "xticks": ["0.1\nmargin", "0.25\nmargin", "0.5\nmargin", "1\nmargin"],
    "files": ["margin_01", "ch025_e4", "ch05_e2", "ch1_e1"],
}
data["contrastive"] = {
    "xticks": ["contrastive\nsingle", "contrastive\nmulti"],
    "files": ["contrastive", "contrastive_multi"],
}

KWARGS = {}

plt.figure(figsize=(12.5, 7.1))


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

plt.xticks(
    range(len(xticks_all)),
    [("\n\n" if x_i % 2 else "") + x for x_i, x in enumerate(xticks_all)]
)

ax2.set_ylabel("Performance (challenge set)")
ax1.set_ylabel("Correlation (MQM)")

plt.xlabel("$\\alpha$ in DA-$\\alpha$ augmentation for $h^-$")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

plt.legend(
    h1 + h2, l1 + l2,
    ncol=4, loc="upper left", bbox_to_anchor=(-0.04, 1.1)
)
plt.tight_layout(rect=(0.01, 0, 1, 0.99), pad=0.1)
plt.savefig("figures/baseline_main_comparison.pdf")
plt.show()
