#!/usr/bin/env python3

import matplotlib.pyplot as plt
import sys
sys.path.append("src_comet_eval")
import fig_utils

data_ch_acc = [0.611, 0.606]
data_ch_tau = [0.223, 0.212]
data_mqm_pearson = [0.303, 0.276]
data_mqm_spearman = [0.346, 0.329]
data_mqm_kendall = [0.259, 0.246]

KWARGS = {"marker": "."}

plt.figure(figsize=(5.5, 4))

XTICKS = [0, 0.5]

ax1 = plt.gca()
ax2 = ax1.twinx()

ax2.plot(
    XTICKS,
    data_ch_acc,
    label="Challenge set accuracy",
    color=fig_utils.COLORS[0],
    **KWARGS
)
# ax2.plot(
#     XTICKS,
#     data_ch_tau,
#     label="Challenge set kendall $\\tau$",
#     color=fig_utils.COLORS[0],
#     linestyle=":",
#     **KWARGS
# )
ax1.plot(
    XTICKS,
    data_mqm_pearson,
    label="MQM correlation (Pearson)",
    color=fig_utils.COLORS[1],
    **KWARGS
)
ax1.plot(
    XTICKS,
    data_mqm_spearman,
    label="MQM correlation (Spearman)",
    color=fig_utils.COLORS[1],
    linestyle=":",
    **KWARGS
)
ax1.plot(
    XTICKS,
    data_mqm_kendall,
    label="MQM correlation (Kendall)",
    color=fig_utils.COLORS[1],
    linestyle="-.",
    **KWARGS
)

plt.xticks(XTICKS)

ax2.set_ylabel("Performance (challenge set)")
ax1.set_ylabel("Correlation (MQM)")

plt.xlabel("$\\alpha$ in DA-$\\alpha$ augmentation for $h^-$")

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()

plt.legend(
    h1+h2, l1+l2,
    ncol=2, loc="upper left", bbox_to_anchor=(-0.09, 1.3)
)
plt.tight_layout(rect=(0.01, 0, 0.95, 0.99), pad=0.1)
plt.show()
