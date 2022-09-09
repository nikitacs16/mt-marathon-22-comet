#!/usr/bin/env python3

import matplotlib.pyplot as plt
from collections import defaultdict
import json
import numpy as np
import sys
sys.path.append("src_comet_eval")
import fig_utils

# scp euler:/cluster/work/sachan/vilem/comet-marathon-22/logs/*.json logs/


xticks = [
    "baseline", "0.5\nthreshold", "0.25\nthreshold", "0.75\nthreshold", "0.25\nmargin", "0.5\nmargin", "1\nmargin"
]
yticks = [
    "addition", "antonym-replacement", "copy-source", "do-not-translate",
    "hallucination", "lexical-overlap", "nonsense", "omission", "ordering-mismatch",
    "untranslated"
]

data = np.random.random((10, len(xticks)))

plt.figure(figsize=(7, 5))
plt.imshow(data, cmap="RdYlGn", aspect=0.5)

for row_i, row in enumerate(data):
    for col_i, val in enumerate(row):
        plt.text(
            col_i, row_i,
            f"{val:.1f}",
            ha="center", va="center",
        )

plt.title("kendall $\\tau$ on specific phenomena")
plt.xticks(range(len(xticks)), xticks)
plt.yticks(range(len(yticks)), yticks)
plt.tight_layout(pad=0.5)
plt.show()
