#!/usr/bin/env python3

import matplotlib.pyplot as plt
from collections import defaultdict
import json
import argparse
import csv
import numpy as np
import sys
sys.path.append("src_comet_eval")
import fig_utils

# scp euler:/cluster/work/sachan/vilem/comet-marathon-22/logs/*.json logs/

ALLOWED_MODELS = [
    # 'Baseline (42220 ckpt)',
    'Baseline (52775 ckpt)',
    'aug-comet-05\n14876',
    # 'aug-comet-05\n29752',
    # 'margin-025\n20185',
    'margin-025\n25230',
    # 'margin-05\n5046',
    'margin-05\n1513',
    # 'margin-1\n5046',
    'margin-1\n10092',
    'margin-01',
    'margin-lse',
    'contrastive',
]

TRAINED_PHENOMENA = [
    "hallucination-date-time",
    "hallucination-named-entity-level-1",
    "hallucination-named-entity-level-2",
    "hallucination-named-entity-level-3",
    "hallucination-number-level-1",
    "hallucination-number-level-2",
    "hallucination-number-level-3",
    "hallucination-unit-conversion-amount-matches-ref",
    "hallucination-unit-conversion-unit-matches-ref",
]


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--csv", default="computed/heatmap_results.csv")
    args = args.parse_args()

    with open(args.csv, "r") as f:
        data = list(csv.DictReader(f))

    phenomena = [x.pop("phenomenon") for x in data]

    # put special phenomena to the back
    for phenomenon in TRAINED_PHENOMENA:
        phenomenon_i = phenomena.index(phenomenon)
        data.append(data.pop(phenomenon_i))

    # essentially transpos
    data = {
        k: [x[k] for x in data]
        for k in data[0].keys()
    }


    data = {
        k: [float(x) for x in data[k]]
        for k in data.keys()
    }

    print("Found the following models", list(data.keys()))

    data = {
        k: data[k] for k in ALLOWED_MODELS
    }


    yticks = list(data.keys())

    data_baseline = np.array(data["Baseline (52775 ckpt)"])
    data = np.array(list(data.values()))
    data -= data_baseline

    masked_data = np.ma.masked_array(data, np.isnan(data))
    average = np.average(masked_data, axis=1)

    # add average for model
    data = np.vstack((data.T, average, average, average)).T

    # xticks = [
    #     "baseline", "0.5\nthreshold", "0.25\nthreshold", "0.75\nthreshold", "0.25\nmargin", "0.5\nmargin", "1\nmargin"
    # ]
    # yticks = [
    #     "addition", "antonym-replacement", "copy-source", "do-not-translate",
    #     "hallucination", "lexical-overlap", "nonsense", "omission", "ordering-mismatch",
    #     "untranslated"
    # ]

    plt.figure(figsize=(8, 5))
    plt.imshow(data, cmap="RdYlGn", aspect="auto")

    for row_i, row in enumerate(data):
        plt.text(
            data.shape[1]-1.5, row_i,
            "+" if data[row_i][-1] > 0 else "-",
            ha="center", va="center",
        )

    # trained metrics
    plt.vlines(
        data.shape[1]-len(TRAINED_PHENOMENA)-0.5-3,
        -0.5, len(yticks)-0.5,
        color="black"
    )
    # average
    plt.vlines(
        data.shape[1]-3,
        -0.5, len(yticks)-0.5,
        color="black"
    )
    # plt.xlim(0, data.shape[1])
    plt.ylim(data.shape[0]-0.5, -0.5)

    plt.title(
        "difference in kendall $\\tau$ on specific phenomena against the baseline"
    )
    # plt.xticks(range(len(xticks)), xticks)
    plt.yticks(range(len(yticks)), yticks)

    plt.tight_layout(pad=0.5)
    plt.show()
