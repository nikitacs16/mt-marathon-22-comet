#!/usr/bin/env python3

import comet
import argparse
import csv
from scipy.stats import pearsonr
import numpy as np
import json

def wmt_kendall_tau(better_scores, worse_scores):
    """ Computes the official WMT19 shared task Kendall correlation score. """
    assert len(better_scores) == len(worse_scores)
    conc, disc = 0, 0
    for b, w in zip(better_scores, worse_scores):
        if b > w:
            conc += 1
        else:
            disc += 1
    return (conc - disc) / (conc + disc)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        default="downloads/mtm-baseline-comet/checkpoints/epoch=3-step=42220.ckpt",
    )
    args.add_argument(
        "--data", default="downloads/edinburgh-zurich-challenge-sets.tsv"
    )
    args.add_argument("-l", "--logfile", default=None)
    args.add_argument("--cpu", action="store_true")
    args.add_argument("--data-n", type=int, default=None)
    args = args.parse_args()

    with open(args.data, "r") as f:
        data = list(csv.DictReader(f, delimiter="\t"))[:args.data_n]

    data = [
        x for x in data
        if all([
            type(x[k]) is str
            for k in ["reference", "source", "good-translation", "incorrect-translation"]
        ])
    ]

    model = comet.load_from_checkpoint(args.model)

    PREDICT_KWARGS = {}
    if args.cpu:
        PREDICT_KWARGS["batch_size"] = 1
        PREDICT_KWARGS["gpus"] = 0
        PREDICT_KWARGS["accelerator"] = "cpu"
    else:
        PREDICT_KWARGS["gpus"] = 1
        PREDICT_KWARGS["accelerator"] = "gpu"

    model.eval()

    data_hp = [
        {
            "src": x["source"],
            "mt": x["good-translation"],
            "ref": x["reference"],
        }
        for x in data
    ]
    data_hm = [
        {
            "src": x["source"],
            "mt": x["incorrect-translation"],
            "ref": x["reference"],
        }
        for x in data
    ]

    print("Running predictions on good")
    scores_hp = model.predict(
        data_hp,
        **PREDICT_KWARGS
    )[0]

    print("Running predictions on incorrect")
    scores_hm = model.predict(
        data_hm,
        **PREDICT_KWARGS
    )[0]

    scores_hm = np.array(scores_hm, dtype=float)
    scores_hp = np.array(scores_hp, dtype=float)

    print("Shapes", scores_hp.shape, scores_hm.shape)

    print("Computing correlations on", len(scores_hp), "sentence scores")
    corr_pearson = pearsonr(scores_hp, scores_hm)[0]
    print(f"Pearson:  {corr_pearson:.2f}")

    acc = np.average([x < y for x, y in zip(scores_hm, scores_hp)])
    print(f"Accuracy: {acc:.2%}")
    avg_diff = np.average([y - x for x, y in zip(scores_hm, scores_hp)])
    print(f"Avg. diff: {avg_diff:.2f}")

    tau = wmt_kendall_tau(scores_hp, scores_hm)
    print(f"Tau: {tau:.2f}")

    outobj = {
        "tau": tau, "pearson (good-bad)": corr_pearson,
        "model": args.model.split("/")[-1],
        "acc": acc, "avg_diff": avg_diff,
        "data": args.data.split("/")[-1].split(".")[0]
    }

    if args.logfile is None:
        print(json.dumps(outobj))
    else:
        with open(args.logfile, "w") as f:
            f.write(json.dumps(outobj) + "\n")
