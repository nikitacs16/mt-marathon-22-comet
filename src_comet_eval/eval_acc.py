#!/usr/bin/env python3

import comet
import argparse
import csv
from scipy.stats import spearmanr, pearsonr, kendalltau
import numpy as np
import json

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

    print("Running predictions on good")
    scores_hp = model.predict(
        [
            {
                "src": x["source"],
                "mt": x["good-translation"],
                "ref": x["reference"],
            }
            for x in data
        ],
        **PREDICT_KWARGS
    )[0]

    print("Running predictions on incorrect")
    scores_hm = model.predict(
        [
            {
                "src": x["source"],
                "mt": x["incorrect-translation"],
                "ref": x["reference"],
            }
            for x in data
        ],
        **PREDICT_KWARGS
    )[0]

    scores_hp = np.array(scores_hp, dtype=float)
    scores_hm = np.array(scores_hm, dtype=float)
    print("Shapes", scores_hp.shape, scores_hm.shape)

    print("Computing correlations on", len(scores_hp), "sentence scores")
    corr_kendall = kendalltau(scores_hp, scores_hm)[0]
    print(f"Kendall:  {corr_kendall:.2f}")
    corr_pearson = pearsonr(scores_hp, scores_hm)[0]
    print(f"Pearson:  {corr_pearson:.2f}")
    corr_spearman = spearmanr(scores_hp, scores_hm)[0]
    print(f"Spearman: {corr_spearman:.2f}")

    acc = np.average([x < y for x, y in zip(scores_hm, scores_hp)])
    print(f"Accuracy: {acc:.2%}")
    avg_diff = np.average([y-x for x, y in zip(scores_hm, scores_hp)])
    print(f"Avg. diff: {avg_diff:.2%}")

    outobj = {
        "kendall": corr_kendall, "pearson": corr_pearson, "spearman": corr_spearman,
        "model": args.model.split("/")[-1],
        "acc": acc, "avg_diff": avg_diff,
        "data": args.data.split("/")[-1].split(".")[0]
    }

    if args.logfile is None:
        print(json.dumps(outobj))
    else:
        with open(args.logfile, "w") as f:
            f.write(json.dumps(outobj))
