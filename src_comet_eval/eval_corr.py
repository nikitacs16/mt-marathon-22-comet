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
    args.add_argument("--data", nargs="+",
    default=[
        "downloads/wmt-ende-newstest2021.csv",
        "downloads/wmt-enru-newstest2021.csv",
        "downloads/wmt-zhen-newstest2021.csv",
        ]
    )
    args.add_argument("-l", "--logfile", default=None)
    args.add_argument("--cpu", action="store_true")
    args.add_argument("--data-n", type=int, default=None)
    args = args.parse_args()

    model = comet.load_from_checkpoint(args.model)

    outobj = {"individual": []}

    for fname in args.data:
        with open(fname, "r") as f:
            data = list(csv.DictReader(f))[:args.data_n]

        PREDICT_KWARGS = {}
        if args.cpu:
            PREDICT_KWARGS["batch_size"] = 1
            PREDICT_KWARGS["gpus"] = 0
            PREDICT_KWARGS["accelerator"] = "cpu"
        else:
            PREDICT_KWARGS["gpus"] = 1
            PREDICT_KWARGS["accelerator"] = "gpu"

        print("Running predictions")
        model.eval()
        scores_pred = model.predict(
            [
                {
                    "src": x["src"],
                    "mt": x["mt"],
                    "ref": x["ref"],
                }
                for x in data
            ],
            **PREDICT_KWARGS
        )[0]

        scores_human = [x["score"] for x in data][:len(scores_pred)]

        scores_pred = np.array(scores_pred, dtype=float)
        scores_human = np.array(scores_human, dtype=float)
        print("Shapes", scores_pred.shape, scores_human.shape)

        print("Computing correlations on", len(scores_human), "sentence scores")
        corr_kendall = kendalltau(scores_pred, scores_human)[0]
        print(f"Kendall:  {corr_kendall:.2f}")
        corr_pearson = pearsonr(scores_pred, scores_human)[0]
        print(f"Pearson:  {corr_pearson:.2f}")
        corr_spearman = spearmanr(scores_pred, scores_human)[0]
        print(f"Spearman: {corr_spearman:.2f}")

        outobj_local = {
            "kendall": corr_kendall, "pearson": corr_pearson, "spearman": corr_spearman,
            "model": args.model.split("/")[-1],
            "data": fname.split("/")[-1].split(".")[0]
        }
        outobj["individual"].append(outobj_local)

    outobj["avg"] = {
        corr_type: np.average([x[corr_type] for x in outobj["individual"]])
        for corr_type in ["kendall", "pearson", "spearman"]
    }

    if args.logfile is None:
        print(json.dumps(outobj))
    else:
        with open(args.logfile, "w") as f:
            f.write(json.dumps(outobj))
