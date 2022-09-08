#!/usr/bin/env python3

import comet
import argparse
import csv
from scipy.stats import spearmanr, pearsonr, kendalltau
import numpy as np

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model",
        default="downloaded/mtm-baseline-comet/checkpoints/epoch=3-step=42220.ckpt",
    )
    args.add_argument("--data", default="downloaded/wmt-ende-newstest2021.csv")
    args = args.parse_args()

    model = comet.load_from_checkpoint(args.model)

    with open(args.data, "r") as f:
        data = list(csv.DictReader(f))[:10]

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
        batch_size=1,
        gpus=0,
        accelerator="cpu",
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