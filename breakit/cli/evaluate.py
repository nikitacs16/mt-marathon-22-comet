#!/usr/bin/env python3

from typing import List

from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr

import pandas as pd
import numpy as np

from breakit.utils import read_file


def get_arg_parser() -> ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = ArgumentParser(description='Command for evaluating different metrics.')

    parser.add_argument('-i', '--inputs',
                        type=Path_fr,
                        nargs='+',
                        required=True,
                        help='Input TSV files with sources, translation hypotheses, reference translations and metric scores.')
    return parser


def comp_corr(good_scores: List[float],
              incorrect_scores: List[float]) -> float:
    '''
    Compute correlation as Kendall Tau scores.
    '''
    concordant = np.sum(good_scores > incorrect_scores)
    discordant = np.sum(good_scores <= incorrect_scores)

    tau = (concordant - discordant) / (concordant + discordant)

    return tau


def evaluate() -> None:
    '''
    Evaluate all models in all TSV files and print Kendall Tau score.
    '''
    cfg = get_arg_parser().parse_args()

    for f in cfg.inputs:
        tsv_f = read_file(f)

        metrics = [m.split('-')[0] for m in tsv_f.columns if m.endswith('-good')]

        # Compute Kendall Tau grouped by phenomenon and metric
        print(f'\n\nEvaluating {f}')
        phenomena_dfs = [(l, df) for l, df in tsv_f.groupby(['phenomena'])]
        for label, p in phenomena_dfs:
            print(f'\n{label}\t{len(p)}')
            for m in metrics:
                tau = comp_corr(p[f'{m}-good'], p[f'{m}-bad'])
                print(f'\t{m}\t{tau}')


if __name__ == '__main__':
        evaluate()
