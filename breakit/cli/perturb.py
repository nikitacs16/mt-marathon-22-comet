#!/usr/bin/env python3

import logging

from jsonargparse import ArgumentParser
from jsonargparse.typing import Path_fr

import pandas as pd

from breakit.perturbers import Perturber
from breakit.utils import read_file


def get_arg_parser() -> ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = ArgumentParser(description='Command for creating adversarial translation hypotheses.')

    parser.add_argument('-i', '--inputs',
                        type=Path_fr,
                        nargs='+',
                        required=True,
                        help='Input TSV files with sources, translation hypothesis and reference translations.')
    parser.add_argument('-o', '--output_suffix',
                        type=str,
                        default='perturbed',
                        help='Output suffix to add to processed TSV files.')
    parser.add_argument('-l', '--lang',
                        type=str,
                        required=True,
                        help='Language of translation hypothesis.')
    parser.add_argument('-m', '--methods',
                        type=str,
                        nargs='+',
                        choices=['negation', 'units', 'named-entities', 'numbers', 'dates'],
                        default=None,
                        required=True,
                        help='List of perturbation methods.')

    return parser


def perturb() -> None:
    '''
    Create adversarial translation hypotheses.
    '''
    cfg = get_arg_parser().parse_args()

    for f in cfg.inputs:
        tsv_f = read_file(f)
        logging.info(f'Perturbing {f}')

        # Apply perturbations
        perturber = Perturber.get_perturber(cfg.lang)
        tsv_f_perturbed = perturber(tsv_f, cfg.methods)

        # Write new TSV file
        new_f = f.split('tsv')[0]+cfg.output_suffix+'.tsv'
        tsv_f_perturbed.to_csv(new_f, index=False, sep='\t')


if __name__ == '__main__':
        perturb()
