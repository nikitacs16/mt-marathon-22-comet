#!/usr/bin/bash

bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" \
    python3 ./src_comet_eval/eval_corr.py \
    -l logs/corr_aug_05.json \
    --model "downloads/mtm-aug-comet-05/checkpoints/epoch=0-step=14876.ckpt";

bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" \
    python3 ./src_comet_eval/eval_corr.py \
    -l logs/corr_baseline.json \
    --model "downloads/mtm-baseline-comet/checkpoints/epoch=3-step=42220.ckpt";



bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" \
    python3 ./src_comet_eval/eval_acc.py \
    -l logs/acc_aug_05.json \
    --model "downloads/mtm-aug-comet-05/checkpoints/epoch=0-step=14876.ckpt";

bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" \
    python3 ./src_comet_eval/eval_acc.py \
    -l logs/acc_baseline.json \
    --model "downloads/mtm-baseline-comet/checkpoints/epoch=3-step=42220.ckpt";