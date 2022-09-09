#!/usr/bin/bash

for mode in acc corr; do
        # "checkpoints_eloquent/checkpoints_margin_1/checkpoints/epoch=1-step=10092.ckpt" \
        # "checkpoint_margin_01/checkpoints/epoch=3-step=20184.ckpt" \
        # "checkpoint_margin_01_reg/checkpoints/epoch=6-step=35322.ckpt" \
        # "checkpoint_contrastive/checkpoints/epoch=3-step=20184.ckpt"; \
        # "checkpoint_threshold_025/checkpoints/epoch=0-step=14876.ckpt"; \
        # "checkpoint_threshold_075/checkpoints/epoch=0-step=14876.ckpt"; \

    for model in \
        "checkpoint_threshold_dyn/checkpoints/epoch=0-step=5046.ckpt"; \
    do
        basedirectory=$(echo "$model" | awk -F "/" '{print $1}');
        nickname=${basedirectory/checkpoint_/};

        echo "Running $nickname";

        bsub -W 4:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" \
            python3 ./src_comet_eval/eval_${mode}.py \
            -l logs/${mode}_${nickname}.json \
            --model "downloads/${model}";
    done;
done;

# for mode in acc corr; do
#     bsub -W 4:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" \
#         python3 ./src_comet_eval/eval_${mode}.py \
#         -l logs/${mode}_ch025_e3.json \
#         --model "downloads/checkpoints_eloquent/checkpoints_margin_025/checkpoints/epoch=3-step=20184.ckpt";

#     bsub -W 4:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" \
#         python3 ./src_comet_eval/eval_${mode}.py \
#         -l logs/${mode}_ch025_e4.json \
#         --model "downloads/checkpoints_eloquent/checkpoints_margin_025/checkpoints/epoch=4-step=25230.ckpt";
        
#     bsub -W 4:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" \
#         python3 ./src_comet_eval/eval_${mode}.py \
#         -l logs/${mode}_ch05_e0.json \
#         --model "downloads/checkpoints_eloquent/checkpoints_margin_05/checkpoints/epoch=0-step=5046.ckpt";

#     bsub -W 4:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" \
#         python3 ./src_comet_eval/eval_${mode}.py \
#         -l logs/${mode}_ch05_e2.json \
#         --model "downloads/checkpoints_eloquent/checkpoints_margin_05/checkpoints/epoch=2-step=15138.ckpt";
        
#     bsub -W 4:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" \
#         python3 ./src_comet_eval/eval_${mode}.py \
#         -l logs/${mode}_ch1_e0.json \
#         --model "downloads/checkpoints_eloquent/checkpoints_margin_1/checkpoints/epoch=0-step=5046.ckpt";

#     bsub -W 4:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" \
#         python3 ./src_comet_eval/eval_${mode}.py \
#         -l logs/${mode}_ch1_e1.json \
#         --model "downloads/checkpoints_eloquent/checkpoints_margin_1/checkpoints/epoch=1-step=10092.ckpt";
# done;


# MQM correlations

# bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" \
#     python3 ./src_comet_eval/eval_corr.py \
#     -l logs/corr_aug_05.json \
#     --model "downloads/mtm-aug-comet-05/checkpoints/epoch=0-step=14876.ckpt";

# bsub -W 4:00 -n 8 -R "rusage[mem=3000,ngpus_excl_p=1]" \
#     python3 ./src_comet_eval/eval_corr.py \
#     -l logs/corr_baseline.json \
#     --model "downloads/mtm-baseline-comet/checkpoints/epoch=3-step=42220.ckpt";

# challenge set

# bsub -W 4:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" \
#     python3 ./src_comet_eval/eval_acc.py \
#     -l logs/acc_aug_05.json \
#     --model "downloads/mtm-aug-comet-05/checkpoints/epoch=0-step=14876.ckpt";

# bsub -W 4:00 -n 8 -R "rusage[mem=4000,ngpus_excl_p=1]" \
#     python3 ./src_comet_eval/eval_acc.py \
#     -l logs/acc_baseline.json \
#     --model "downloads/mtm-baseline-comet/checkpoints/epoch=3-step=42220.ckpt";