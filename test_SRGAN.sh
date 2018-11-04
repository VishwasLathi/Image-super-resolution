#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/SRGAN/ \
    --summary_dir ./result/SRGAN/log/ \
    --mode test \
    --is_training False \
    --task SRGAN \
    --batch_size 16 \
    --input_dir_LR ./data/TESTIMAGES_SAMPLING_PATTERNS_LR/ \
    --input_dir_HR ./data/TESTIMAGES_SAMPLING_PATTERNS_HR/ \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint ./experiment_SRGAN_VGG54/model-200000

