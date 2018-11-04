#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/MAD_SRGAN/ \
    --summary_dir ./result/MAD_SRGAN/log/ \
    --mode test \
    --is_training False \
    --task MAD_SRGAN \
    --batch_size 16 \
    --input_dir_LR ./data/TESTIMAGES_SAMPLING_PATTERNS_LR/ \
    --input_dir_HR ./data/TESTIMAGES_SAMPLING_PATTERNS_HR/ \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint ./experiment_MAD_SRGAN_VGG54/new/model-200000

