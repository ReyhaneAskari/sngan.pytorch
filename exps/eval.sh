#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
# python test.py \
# --img_size 32 \
# --model sngan_cifar10 \
# --latent_dim 128 \
# --gf_dim 256 \
# --g_spectral_norm False \
# --load_path pre_trained/sngan_cifar10.pth \
# --exp_name test_sngan_cifar10 \
# --num_eval_imgs 50000

python test.py \
--img_size 32 \
--model sngan_cifar10 \
--latent_dim 128 \
--gf_dim 256 \
--g_spectral_norm False \
--load_path logs/sngan_cifar10_reproduce_2_2020_09_27_00_17_09/Model/checkpoint_best.pth \
--exp_name test_sngan_cifar10 \
--num_eval_imgs 50000