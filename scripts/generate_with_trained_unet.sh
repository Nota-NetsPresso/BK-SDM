# ------------------------------------------------------------------------------------
# Copyright 2023–2024 Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

: <<'END'
Assuming the below directory structure from training (scripts/kd_train.sh):
    ./results/kd_bk_small
    |-- checkpoint-40000
    |    |-- unet
    |    |-- unet_ema
    |-- checkpoint-45000
    |    |-- unet
    |    |-- unet_ema
    |-- ...
    |-- text_encoder
    |-- unet
    |-- vae
END

# (A) To test with the lastest checkpoint (results/kd_bk_small/unet), use:
IMG_DIR=./outputs/kd_bk_small/latest
python3 src/generate.py --model_id ./results/kd_bk_small \
    --save_dir $IMG_DIR --img_sz 512

# (B) To test with a specific checkpoint (results/kd_bk_small/checkpoint-45000/unet), use:
IMG_DIR=./outputs/kd_bk_small/checkpoint-45000
python3 src/generate.py \
    --unet_path ./results/kd_bk_small/checkpoint-45000 --model_id CompVis/stable-diffusion-v1-4 \
    --save_dir $IMG_DIR --img_sz 512

# (C) Examples for SD-v2:
IMG_DIR=./outputs/v2_kd_bk_small/checkpoint-2
python3 src/generate.py \
    --unet_path ./results/v2_kd_bk_small/checkpoint-2 --model_id stabilityai/stable-diffusion-2-1 \
    --save_dir $IMG_DIR --img_sz 768

IMG_DIR=./outputs/v2-base_kd_bk_small/checkpoint-2
python3 src/generate.py \
    --unet_path ./results/v2-base_kd_bk_small/checkpoint-2 --model_id stabilityai/stable-diffusion-2-1-base \
    --save_dir $IMG_DIR --img_sz 512
