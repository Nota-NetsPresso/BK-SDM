# ------------------------------------------------------------------------------------
# Copyright (c) 2023 Nota Inc. All Rights Reserved.
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
python3 src/generate.py --model_id ./results/kd_bk_small --save_dir $IMG_DIR

# (B) To test with a specific checkpoint (results/kd_bk_small/checkpoint-45000/unet), use:
IMG_DIR=./outputs/kd_bk_small/checkpoint-45000
python3 src/generate.py --unet_path ./results/kd_bk_small/checkpoint-45000 --model_id CompVis/stable-diffusion-v1-4 --save_dir $IMG_DIR
