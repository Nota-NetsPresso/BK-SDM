# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/peft/tree/v0.1.0#parameter-efficient-tuning-of-diffusion-models
# ------------------------------------------------------------------------------------

MODEL_NAME="CompVis/stable-diffusion-v1-4" # to load text encoder and image decoder
UNET_TYPE="nota-ai/bk-sdm-base" # to load unet

IDENTIFIER_NAME="sks11df" # arbitrary word without meaning
SUBJECT_NAME="dog2" # 1st column value in https://github.com/google/dreambooth/blob/main/dataset/prompts_and_classes.txt#L1-L33
CLASS_NAME="dog" # 2nd column value in https://github.com/google/dreambooth/blob/main/dataset/prompts_and_classes.txt#L1-L33

DATA_DIR="./data/dreambooth/dataset/$SUBJECT_NAME"
CLASS_DIR="./results_ft/cls_img/$CLASS_NAME"
OUTPUT_DIR="./results_ft/full/$SUBJECT_NAME/bk-sdm-base" # please adjust it if needed

StartTime=$(date +%s)

CUDA_VISIBLE_DEVICES=0 accelerate launch src/dreambooth_finetune.py \
--pretrained_model_name_or_path $MODEL_NAME  \
--unet_path $UNET_TYPE \
--instance_data_dir $DATA_DIR \
--class_data_dir $CLASS_DIR \
--output_dir $OUTPUT_DIR \
--with_prior_preservation \
--prior_loss_weight 1.0 \
--instance_prompt "a $IDENTIFIER_NAME $CLASS_NAME" \
--class_prompt "a $CLASS_NAME" \
--validation_prompt "a $IDENTIFIER_NAME $CLASS_NAME in the jungle" \
--resolution 512 \
--train_batch_size 1 \
--lr_scheduler "constant" \
--lr_warmup_steps 0 \
--num_class_images 200 \
--train_text_encoder \
--learning_rate 1e-6 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing \
--max_train_steps 800 \
--seed 1234

EndTime=$(date +%s)
echo "** Finetuning takes $(($EndTime - $StartTime)) seconds."

