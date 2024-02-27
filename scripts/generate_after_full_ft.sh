# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

IDENTIFIER_NAME="sks11df" # must be the unique identifier used in DreamBooth finetuning
SUBJECT_NAME="dog2" # 1st column value in https://github.com/google/dreambooth/blob/main/dataset/prompts_and_classes.txt#L1-L33
CLASS_NAME="dog" # 2nd column value in https://github.com/google/dreambooth/blob/main/dataset/prompts_and_classes.txt#L1-L33

MODEL_DIR="./results_ft/full/$SUBJECT_NAME/bk-sdm-base" # the result folder of finetuned model; please adjust it if needed
IMG_DIR="./outputs_ft/full/$SUBJECT_NAME/bk-sdm-base" # please adjust it if needed

# prompt examples: https://github.com/google/dreambooth/blob/main/dataset/prompts_and_classes.txt#L35-L95
VAL_TEXT="a $IDENTIFIER_NAME $CLASS_NAME in the snow"
# VAL_TEXT="a $IDENTIFIER_NAME $CLASS_NAME in the jungle"
# VAL_TEXT="a $IDENTIFIER_NAME $CLASS_NAME with a city in the background"
# VAL_TEXT="a painting of $IDENTIFIER_NAME $CLASS_NAME in the style of leonardo da vinci"

CUDA_VISIBLE_DEVICES=3 python3 src/generate_single_prompt.py \
    --use_dpm_solver \
    --model_id $MODEL_DIR \
    --save_dir $IMG_DIR \
    --num_images 4 \
    --val_prompt "$VAL_TEXT"
