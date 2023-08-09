# ------------------------------------------------------------------------------------
# Copyright (c) 2023 Nota Inc. All Rights Reserved.
# Download a subset of LAION-Aesthetics V2 6.5+
#   ref. original data: https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6.5plus
# ------------------------------------------------------------------------------------

# download "laion_aes_11k": 10274 image-text pairs (1.7GB in tar.gz; 1.8GB data folder)
# bash scripts/get_laion_data.sh laion_aes_11k

# download "laion_aes_212k": 212776 image-text pairs (18GB tar.gz; 20GB data folder)
# bash scripts/get_laion_data.sh laion_aes_212k

DATA_TYPE=$1  # preprocessed_212k, preprocessed_11k
FILE_NAME="${DATA_TYPE}.tar.gz"

DATA_DIR="./data/laion_aes/"
FILE_UNZIP_DIR="${DATA_DIR}${DATA_TYPE}"
FILE_PATH="${DATA_DIR}${FILE_NAME}"

# S3_URL="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/data/improved_aesthetics_6.5plus/${FILE_NAME}"

# wget $S3_URL -O $FILE_PATH
tar -xvzf $FILE_PATH -C $DATA_DIR
echo "downloaded to ${FILE_UNZIP_DIR}"