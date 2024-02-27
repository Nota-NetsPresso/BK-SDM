# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Download a subset of LAION-Aesthetics V2 6.5+
#   ref. original data: https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6.5plus
# ------------------------------------------------------------------------------------

# preprocessed_11k: 10,274 image-text pairs (1.7GB in tar.gz; 1.8GB data folder)
# preprocessed_212k: 212,776 image-text pairs (18GB tar.gz; 20GB data folder)
# preprocessed_2256k: 2,256,472 image-text pairs (182GB tar.gz; 204GB data folder)

DATA_TYPE=$1  # {preprocessed_11k, preprocessed_212k, preprocessed_2256k}
FILE_NAME="${DATA_TYPE}.tar.gz"

DATA_DIR="./data/laion_aes/"
FILE_UNZIP_DIR="${DATA_DIR}${DATA_TYPE}"
FILE_PATH="${DATA_DIR}${FILE_NAME}"

if [ "$DATA_TYPE" = "preprocessed_11k" ] || [ "$DATA_TYPE" = "preprocessed_212k" ]; then
    S3_URL="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/data/improved_aesthetics_6.5plus/${FILE_NAME}"
elif [ "$DATA_TYPE" = "preprocessed_2256k" ]; then
    S3_URL="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/data/improved_aesthetics_6.25plus/${FILE_NAME}"
else
    echo "Something wrong in data folder name"
    exit
fi

wget $S3_URL -O $FILE_PATH
tar -xvzf $FILE_PATH -C $DATA_DIR
echo "downloaded to ${FILE_UNZIP_DIR}"