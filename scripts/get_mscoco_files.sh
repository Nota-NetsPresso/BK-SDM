# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

# download "real_im256.npz"
S3_URL="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/data/mscoco_val2014_41k_full/real_im256.npz"
FILE_PATH="./data/mscoco_val2014_41k_full/real_im256.npz"
wget $S3_URL -O $FILE_PATH
echo "downloaded to $FILE_PATH"

# download "metadata.csv"
S3_URL="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/data/mscoco_val2014_30k/metadata.csv"
FILE_PATH="./data/mscoco_val2014_30k/metadata.csv"
wget $S3_URL -O $FILE_PATH
echo "downloaded to $FILE_PATH"
