# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import argparse
import zipfile
import subprocess
import time
from utils.misc import change_img_size

ZIP_URL="http://images.cocodataset.org/zips/val2014.zip"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="./data/mscoco_val2014_41k_full")   
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(os.path.join(args.save_dir, 'val2014.zip')):
        t0 = time.time()
        print(f"Download the zip file from {ZIP_URL}")
        command = f"wget {ZIP_URL} -P {args.save_dir}"
        subprocess.call(command, shell=True)
        print(f"** {time.time()-t0} sec elapsed") # 460 sec elapsed

    img_dir = os.path.join(args.save_dir, 'val2014')
    if not os.path.exists(img_dir):
        t0 = time.time()
        print(f"Unzip to {img_dir}")
        with zipfile.ZipFile(os.path.join(args.save_dir,'val2014.zip'), 'r') as f:
            f.extractall(args.save_dir)
        print(f"** {time.time()-t0} sec elapsed") # 12 sec elapsed

    img_list = sorted([file for file in os.listdir(img_dir) if file.endswith('.jpg')])
    if len(img_list) != 40504:
        raise ValueError(f"the number of images {len(img_list)} is something wrong; 40504 is expected")    
    
    output_dir = os.path.join(args.save_dir, 'val2014_im256')
    if not os.path.exists(output_dir):       
        t0 = time.time()
        os.makedirs(output_dir, exist_ok=True)
        print(f"Resize to 256x256: {output_dir}")
        change_img_size(img_dir, output_dir, resz=256)
        print(f"** {time.time()-t0} sec elapsed") # 350 sec elapsed

    output_npz = os.path.join(args.save_dir, 'real_im256.npz')
    command = f"CUDA_VISIBLE_DEVICES=0 python -m pytorch_fid --save-stats {output_dir} {output_npz}"
    subprocess.call(command, shell=True)
    print(f"FID stat for real images: {output_npz}")
          
if __name__ == "__main__":
    main()