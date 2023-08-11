# ------------------------------------------------------------------------------------
# Copyright (c) 2023 Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import argparse
import time
from utils.inference_pipeline import InferencePipeline
from utils.misc import get_file_list_from_csv, change_img_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-small",
                        help="CompVis/stable-diffusion-v1-4, nota-ai/bk-sdm-base, nota-ai/bk-sdm-small, nota-ai/bk-sdm-small-2m, nota-ai/bk-sdm-tiny")    
    parser.add_argument("--save_dir", type=str, default="./results/bk-sdm-small",
                        help="$save_dir$/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")    
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device)
    pipeline.set_pipe_and_generator()            

    save_dir_im512 = os.path.join(args.save_dir, 'im512')
    os.makedirs(save_dir_im512, exist_ok=True)
    save_dir_im256 = os.path.join(args.save_dir, 'im256')
    os.makedirs(save_dir_im256, exist_ok=True)       

    file_list = get_file_list_from_csv(args.data_list)
    params_str = pipeline.get_sdm_params()
    
    t0 = time.time()
    for i, file_info in enumerate(file_list):
        img_name = file_info[0]
        val_prompt = file_info[1]

        print("---")
        print(f"{i}/{len(file_list)} | {img_name} {val_prompt} | {args.num_inference_steps} steps")
        print(params_str)

        img = pipeline.generate(prompt = val_prompt,
                                n_steps = args.num_inference_steps,
                                img_sz = args.img_sz)
        img.save(os.path.join(save_dir_im512, img_name))
        img.close()

    pipeline.clear()
    
    change_img_size(save_dir_im512, save_dir_im256, args.img_resz)
    print(f"{time.time()-t0} sec elapsed")

