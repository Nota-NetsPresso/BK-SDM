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
                        help="CompVis/stable-diffusion-v1-4, nota-ai/bk-sdm-base, nota-ai/bk-sdm-small, nota-ai/bk-sdm-tiny")    
    parser.add_argument("--save_dir", type=str, default="./results/bk-sdm-small",
                        help="$save_dir/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument("--unet_path", type=str, default=None)   
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")    
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device)
    pipeline.set_pipe_and_generator()    

    if args.unet_path is not None: # use a separate trained unet for generation
        if args.model_id != "CompVis/stable-diffusion-v1-4" and not args.model_id.startswith("nota-ai/bk-sdm"):
            raise ValueError("args.model_id must be either 'CompVis/stable-diffusion-v1-4' or 'nota-ai/bk-sdm-*'"+
                             f" for text encoder and image decoder\n  ** current args.model_id: {args.model_id}")
        
        from diffusers import UNet2DConditionModel 
        unet = UNet2DConditionModel.from_pretrained(args.unet_path, subfolder='unet')
        pipeline.pipe.unet = unet.half().to(args.device)
        print(f"** load unet from {args.unet_path}")        

    save_dir_im512 = os.path.join(args.save_dir, 'im512')
    os.makedirs(save_dir_im512, exist_ok=True)
    save_dir_im256 = os.path.join(args.save_dir, 'im256')
    os.makedirs(save_dir_im256, exist_ok=True)       

    file_list = get_file_list_from_csv(args.data_list)
    params_str = pipeline.get_sdm_params()
    
    t0 = time.perf_counter()
    for batch_start in range(0, len(file_list), args.batch_size):
        batch_end = batch_start + args.batch_size
        img_names = [file_info[0] for file_info in file_list[batch_start: batch_end]]
        val_prompts = [file_info[1] for file_info in file_list[batch_start: batch_end]]

        for i, (img_name, val_prompt) in enumerate(zip(img_names, val_prompts)):
            print("---")
            print(f"{batch_start + i}/{len(file_list)} | {img_name} {val_prompt} | {args.num_inference_steps} steps")
            print(params_str)

        imgs = pipeline.generate(prompt = val_prompts,
                                 n_steps = args.num_inference_steps,
                                 img_sz = args.img_sz)

        for img, img_name in zip(imgs, img_names):
            img.save(os.path.join(save_dir_im512, img_name))
            img.close()

    pipeline.clear()
    
    change_img_size(save_dir_im512, save_dir_im256, args.img_resz)
    print(f"{(time.perf_counter()-t0):.2f} sec elapsed")

