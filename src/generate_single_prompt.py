# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import argparse
import time
import torch
from utils.inference_pipeline import InferencePipeline, load_and_set_lora_ckpt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="nota-ai/bk-sdm-small")    
    parser.add_argument("--save_dir", type=str, default="./results/bk-sdm-small")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--val_prompt", type=str, default="a photo of sks11df dog in the snow")
    parser.add_argument("--use_dpm_solver", action='store_true', help='use DPMSolverMultistepScheduler')
    parser.add_argument("--is_lora_checkpoint", action='store_true', help='specify whether to use LoRA finetuning')
    parser.add_argument("--lora_weight_path", type=str, default=None, help='dir path including lora.pt and lora_config.json')    
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device)
    pipeline.set_pipe_and_generator()

    if args.use_dpm_solver:    
        print(" ** replace PNDM scheduler into DPM-Solver")
        from diffusers import DPMSolverMultistepScheduler
        pipeline.pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.pipe.scheduler.config)        

    if args.is_lora_checkpoint:
        print(" ** use lora checkpoints")
        load_and_set_lora_ckpt(pipe=pipeline.pipe,
                               weight_path=os.path.join(args.lora_weight_path, 'lora.pt'),
                               config_path=os.path.join(args.lora_weight_path, 'lora_config.json'),
                               dtype=torch.float16)
        
    save_path = os.path.join(args.save_dir, args.val_prompt)
    os.makedirs(save_path, exist_ok=True)

    t0 = time.perf_counter()
    for i in range(args.num_images):
        print(f"Generate {args.val_prompt} --- {i}")
        img = pipeline.generate(prompt = args.val_prompt,
                                n_steps = args.num_inference_steps,
                                img_sz = args.img_sz)[0]
        img.save(os.path.join(save_path, f"{i}.png"))
        img.close()

    pipeline.clear()
    print(f"Save to {save_path}")
    print(f"{(time.perf_counter()-t0):.2f} sec elapsed")
