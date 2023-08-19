# ------------------------------------------------------------------------------------
# Copyright (c) 2023 Nota Inc. All Rights Reserved.
# Code modified from https://huggingface.co/blog/stable_diffusion
# ------------------------------------------------------------------------------------

import diffusers
from diffusers import StableDiffusionPipeline
import torch
import gc

diffusers_version = int(diffusers.__version__.split('.')[1])

class InferencePipeline:
    def __init__(self, weight_folder, seed, device):
        self.weight_folder = weight_folder
        self.seed = seed
        self.device = torch.device(device)

        self.pipe = None
        self.generator = None

    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation** 
        if diffusers_version == 15: # for the specified version in requirements.txt
            self.pipe = StableDiffusionPipeline.from_pretrained(self.weight_folder,
                                                                torch_dtype=torch.float16).to(self.device)
            self.pipe.safety_checker = lambda images, clip_input: (images, False) 
        elif diffusers_version >= 19: # for recent diffusers versions
            self.pipe = StableDiffusionPipeline.from_pretrained(self.weight_folder,
                                                                safety_checker=None, torch_dtype=torch.float16).to(self.device)
        else: # for the versions between 0.15 and 0.19, the benchmark scores are not guaranteed.
            raise Exception(f"Use diffusers version as either ==0.15.0 or >=0.19 (from current {diffusers.__version__})")

        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    def generate(self, prompt: str, n_steps: int, img_sz: int):
        out = self.pipe(
            prompt,
            num_inference_steps=n_steps,
            height = img_sz,
            width = img_sz,
            generator=self.generator,
        ) 
        return out.images[0]
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_sdm_params(self):
        params_unet = self._count_params(self.pipe.unet)
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        params_total = params_unet + params_text_enc + params_image_dec
        return f"Total {(params_total/1e6):.1f}M (U-Net {(params_unet/1e6):.1f}M; TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"
