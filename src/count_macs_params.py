# ------------------------------------------------------------------------------------
# Copyright 2024. Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import torch
from diffusers import StableDiffusionPipeline
from thop import profile

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def get_macs_params(model_id, img_size=512, txt_emb_size=768, device="cuda", batch_size=1):
    pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    text_encoder = pipeline.text_encoder
    unet = pipeline.unet
    vae_decoder = pipeline.vae.decoder

    # text encoder    
    dummy_input_ids = torch.zeros(batch_size, 77).long().to(device)
    macs_txt_enc, _ = profile(text_encoder, inputs=(dummy_input_ids,))
    macs_txt_enc = macs_txt_enc/batch_size
    params_txt_enc = count_params(text_encoder)

    # unet
    dummy_noisy_latents = torch.zeros(batch_size, 4, int(img_size/8), int(img_size/8)).to(device)
    dummy_timesteps = torch.zeros(batch_size).to(device)
    dummy_text_emb = torch.zeros(batch_size, 77, txt_emb_size).to(device)
    macs_unet, _ = profile(unet, inputs= (dummy_noisy_latents, dummy_timesteps, dummy_text_emb))
    macs_unet = macs_unet/batch_size
    params_unet = count_params(unet)

    # image decoder
    dummy_latents = torch.zeros(batch_size, 4, 64, 64).to(device)
    macs_img_dec, _ = profile(vae_decoder, inputs= (dummy_latents,))
    macs_img_dec = macs_img_dec/batch_size
    params_img_dec = count_params(vae_decoder)
    
    # total
    macs_total = macs_txt_enc+macs_unet+macs_img_dec
    params_total = params_txt_enc+params_unet+params_img_dec
    
    # print
    print(f"== {model_id} | {img_size}x{img_size} img generation ==")
    print(f"  [Text Enc] MACs: {(macs_txt_enc/1e9):.1f}G = {int(macs_txt_enc)}")
    print(f"  [Text Enc] Params: {(params_txt_enc/1e6):.1f}M = {int(params_txt_enc)}")
    print(f"  [U-Net] MACs: {(macs_unet/1e9):.1f}G = {int(macs_unet)}")
    print(f"  [U-Net] Params: {(params_unet/1e6):.1f}M = {int(params_unet)}")
    print(f"  [Img Dec] MACs: {(macs_img_dec/1e9):.1f}G = {int(macs_img_dec)}")
    print(f"  [Img Dec] Params: {(params_img_dec/1e6):.1f}M = {int(params_img_dec)}")    
    print(f"  [Total] MACs: {(macs_total/1e9):.1f}G = {int(macs_total)}")
    print(f"  [Total] Params: {(params_total/1e6):.1f}M = {int(params_total)}")

if __name__ == "__main__":    
    device="cuda:5"
    get_macs_params(model_id="CompVis/stable-diffusion-v1-4", img_size=512, txt_emb_size=768, device=device)
    get_macs_params(model_id="nota-ai/bk-sdm-base", img_size=512, txt_emb_size=768, device=device)
    get_macs_params(model_id="nota-ai/bk-sdm-small", img_size=512, txt_emb_size=768, device=device)
    get_macs_params(model_id="nota-ai/bk-sdm-tiny", img_size=512, txt_emb_size=768, device=device)    
    get_macs_params(model_id="runwayml/stable-diffusion-v1-5", img_size=512, txt_emb_size=768, device=device)
    get_macs_params(model_id="stabilityai/stable-diffusion-2-1-base", img_size=512, txt_emb_size=1024, device=device)
    get_macs_params(model_id="stabilityai/stable-diffusion-2-1", img_size=768, txt_emb_size=1024, device=device)

