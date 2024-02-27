# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/mlfoundations/open_clip/tree/37b729bc69068daa7e860fb7dbcf1ef1d03a4185#usage
# ------------------------------------------------------------------------------------

import os
import argparse
import torch
import open_clip
from PIL import Image
from utils.misc import get_file_list_from_csv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_txt", type=str, default="./results/bk-sdm-small/im256_clip.txt")
    parser.add_argument("--data_list", type=str, default="./data/mscoco_val2014_30k/metadata.csv")   
    parser.add_argument("--img_dir", type=str, default="./results/bk-sdm-small/im256")  
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-g-14',
                                                                 pretrained='laion2b_s34b_b88k',
                                                                 device=args.device)
    tokenizer = open_clip.get_tokenizer('ViT-g-14')

    file_list = get_file_list_from_csv(args.data_list)
    score_arr = []
    for i, file_info in enumerate(file_list):
        img_path = os.path.join(args.img_dir, file_info[0])
        val_prompt = file_info[1]           
        text = tokenizer([val_prompt]).to(args.device)

        image = preprocess(Image.open(img_path)).unsqueeze(0).to(args.device)
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        probs = (text_features.cpu().numpy() @ image_features.cpu().numpy().T)
        score_arr.append(probs[0][0])

        if i % 1000 == 0:
            print(f"{i}/{len(file_list)} | {val_prompt} | probs {probs[0][0]}") 
    
    final_score = sum(score_arr) / len(score_arr)
    with open(args.save_txt, 'w') as f:
        f.write(f"FINAL clip score {final_score}\n")
        f.write(f"-- sum score {sum(score_arr)}\n")
        f.write(f"-- len {len(score_arr)}\n")