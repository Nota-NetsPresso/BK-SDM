# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import csv
import os
from PIL import Image

def get_file_list_from_csv(csv_file_path):
    file_list = []
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)        
        next(csv_reader, None) # Skip the header row
        for row in csv_reader: # (row[0], row[1]) = (img name, txt prompt) 
            file_list.append(row)
    return file_list

def change_img_size(input_folder, output_folder, resz=256):
    img_list = sorted([file for file in os.listdir(input_folder) if file.endswith('.jpg')])
    for i, filename in enumerate(img_list):
        img = Image.open(os.path.join(input_folder, filename))
        img.resize((resz, resz)).save(os.path.join(output_folder, filename))
        img.close()
        if i % 2000 == 0:
            print(f"{i}/{len(img_list)} | {filename}: resize to {resz}")

