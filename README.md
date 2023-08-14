# Block-removed Knowledge-distilled Stable Diffusion Model (BK-SDM)

This is the official codebase for [**BK-SDM: Architecturally Compressed Stable Diffusion for Efficient Text-to-Image Generation**](https://openreview.net/forum?id=bOVydU0XKC), which has been accepted to [ICCV 2023 Demo Track](https://iccv2023.thecvf.com/) and [ICML 2023 Workshop on ES-FoMo](https://es-fomo.com/).


BK-SDM-{[Base](https://huggingface.co/nota-ai/bk-sdm-base), [Small](https://huggingface.co/nota-ai/bk-sdm-small), [Tiny](https://huggingface.co/nota-ai/bk-sdm-tiny)} are lightweight text-to-image synthesis models, achieved by compressing [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4): 
  - Certain residual and attention blocks are eliminated from the U-Net of SDM-v1.4.
  - Distillation pretraining is conducted with very limited data, but it (surprisingly) remains effective.

## Notice
  - [Aug/13/2023] Support multi-gpu training. 
  - [Aug/12/2023] 🎉Release **our [training code](https://github.com/Nota-NetsPresso/BK-SDM#distillation-pretraining)** and **BK-SDM-[Small-2M](https://huggingface.co/nota-ai/bk-sdm-small-2m)** (trained with 10× more data). 
    - MODEL_CARD.md includes [the process of distillation pretraining](https://github.com/Nota-NetsPresso/BK-SDM/blob/main/MODEL_CARD.md#distillation-pretraining) and [results using various data volumes](https://github.com/Nota-NetsPresso/BK-SDM/blob/main/MODEL_CARD.md#effect-of-different-data-sizes-for-training-bk-sdm-small).
  - [Aug/02/2023] Segmind introduces [their BK-SDM implementation](https://github.com/segmind/distill-sd), big thanks!

 

## Installation
```bash
conda create -n bk-sdm python=3.8
conda activate bk-sdm
git clone https://github.com/Nota-NetsPresso/BK-SDM.git
cd BK-SDM
pip install -r requirements.txt
```

## Minimal Example with 🤗[Diffusers](https://github.com/huggingface/diffusers)

With the default PNDM scheduler and 50 denoising steps:
```python
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("nota-ai/bk-sdm-small", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a golden vase with different flowers"
image = pipe(prompt).images[0]  
    
image.save("example.png")
```
<details>
<summary>An equivalent code, which modifies solely the U-Net of SDM-v1.4 while preserving its Text Encoder and Image Decoder:</summary>

```python
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet = UNet2DConditionModel.from_pretrained("nota-ai/bk-sdm-small", subfolder="unet", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a golden vase with different flowers"
image = pipe(prompt).images[0]  
    
image.save("example.png")
```

</details>


## Evaluation on MS-COCO Benchmark
We used the following codes to obtain the results on MS-COCO. After generating 512×512 images with the PNDM scheduler and 25 denoising steps, we downsampled them to 256×256 for computing scores.

### Code (using [BK-SDM-Small](https://huggingface.co/nota-ai/bk-sdm-small) as default)

(1) Download `metadata.csv` and `real_im256.npz`:
  ```bash
  bash scripts/get_mscoco_files.sh

  # ./data/mscoco_val2014_30k/metadata.csv: 30K prompts from the MS-COCO validation set (used in '(2)')  
  # ./data/mscoco_val2014_41k_full/real_im256.npz: FID statistics of 41K real images (used in '(3)')
  ```

(2) Generate 512×512 images over 30K prompts from the MS-COCO validation set → Resize them to 256×256:
  ```bash
  python3 src/generate.py 

  # For the other models, use the followings:
  # python3 src/generate.py --model_id nota-ai/bk-sdm-base --save_dir ./results/bk-sdm-base
  # python3 src/generate.py --model_id nota-ai/bk-sdm-tiny --save_dir ./results/bk-sdm-tiny  
  ```

(3) Compute FID, IS, and CLIP score:
  ```bash
  bash scripts/eval_scores.sh

  # For the other models, modify the `./results/bk-sdm-*` path in the scripts to specify different models.
  ```
Note
- Following the evaluation protocol [[DALL·E](https://arxiv.org/abs/2102.12092), [Imagen](https://arxiv.org/abs/2205.11487)], the FID stat for real images was computed over the full validation set (41K images) of MS-COCO. A precomputed stat file is downloaded via '(1)' at `./data/mscoco_val2014_41k_full/real_im256.npz`.
  - Additionally, `real_im256.npz` can be computed with `python3 src/get_stat_mscoco_val2014.py`, which downloads the whole images, resizes them to 256×256, and computes the FID stat.
- On a single 3090 GPU, '(2)' takes about 10 hours per model, and '(3)' takes a few minutes.

### Results on Zero-shot MS-COCO 256×256 30K
See [Results in MODEL_CARD.md](https://github.com/Nota-NetsPresso/BK-SDM/blob/main/MODEL_CARD.md#results-on-ms-coco-benchmark)


## Distillation Pretraining
Our training code was based on [train_text_to_image.py](https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image) of Diffusers `0.15.0.dev0`. To access the latest version, please use [this link](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).

### [Optional] Toy to check runnability
  ```bash
  bash scripts/get_laion_data.sh preprocessed_11k
  bash scripts/kd_train_toy.sh
  ```
- A toy dataset (11K img-txt pairs) will be downloaded at `./data/laion_aes/preprocessed_11k` (1.7GB in tar.gz; 1.8GB data folder).
- A toy script can be used to verify the code executability and find the batch size that matches your GPU. With a batch size of `8` (=4×2), training BK-SDM-Base for 20 iterations takes about 5 minutes and 22GB GPU memory.

### Script for BK-SDM-{[Base](https://huggingface.co/nota-ai/bk-sdm-base), [Small](https://huggingface.co/nota-ai/bk-sdm-small), [Tiny](https://huggingface.co/nota-ai/bk-sdm-tiny)}
  ```bash
  bash scripts/get_laion_data.sh preprocessed_212k
  bash scripts/kd_train.sh
  ```
- The dataset with 212K (=0.22M) pairs will be downloaded at `./data/laion_aes/preprocessed_212k` (18GB tar.gz; 20GB data folder).
- With a batch size of `256` (=4×64), training BK-SDM-Base for 50K iterations takes about 300 hours and 53GB GPU memory. With a batch size of `64` (=4×16), it takes 60 hours and 28GB GPU memory.
- Training BK-SDM-{Small, Tiny} results in 5∼10% decrease in GPU memory usage.

### Multi-gpu training
  ```bash
  bash scripts/kd_train_toy_ddp.sh
  ```
- Multi-GPU training is supported (sample results: [link](https://github.com/Nota-NetsPresso/BK-SDM/issues/10#issuecomment-1676038203)), although all experiments for our paper were conducted using a single GPU. Thanks @youngwanLEE for sharing the script :)

### Key segments for KD training
- Define Student U-Net by adjusting config.json [[link](https://github.com/Nota-NetsPresso/BK-SDM/blob/5fc4a8be8076766d4c123b4916d0404f1f99b57b/src/kd_train_text_to_image.py#L437-L438)]
- Initialize Student U-Net by copying Teacher U-Net's weights [[link](https://github.com/Nota-NetsPresso/BK-SDM/blob/5fc4a8be8076766d4c123b4916d0404f1f99b57b/src/kd_train_text_to_image.py#L72-L117)]
- Define hook locations for feature KD [[link](https://github.com/Nota-NetsPresso/BK-SDM/blob/5fc4a8be8076766d4c123b4916d0404f1f99b57b/src/kd_train_text_to_image.py#L693-L715)]
- Define losses for feature-and-output KD [[link](https://github.com/Nota-NetsPresso/BK-SDM/blob/5fc4a8be8076766d4c123b4916d0404f1f99b57b/src/kd_train_text_to_image.py#L764-L779)]

### Key learning hyperparams
  ```
  unet_config_name "bk_small" # option: ["bk_base", "bk_small", "bk_tiny"]
  learning_rate 5e-05
  train_batch_size 64
  gradient_accumulation_steps 4
  lambda_sd 1.0
  lambda_kd_output 1.0
  lambda_kd_feat 1.0
  ```

## Gradio Demo
Check out our [Gradio demo](https://huggingface.co/spaces/nota-ai/compressed-stable-diffusion) and the [codes](https://huggingface.co/spaces/nota-ai/compressed-stable-diffusion/tree/main) (main: app.py)!

## Model Description
See [Compression Method in MODEL_CARD.md](https://github.com/Nota-NetsPresso/BK-SDM/blob/main/MODEL_CARD.md#compression-method)


## License
This project, along with its weights, is subject to the [CreativeML Open RAIL-M license](LICENSE), which aims to mitigate any potential negative effects arising from the use of highly advanced machine learning systems. [A summary of this license](https://huggingface.co/blog/stable_diffusion#license) is as follows.

```
1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content,
2. We claim no rights on the outputs you generate, you are free to use them and are accountable for their use which should not go against the provisions set in the license, and
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users.
```


## Acknowledgments
- We express our gratitude to [Microsoft for Startups Founders Hub](https://www.microsoft.com/en-us/startups) for generously providing the Azure credits used during pretraining.
- We deeply appreciate the pioneering research on Latent/Stable Diffusion conducted by [CompVis](https://github.com/CompVis/latent-diffusion), [Runway](https://runwayml.com/), and [Stability AI](https://stability.ai/).
- Special thanks to the contributors to [LAION](https://laion.ai/), [Diffusers](https://github.com/huggingface/diffusers), and [Gradio](https://www.gradio.app/) for their valuable support.


## Citation
```bibtex
@article{kim2023architectural,
  title={On Architectural Compression of Text-to-Image Diffusion Models},
  author={Kim, Bo-Kyeong and Song, Hyoung-Kyu and Castells, Thibault and Choi, Shinkook},
  journal={arXiv preprint arXiv:2305.15798},
  year={2023},
  url={https://arxiv.org/abs/2305.15798}
}
```
```bibtex
@article{Kim_2023_ICMLW,
  title={BK-SDM: Architecturally Compressed Stable Diffusion for Efficient Text-to-Image Generation},
  author={Kim, Bo-Kyeong and Song, Hyoung-Kyu and Castells, Thibault and Choi, Shinkook},
  journal={ICML Workshop on Efficient Systems for Foundation Models (ES-FoMo)},
  year={2023},
  url={https://openreview.net/forum?id=bOVydU0XKC}
}
```
