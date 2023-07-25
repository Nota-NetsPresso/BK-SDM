# Block-removed Knowledge-distilled Stable Diffusion Model (BK-SDM)

This is the official codebase for [BK-SDM: Architecturally Compressed Stable Diffusion for Efficient Text-to-Image Generation](https://openreview.net/forum?id=bOVydU0XKC), which has been accepted to [ICCV 2023 Demo Track](https://iccv2023.thecvf.com/) and [ICML 2023 Workshop on ES-FoMo](https://es-fomo.com/).


BK-SDMs are lightweight text-to-image synthesis models, achieved by compressing [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4): 
  - Certain residual and attention blocks are eliminated from the U-Net of SDM-v1.4.
  - Distillation pretraining is conducted with very limited training data, but it (surprisingly) remains effective.


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
| Model | FID↓ | IS↑ | CLIP Score↑<br>(ViT-g/14) | # Params,<br>U-Net | # Params,<br>Whole SDM |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Stable Diffusion v1.4 | 13.05 | 36.76 | 0.2958 | 0.86B | 1.04B |
| [BK-SDM-Base](https://huggingface.co/nota-ai/bk-sdm-base) (Ours) | 15.76 | 33.79 | 0.2878 | 0.58B | 0.76B |
| [BK-SDM-Small](https://huggingface.co/nota-ai/bk-sdm-small) (Ours) | 16.98 | 31.68 | 0.2677 | 0.49B | 0.66B |
| [BK-SDM-Tiny](https://huggingface.co/nota-ai/bk-sdm-tiny) (Ours) | 17.12 | 30.09 | 0.2653 | 0.33B | 0.50B |


## Gradio Demo
Check out our [Gradio demo](https://huggingface.co/spaces/nota-ai/compressed-stable-diffusion) and the [codes](https://huggingface.co/spaces/nota-ai/compressed-stable-diffusion/tree/main) (main: app.py)!
  - A free CPU is commonly used for the demo to control expenses😅; but we sometimes switch to T4 during special events.


## Model Description

### U-Net Architecture
We removed several residual and attention blocks from the 0.86B-parameter U-Net in the 1.04B-param SDM-v1.4, and our compressed models are summarized as follows.
- [0.76B-param **BK-SDM-Base**](https://huggingface.co/nota-ai/bk-sdm-base) (0.58B-param U-Net): obtained with ① fewer blocks in outer stages.
- [0.66B-param **BK-SDM-Small**](https://huggingface.co/nota-ai/bk-sdm-small) (0.49B-param U-Net): obtained with ① and ② mid-stage removal.
- [0.50B-param **BK-SDM-Tiny**](https://huggingface.co/nota-ai/bk-sdm-tiny) (0.33B-param U-Net): obtained with ①, ②, and ③ further inner-stage removal.

### Distillation Pretraining
The compact U-Net was trained to mimic the behavior of the original U-Net. We leveraged feature-level and output-level distillation, along with the denoising task loss.

<center>
    <img alt="U-Net architectures and KD-based pretraining" img src="https://huggingface.co/spaces/nota-ai/compressed-stable-diffusion/resolve/e6fb31631f0b2948cf6ec54006ea050d6c83e940/docs/fig_model.png" width="100%">
</center>

<br/>

- **Training Data**: 212,776 image-text pairs (i.e., 0.22M pairs) from [LAION-Aesthetics V2 6.5+](https://laion.ai/blog/laion-aesthetics/).
- **Hardware:** A single NVIDIA A100 80GB GPU
- **Gradient Accumulations**: 4
- **Batch:** 256 (=4×64)
- **Optimizer:** AdamW
- **Learning Rate:** a constant learning rate of 5e-5 for 50K-iteration pretraining


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
