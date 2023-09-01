# Block-removed Knowledge-distilled Stable Diffusion

This is the official codebase for [**BK-SDM: Architecturally Compressed Stable Diffusion for Efficient Text-to-Image Generation**](https://openreview.net/forum?id=bOVydU0XKC) [[ICCV 2023 Demo Track](https://iccv2023.thecvf.com/)] [[ICML 2023 Workshop on ES-FoMo](https://es-fomo.com/)].


BK-SDMs are lightweight text-to-image (T2I) synthesis models: 
  - Certain residual and attention blocks are eliminated from the U-Net of [SD-v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4).
  - Distillation pretraining is conducted with very limited data, but it (surprisingly) remains effective.

⚡Quick Links: [KD Pretraining](https://github.com/Nota-NetsPresso/BK-SDM#distillation-pretraining) | [Evaluation on MS-COCO](https://github.com/Nota-NetsPresso/BK-SDM#evaluation-on-ms-coco-benchmark) | [DreamBooth Finetuning](https://github.com/Nota-NetsPresso/BK-SDM#dreambooth-finetuning-with-peft) | [Demo](https://github.com/Nota-NetsPresso/BK-SDM#gradio-demo)

## Notice
  - [Aug/23/2023] Release Core ML weights of BK-SDMs for iOS and macOS.
  - [Aug/20/2023] Release finetuning code for personalized T2I.
  - [Aug/14/2023] Release BK-SDM-*-2M models (trained with **10× more data**).
  - [Aug/12/2023] 🎉**Release pretraining code** for general-purpose T2I. 
    - MODEL_CARD.md includes [the process of distillation pretraining](https://github.com/Nota-NetsPresso/BK-SDM/blob/main/MODEL_CARD.md#distillation-pretraining) and [results using various data volumes](https://github.com/Nota-NetsPresso/BK-SDM/blob/main/MODEL_CARD.md#effect-of-different-data-sizes-for-training-bk-sdm-small).
  - [Aug/02/2023] [Segmind](https://www.segmind.com/) introduces [their BK-SDM implementation](https://github.com/segmind/distill-sd), big thanks!
  - [Aug/01/2023] Hugging Face [Spaces of the week 🔥](https://huggingface.co/spaces) introduces [our demo](https://huggingface.co/spaces/nota-ai/compressed-stable-diffusion), many thanks!
 
 
## Model Description
- See [Compression Method in MODEL_CARD.md](https://github.com/Nota-NetsPresso/BK-SDM/blob/main/MODEL_CARD.md#compression-method)
- Available at 🤗Hugging Face Models: BK-SDM-{[Base](https://huggingface.co/nota-ai/bk-sdm-base), [Small](https://huggingface.co/nota-ai/bk-sdm-small), [Tiny](https://huggingface.co/nota-ai/bk-sdm-tiny)} and {[Base-2M](https://huggingface.co/nota-ai/bk-sdm-base-2m), [Small-2M](https://huggingface.co/nota-ai/bk-sdm-small-2m), [Tiny-2M](https://huggingface.co/nota-ai/bk-sdm-tiny-2m)}


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
<summary>An equivalent code (modifying solely the U-Net of SD-v1.4 while preserving its Text Encoder and Image Decoder):</summary>

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


## Distillation Pretraining
Our code was based on [train_text_to_image.py](https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image) of Diffusers `0.15.0.dev0`. To access the latest version, use [this link](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).

#### [Optional] Toy to check runnability
  ```bash
  bash scripts/get_laion_data.sh preprocessed_11k
  bash scripts/kd_train_toy.sh
  ```
- A toy dataset (11K img-txt pairs) is downloaded at `./data/laion_aes/preprocessed_11k` (1.7GB in tar.gz; 1.8GB data folder).
- A toy script can be used to verify the code executability and find the batch size that matches your GPU. With a batch size of `8` (=4×2), training BK-SDM-Base for 20 iterations takes about 5 minutes and 22GB GPU memory.

#### Single-gpu training for BK-SDM-{[Base](https://huggingface.co/nota-ai/bk-sdm-base), [Small](https://huggingface.co/nota-ai/bk-sdm-small), [Tiny](https://huggingface.co/nota-ai/bk-sdm-tiny)}
  ```bash
  bash scripts/get_laion_data.sh preprocessed_212k
  bash scripts/kd_train.sh
  ```
- The dataset with 212K (=0.22M) pairs is downloaded at `./data/laion_aes/preprocessed_212k` (18GB tar.gz; 20GB data folder).
- With a batch size of `256` (=4×64), training BK-SDM-Base for 50K iterations takes about 300 hours and 53GB GPU memory. With a batch size of `64` (=4×16), it takes 60 hours and 28GB GPU memory.
- Training BK-SDM-{Small, Tiny} results in 5∼10% decrease in GPU memory usage.

#### Multi-gpu training
  ```bash
  bash scripts/kd_train_toy_ddp.sh
  ```
- Multi-GPU training is supported (sample results: [link](https://github.com/Nota-NetsPresso/BK-SDM/issues/10#issuecomment-1676038203)), although all experiments for our paper were conducted using a single GPU. Thanks [@youngwanLEE](https://github.com/youngwanLEE) for sharing the script :)

#### [After training] Generation with a trained U-Net
  ```bash
  bash scripts/get_mscoco_files.sh
  bash scripts/generate_with_trained_unet.sh
  ```
- A trained U-Net is used for [Step (2) of the benchmark evaluation](https://github.com/Nota-NetsPresso/BK-SDM#code-using-bk-sdm-small-as-default).
- To test with [a specific checkpoint](https://github.com/Nota-NetsPresso/BK-SDM/blob/60939ebaea65271579df734cb3ad44e1f84ca18f/scripts/generate_with_trained_unet.sh#L26), modify `--unet_path` by referring to [the example directory structure](https://github.com/Nota-NetsPresso/BK-SDM/blob/60939ebaea65271579df734cb3ad44e1f84ca18f/scripts/generate_with_trained_unet.sh#L7-L17).

#### Note on training code
  <details>
  <summary>
  Key segments for KD training
  </summary>

  - Define Student U-Net by adjusting config.json [[link](https://github.com/Nota-NetsPresso/BK-SDM/blob/5fc4a8be8076766d4c123b4916d0404f1f99b57b/src/kd_train_text_to_image.py#L437-L438)]
  - Initialize Student U-Net by copying Teacher U-Net's weights [[link](https://github.com/Nota-NetsPresso/BK-SDM/blob/5fc4a8be8076766d4c123b4916d0404f1f99b57b/src/kd_train_text_to_image.py#L72-L117)]
  - Define hook locations for feature KD [[link](https://github.com/Nota-NetsPresso/BK-SDM/blob/5fc4a8be8076766d4c123b4916d0404f1f99b57b/src/kd_train_text_to_image.py#L693-L715)]
  - Define losses for feature-and-output KD [[link](https://github.com/Nota-NetsPresso/BK-SDM/blob/5fc4a8be8076766d4c123b4916d0404f1f99b57b/src/kd_train_text_to_image.py#L764-L779)]

  </details>

  <details>
  <summary>
  Key learning hyperparams
  </summary>

  ```bash
  --unet_config_name "bk_small" # option: ["bk_base", "bk_small", "bk_tiny"]
  --use_copy_weight_from_teacher # initialize student unet with teacher weights
  --learning_rate 5e-05
  --train_batch_size 64
  --gradient_accumulation_steps 4
  --lambda_sd 1.0
  --lambda_kd_output 1.0
  --lambda_kd_feat 1.0
  ```
  
   </details>


## Evaluation on MS-COCO Benchmark
We used the following codes to obtain the results on MS-COCO. After generating 512×512 images with the PNDM scheduler and 25 denoising steps, we downsampled them to 256×256 for computing scores.

#### Code (using BK-SDM-[Small](https://huggingface.co/nota-ai/bk-sdm-small) as default)
On a single 3090 GPU, '(2)' takes ~10 hours per model, and '(3)' takes a few minutes.

- (1) Download `metadata.csv` and `real_im256.npz`:
    ```bash
    bash scripts/get_mscoco_files.sh

    # ./data/mscoco_val2014_30k/metadata.csv: 30K prompts from the MS-COCO validation set (used in '(2)')  
    # ./data/mscoco_val2014_41k_full/real_im256.npz: FID statistics of 41K real images (used in '(3)')
    ```

  <details>
  <summary>
  Note on 'real_im256.npz'
  </summary>

  * Following the evaluation protocol [[DALL·E](https://arxiv.org/abs/2102.12092), [Imagen](https://arxiv.org/abs/2205.11487)], the FID stat for real images was computed over the full validation set (41K images) of MS-COCO. A precomputed stat file is downloaded via '(1)' at `./data/mscoco_val2014_41k_full/real_im256.npz`.
  * Additionally, `real_im256.npz` can be computed with `python3 src/get_stat_mscoco_val2014.py`, which downloads the whole images, resizes them to 256×256, and computes the FID stat.

  </details>

- (2) Generate 512×512 images over 30K prompts from the MS-COCO validation set → Resize them to 256×256:
    ```bash
    python3 src/generate.py 

    # python3 src/generate.py --model_id nota-ai/bk-sdm-base --save_dir ./results/bk-sdm-base
    # python3 src/generate.py --model_id nota-ai/bk-sdm-tiny --save_dir ./results/bk-sdm-tiny  
    ```

- (3) Compute FID, IS, and CLIP score:
    ```bash
    bash scripts/eval_scores.sh

    # For the other models, modify the `./results/bk-sdm-*` path in the scripts to specify different models.
    ```

#### Results on Zero-shot MS-COCO 256×256 30K
See [Results in MODEL_CARD.md](https://github.com/Nota-NetsPresso/BK-SDM/blob/main/MODEL_CARD.md#results-on-ms-coco-benchmark)


## DreamBooth Finetuning with 🤗[PEFT](https://github.com/huggingface/peft)

Our lightweight SD backbones can be used for efficient personalized generation. [DreamBooth](https://arxiv.org/abs/2208.12242) refines text-to-image diffusion models given a small number of images. DreamBooth+[LoRA](https://arxiv.org/abs/2106.09685) can drastically reduce finetuning cost.

#### DreamBooth dataset 
The dataset is downloaded at `./data/dreambooth/dataset` [[folder tree](https://github.com/google/dreambooth/tree/main/dataset)]: 30 subjects × 25 prompts × 4∼6 images.

  ```bash
  git clone https://github.com/google/dreambooth ./data/dreambooth
  ```

#### DreamBooth finetuning (using BK-SDM-[Base](https://huggingface.co/nota-ai/bk-sdm-base) as default)
Our code was based on [train_dreambooth.py](https://github.com/huggingface/peft/tree/v0.1.0#parameter-efficient-tuning-of-diffusion-models) of PEFT `0.1.0`. To access the latest version, use [this link](https://github.com/huggingface/peft/blob/main/examples/lora_dreambooth/train_dreambooth.py).

- (1) **without LoRA** — full finetuning & used in our paper
  ```bash
  bash scripts/finetune_full.sh # learning rate 1e-6
  bash scripts/generate_after_full_ft.sh
  ```
- (2) **with LoRA** — parameter-efficient finetuning
  ```bash
  bash scripts/finetune_lora.sh # learning rate 1e-4
  bash scripts/generate_after_lora_ft.sh  
  ```
- On a single 3090 GPU, finetuning takes 10~20 minutes per subject.

#### Results of Personalized Generation
See [DreamBooth Results in MODEL_CARD.md](https://github.com/Nota-NetsPresso/BK-SDM/blob/main/MODEL_CARD.md#personalized-generation-full-finetuning)


## Gradio Demo
Check out our [Gradio demo](https://huggingface.co/spaces/nota-ai/compressed-stable-diffusion) and the [codes](https://huggingface.co/spaces/nota-ai/compressed-stable-diffusion/tree/main) (main: app.py)!
    <details>
    <summary>
    [Aug/01/2023] featured in Hugging Face [Spaces of the week 🔥](https://huggingface.co/spaces)
    </summary>
    <img alt="Spaces of the week" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/assets-bk-sdm/screenshot_spaces_of_the_week.png" width="100%">
    </details>
    
## Core ML Weights
For iOS or macOS applications, we have converted our models to Core ML format. They are available at 🤗Hugging Face Models ([nota-ai/coreml-bk-sdm](https://huggingface.co/nota-ai/coreml-bk-sdm)) and can be used with Apple's [Core ML Stable Diffusion library](https://github.com/apple/ml-stable-diffusion).

## License
This project, along with its weights, is subject to the [CreativeML Open RAIL-M license](LICENSE), which aims to mitigate any potential negative effects arising from the use of highly advanced machine learning systems. [A summary of this license](https://huggingface.co/blog/stable_diffusion#license) is as follows.

```
1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content,
2. We claim no rights on the outputs you generate, you are free to use them and are accountable for their use which should not go against the provisions set in the license, and
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users.
```


## Acknowledgments
- [Microsoft for Startups Founders Hub](https://www.microsoft.com/en-us/startups) for generously providing the Azure credits used during pretraining.
- [CompVis](https://github.com/CompVis/latent-diffusion), [Runway](https://runwayml.com/), and [Stability AI](https://stability.ai/) for the pioneering research on Stable Diffusion.
- [LAION](https://laion.ai/), [Diffusers](https://github.com/huggingface/diffusers), [PEFT](https://github.com/huggingface/peft), [DreamBooth](https://dreambooth.github.io/), [Gradio](https://www.gradio.app/), and [Core ML Stable Diffusion](https://github.com/apple/ml-stable-diffusion) for their valuable contributions.


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
@article{kim2023bksdm,
  title={BK-SDM: Architecturally Compressed Stable Diffusion for Efficient Text-to-Image Generation},
  author={Kim, Bo-Kyeong and Song, Hyoung-Kyu and Castells, Thibault and Choi, Shinkook},
  journal={ICML Workshop on Efficient Systems for Foundation Models (ES-FoMo)},
  year={2023},
  url={https://openreview.net/forum?id=bOVydU0XKC}
}
```
