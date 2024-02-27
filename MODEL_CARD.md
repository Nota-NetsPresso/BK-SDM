# BK-SDM Model Card

## Compression Method

### U-Net Architecture
Our method is directly applicable to all SD-v1 and v2 versions, removing specific residual and attention blocks from the U-Net architecture. For further details, refer to [our ArXiv paper](https://arxiv.org/abs/2305.15798). Below, SD-v1.4 is shown as an example.


- 1.04B-param [SDM-v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) (0.86B-param U-Net): the original source model.
- 0.76B-param [**BK-SDM-Base**](https://huggingface.co/nota-ai/bk-sdm-base) (0.58B-param U-Net): obtained with ① fewer blocks in outer stages.
- 0.66B-param [**BK-SDM-Small**](https://huggingface.co/nota-ai/bk-sdm-small) (0.49B-param U-Net): obtained with ① and ② mid-stage removal.
- 0.50B-param [**BK-SDM-Tiny**](https://huggingface.co/nota-ai/bk-sdm-tiny) (0.33B-param U-Net): obtained with ①, ②, and ③ further inner-stage removal.

<center>
    <img alt="U-Net architectures" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/assets-bk-sdm/fig_arch.png" width="100%">
</center>

### Distillation Pretraining
The compact U-Net was trained to mimic the behavior of the original U-Net. We leveraged feature-level and output-level distillation, along with the denoising task loss.

<center>
    <img alt="KD-based pretraining" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/assets-bk-sdm/fig_kd_bksdm.png" width="100%">
</center>

<br/>

- **Training Data**
    - **BK-SDM**: 212,776 image-text pairs (i.e., 0.22M pairs) from [LAION-Aesthetics V2 6.5+](https://laion.ai/blog/laion-aesthetics/).
    - **BK-SDM-2M**: 2,256,472 image-text pairs (i.e., 2.3M pairs) from [LAION-Aesthetics V2 6.25+](https://laion.ai/blog/laion-aesthetics/).
- **Hardware:** A single NVIDIA A100 80GB GPU
- **Gradient Accumulations**: 4
- **Batch:** 256 (=4×64)
- **Optimizer:** AdamW
- **Learning Rate:** a constant learning rate of 5e-5 for 50K-iteration pretraining


## Results on MS-COCO Benchmark
The following table shows the results on 30K samples from the MS-COCO validation split. After generating 512×512 images with the PNDM scheduler and 25 denoising steps, we downsampled them to 256×256 for evaluating generation scores.

### Zero-shot MS-COCO 256×256 30K

- Our models were drawn at the 50K-th training iteration.

| Model | FID↓ | IS↑ | CLIP Score↑<br>(ViT-g/14) | # Params,<br>U-Net | # Params,<br>Whole SDM |
|---|:---:|:---:|:---:|:---:|:---:|
| [Stable Diffusion v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) | 13.05 | 36.76 | 0.2958 | 0.86B | 1.04B |
| [BK-SDM-Base](https://huggingface.co/nota-ai/bk-sdm-base) (Ours) | 15.76 | 33.79 | 0.2878 | 0.58B | 0.76B |
| [BK-SDM-Base-2M](https://huggingface.co/nota-ai/bk-sdm-base-2m) (Ours) | 14.81 | 34.17 | 0.2883 | 0.58B | 0.76B |
| [BK-SDM-Small](https://huggingface.co/nota-ai/bk-sdm-small) (Ours) | 16.98 | 31.68 | 0.2677 | 0.49B | 0.66B |
| [BK-SDM-Small-2M](https://huggingface.co/nota-ai/bk-sdm-small-2m) (Ours) | 17.05 | 33.10 | 0.2734 | 0.49B | 0.66B |
| [BK-SDM-Tiny](https://huggingface.co/nota-ai/bk-sdm-tiny) (Ours) | 17.12 | 30.09 | 0.2653 | 0.33B | 0.50B |
| [BK-SDM-Tiny-2M](https://huggingface.co/nota-ai/bk-sdm-tiny-2m) (Ours) | 17.53 | 31.32 | 0.2690 | 0.33B | 0.50B |


The following figure depicts synthesized images with some MS-COCO captions.
<center>
    <img alt="Visual results" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/assets-bk-sdm/fig_results.png" width="100%">
</center>

### Effect of Different Data Sizes for Training BK-SDM-Small
Increasing the number of training pairs improves the IS and CLIP scores over training progress. The MS-COCO 256×256 30K benchmark was used for evaluation.

<center>
    <img alt="Training progress with different data sizes" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/assets-bk-sdm/fig_iter_data_size.png" width="100%">
</center>

Furthermore, with the growth in data volume, visual results become more favorable (e.g., better image-text alignment and clear distinction among objects). 

<center>
    <img alt="Visual results with different data sizes" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/assets-bk-sdm/fig_results_data_size.png" width="100%">
</center>

### Additional Visual Examples

<center>
    <img alt="additional visual examples" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/assets-bk-sdm/fig_results_models_2m.png" width="100%">
</center>


## Personalized Generation (Full Finetuning)
To show the applicability of our lightweight SD backbones, we use [DreamBooth](https://arxiv.org/abs/2208.12242) finetuning for personalized generation.
- Each subject is marked as "a [identifier] [class noun]" (e.g., "a [V] dog").
- Our BK-SDMs can synthesize the input subjects in different backgrounds while preserving their appearance.

<center>
    <img alt="dreambooth results" img src="https://netspresso-research-code-release.s3.us-east-2.amazonaws.com/assets-bk-sdm/fig_results_dreambooth.png" width="100%">
</center>
