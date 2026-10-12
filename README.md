# [CVPR 2026 Highlight] IntrinsicWeather: Controllable Weather Editing in Intrinsic Space

<p align="center">
  <img src="assets/teaser.png" alt="IntrinsicWeather teaser" width="100%">
</p>

<p align="center">
  <a href="https://yixinzhu042.github.io/IntrinsicWeather/">Project page</a>
  &nbsp;·&nbsp;
  <a href="https://arxiv.org/pdf/2508.06982v6">Paper</a>
  &nbsp;·&nbsp;
  <a href="assets/supp.pdf">Supplementary</a>
</p>

<p align="center">
  <a href="https://yixinzhu042.github.io/">Yixin Zhu</a>,
  <a href="https://github.com/NK-CS-ZZL">Zuo-Liang Zhu</a>,
  <a href="https://www.patternrecognition.asia/jian/">Jian Yang</a>,
  <a href="https://miloshasan.net/">Miloš Hašan</a>,
  <a href="https://csjinxie.github.io/">Jin Xie</a>,
  <a href="https://wangningbei.github.io/">Beibei Wang</a>
</p>

---

## Fork milestone: unified end-to-end Gradio

On 2026-10-12, this fork completed one real **RGB → InverseRenderer-512 → ForwardRenderer** task through Gradio 5.44.1 on an NVIDIA A10. The uploaded original RGB photograph was decomposed into Albedo, Normal, Roughness, Metallic and Irradiance; the five maps were streamed before the single final weather image and downloadable experiment ZIP. This is a verified single-example integration, not a night-road benchmark or general quality guarantee.

Run from the repository root in the existing verified Linux environment with the local weights below. Do not start a second service if port 7861 is already occupied by the current demo.

```bash
# CPU synthetic workflow; explicitly labelled MOCK. This is also the default mode.
python -B -u gradio_end2end_demo.py --mock

# Real mode: GPU loading starts only when Generate is clicked.
python -B -u gradio_end2end_demo.py --allow-real-models --results-root outputs/end2end
```

Both modes listen on **127.0.0.1:7861**, with **share=False**. Forward remote port 7861 through VS Code Remote-SSH's Ports panel, then use its displayed local address in the Windows browser. The page provides one upload, a task-specific original preview, Generate/Cancel, five intrinsic maps, one final image, prompt/seed/steps/guidance controls, stage/elapsed-time/errors and JSON/ZIP downloads. Defaults are `A rainy day.`, seed 42, 50 steps, Forward guidance 6.0 and image guidance 1.5; inverse guidance remains 0/0.

The UI does not keep CUDA models resident. Inverse and Forward use separate supervised worker processes; Inverse must exit and cleanup must pass before Forward is created. Unique task manifests and hashes prevent mixing outputs. Official 512 preprocessing and raw float arrays are retained; official PNG conversion feeds the original Forward loader (RGB, NCHW, [-1,1], CUDA FP16). Albedo/Normal/Roughness/Metallic condition Forward; Irradiance is saved/displayed but remains `None` in its pipeline. The 96-channel transformer is unchanged.

Each stage keeps the 1200 s timeout, 4 GiB available-RAM floor, 22 GiB worker-RSS ceiling, zero automatic retries and owned-process-tree cleanup. A single Gradio concurrency group, repeat-click restriction, process mutex and OS lock serialize GPU tasks. GPU query exceptions are isolated; each real worker rejects other GPU compute processes. Cancel and session disconnect request bounded cleanup. Gradio 5.44.1 `allowed_paths` is limited to the configured result root, and its unload callback returns no outputs.

The successful task used `assets/examples/image/1.png` (original 1024×1024, official 512 preprocessing) and the default rainy-day parameters. The page generated the result in approximately 407.6 s; client observation including transfers was 410.158 s. Inverse/Forward supervision took 206.751/199.362 s, with sampled GPU peaks of 18501/16659 MiB. Both workers exited normally, no stage process remained, and GPU usage returned to 0 MiB. The verified environment passed 21 new lightweight tests and 17 existing Linux supervision regressions; live HTTP mock upload, intermediate streaming, downloads and cancellation also passed. No training or automatic GPU retry was performed.

See [the unified demo guide](docs/end2end_gradio.md), [verified source fingerprints](docs/end2end_verified_sources.json) and [proposed real night-road test](docs/night_road_test_plan.md). Only the rainy-day official example has completed the joint chain; real night-road inputs, `A rainy night.`, batches and repeated-run stability remain unvalidated. Raw arrays are postprocessed image outputs, not calibrated physical properties. Native Windows real GPU supervision is unsupported. Weights, generated outputs, private inputs, server logs and machine-specific reports are excluded from Git.

## Previous fork milestone: separately verified pretrained inference

This fork preserves the [original IntrinsicWeather implementation](https://github.com/YixinZhu042/IntrinsicWeather) and adds memory-conscious loading and supervised single-example inference. On 2026-10-11, both renderers completed real GPU inference separately on Ubuntu 22.04.5, an NVIDIA A10 (23028 MiB VRAM), and approximately 28 GiB system RAM. No training or joint inverse-to-forward experiment was performed at that earlier milestone; the later unified milestone is documented above.

| Module | Successful run | Official example and configuration | Observed result |
| --- | --- | --- | --- |
| ForwardRenderer | `attempt_03` | `image_11` intrinsic maps; 512×512; rainy; seed 42; 50 steps; guidance 6.0 / image guidance 1.5 | One valid rainy PNG; loading 102.538 s; inference 13.353 s; peak GPU sample 16659 MiB |
| InverseRenderer-512 | `inverse_attempt_01` | `assets/examples/image/1.png`; 512×512; seed 42; five AOVs, each 50 steps; guidance 0 / image guidance 0 | Five float32 `.npy` arrays and PNGs; all finite and nonconstant; loading 141.072 s; complete inference stage 22.280 s; peak GPU sample 18501 MiB |

The diffusion models ran in CUDA FP16; inverse DINOv2 and IMAA remained FP32. The loading changes use early FP16, `low_cpu_mem_usage`, memory-mapped `weights_only` checkpoints, strict state-dict/index checks, and prompt release of checkpoint references. The forward LoRA uses an explicit local filename. The paper's inference algorithms and preprocessing remain unchanged. See [the compact milestone record](docs/pretrained_inference_milestone.md) for parameters, verified source hashes, and limits. Generated images, weights, raw logs, deployment backups, and machine-specific handoff documents are kept outside Git.

### Reproduce the verified environment on Linux

The supervised workers use Linux `resource`, POSIX signals/process groups, `/proc`, and `nvidia-smi`. The Windows checkout is used for preparation; native Windows GPU supervision has not been validated. The tested environment is Python 3.12.11, PyTorch 2.7.1+cu128 / torchvision 0.22.1+cu128, diffusers 0.35.1, transformers 4.56.2, accelerate 1.10.1, peft 0.17.1, and Gradio 5.44.1. The observed driver was 580.126.09. The pinned file is a tested top-level dependency snapshot, not a complete transitive lockfile.

```bash
conda create -n intrinsicweather python=3.12.11 -y
conda activate intrinsicweather
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-server-cu128.txt
python -m pip check
```

These are setup instructions for a new environment; an already verified environment does not need reinstallation.

### Obtain the pinned weights separately

Download from the official model repositories into the paths below; weights are not included in this fork. Review each model's own terms and obtain access where required. Authenticate interactively with the Hugging Face CLI if needed; never place tokens in code, commands, or Git. No dataset download is needed for the two official examples.

```bash
hf download stabilityai/stable-diffusion-3.5-medium --revision b940f670f0eda2d07fbb75229e779da1ad11eb80 --local-dir sd3.5_medium --include 'model_index.json' 'scheduler/*' 'tokenizer*/*' 'text_encoder*/*.json' 'text_encoder*/*.safetensors' 'transformer/*.json' 'transformer/*.safetensors' 'vae/*.json' 'vae/*.safetensors'
hf download facebook/dinov2-base --revision f9e44c814b77203eaa57a6bdbbd535f21ede1415 --local-dir dinov2_base --include config.json preprocessor_config.json model.safetensors
hf download GilgameshYX/InverseRenderer-512 --revision 3eb4e9a712721dd4a2ff23070a6b203b4837490d --local-dir checkpoints/InverseRenderer-512 --include pytorch_model-00001-of-00002.bin pytorch_model-00002-of-00002.bin pytorch_model.bin.index.json imaa.pth
hf download GilgameshYX/ForwardRenderer --revision 0c767208f390e71aebfa6fe9f15ac651deeb2a32 --local-dir checkpoints/ForwardRenderer --include pytorch_model.bin pytorch_lora_weights.safetensors
```

The selected files total approximately 42.3 GiB. Downloading and inference are separate stages: all required files must be present before offline inference starts. Use a CLI version supporting `hf download`; the tested Linux Python dependency environment predates the local downloader CLI and is not a guarantee of CLI availability.

### Run one supervised example at a time

Run from the repository root in the tested Linux environment. Use a new, empty output directory for each explicitly intended run. `--confirmed` acknowledges that model/GPU loading will begin. Do not run both commands concurrently or keep a model-loading Gradio demo resident at the same time. The inverse worker rejects other GPU compute processes; the forward worker relies on the operator's GPU-idle precheck.

```bash
nvidia-smi
python -B -u scripts/forward_minimal_supervisor.py --confirmed --output-dir results/forward_example
# Wait for normal exit and confirm the GPU is idle before the separate inverse run.
nvidia-smi
python -B -u scripts/inverse_minimal_supervisor.py --confirmed --output-dir results/inverse_example
```

Each supervisor permits one attempt with zero automatic retries, a 1200 s deadline, termination below 4 GiB available RAM or at/above 22 GiB worker RSS, persistent resource/log records, and cleanup of its worker and tracked children. GPU query timeouts are caught independently of the RAM/deadline loop. A completed run must also have successful worker metrics, complete valid outputs, and no residual worker; do not treat supervisor exit alone as a physical-accuracy assessment.

CPU/tiny/mock regression tests are available in `tests/test_forward_loading_and_supervisor.py`, `tests/test_inverse_loading_light.py`, and `tests/test_inverse_worker_light.py`. On Linux with the tested dependencies, run each test file directly with `CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 python -B tests/<test_file>.py`. These tests exercise tiny fixtures and mocked APIs; they do not load the full pretrained models.

Limits of the earlier separate milestone: it validated two 512×512 official examples, without a joint chain. The later unified rainy-day task adds one successful chain, but there is still no benchmark, training validation, reference-output agreement or real-night-road quality result. Inverse outputs are normalized pipeline image arrays, not calibrated physical measurements. Official DINO signed-input clipping and the deprecated `[-1,1]` image-tensor convention remain unchanged; `torch_dtype`, tokenizer conversion, and slow-image-processor warnings are recorded. FP16 and uint8 visualization can affect precision. Device sampling may miss brief VRAM peaks; RAM/VRAM requirements are measurements for these runs, not guarantees for other inputs or versions.

The remaining sections retain the upstream dataset, training, and interactive-demo instructions. Those workflows are outside this fork's validated milestone.

## News

- **2026-06-29:** Training code is now open-sourced, including WeatherSynthetic latent preparation, inverse-renderer training, and forward-renderer LoRA training.

## TODO

- [x] Release the paper and supplementary material
- [x] Release WeatherSynthetic dataset
- [ ] Release WeatherReal construction pipeline
- [x] Release pretrained checkpoints
- [x] Add Gradio for inference
- [x] Release training code for WeatherSynthetic

---

## Environment

Create a conda env and install dependencies:

```bash
conda create -n IntrinsicWeather python=3.12 -y
conda activate IntrinsicWeather
pip install -r requirements.txt
```

Install PyTorch for your CUDA/CPU from [pytorch.org](https://pytorch.org) if needed.

### Base models

Download **Stable Diffusion 3.5 Medium**:

```bash
hf download stabilityai/stable-diffusion-3.5-medium --local-dir sd3.5_medium
```

Download **DINOv2-base** for IMAA / DINO patch tokens:

```bash
hf download facebook/dinov2-base --local-dir dinov2_base
```

---

## Dataset

### WeatherSynthetic

The synthetic dataset is on Hugging Face: **[GilgameshYX/WeatherSynthetic](https://huggingface.co/datasets/GilgameshYX/WeatherSynthetic)**.

Download:

```bash
hf download --repo-type dataset GilgameshYX/WeatherSynthetic --local-dir WeatherSynthetic_dataset
```

Expected layout:

```text
WeatherSynthetic_dataset/
├── scene.txt              # one scene name per line
├── prompt.json            # image_path → prompt, optional for some workflows
├── Modern_city/
│   ├── image/{weather}/   # {id}_image.exr, {id}_irradiance.exr
│   └── property/          # albedo, normal, roughness, metallic
├── Small_city/
└── ...
```

Weather types include `sunny`, `rainy`, `foggy`, `snowy`, `overcast`, `night`, `early_morning`, `rain_storm`, `sand_storm`, and others.

Visualize one sample:

```bash
python -m data.WeatherSynthetic \
  --root WeatherSynthetic_dataset \
  --output weather_synthetic_vis.png
```

---

## Training code

The training pipeline uses the released WeatherSynthetic dataset via an explicit latent-cache step.

### 1. Precompute VAE/DINO latents

```bash
python scripts/prepare_weather_synthetic_latents.py \
  --dataset_root WeatherSynthetic_dataset \
  --scene_list_file scene.txt \
  --sd3_path sd3.5_medium \
  --dino_path dinov2_base \
  --output_dir latent/weatherSynthetic \
  --resolution 512 \
  --map_latent_scaling unscaled \
  --image_latent_scaling scaled \
  --encode_prompts
```

Useful options:

- `--skip_dino`: skip `patch_tokens`; this is okay for forward-renderer LoRA training but **not** for inverse training.
- `--encode_prompts`: saves SD3 text embeddings for forward training. If omitted, `training/train_forward.py` / `training/train_forward_lora.py` compute frozen prompt embeddings online.
- `--max_samples N`: quick debug run.
- `--overwrite`: regenerate existing HDF5 files.

### Latent scaling convention

Keep the default convention unless you intentionally modify all related code:

- `latent_image` is **scaled** by `vae.config.scaling_factor`.
- intrinsic maps (`latent_albedo`, `latent_normal`, `latent_roughness`, `latent_metallic`, `latent_irradiance`) are **unscaled** by default.

The training and inference code follow this convention consistently: render-image targets use SD latent scaling, while intrinsic-map latents are stored without multiplying by the VAE scaling factor. The inference pipelines also leave conditioning image/map latents unscaled and divide generated latents by `vae.config.scaling_factor` only before VAE decoding.

### 2. Train inverse renderer

```bash
LATENT_DIR=latent/weatherSynthetic \
MODEL_NAME=sd3.5_medium \
NUM_PROCESSES=1 \
bash training/train_inverse.sh
```

Main script:

```bash
accelerate launch training/train_inverse.py \
  --pretrained_model_name_or_path sd3.5_medium \
  --latent_data_dir latent/weatherSynthetic \
  --output_dir checkpoints/inverse_weather_synthetic \
  --mixed_precision bf16 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --gradient_checkpointing
```

### 3. Train forward renderer

```bash
LATENT_DIR=latent/weatherSynthetic \
MODEL_NAME=sd3.5_medium \
NUM_PROCESSES=1 \
bash training/train_forward.sh
```

Main script:

```bash
accelerate launch training/train_forward.py \
  --pretrained_model_name_or_path sd3.5_medium \
  --latent_data_dir latent/weatherSynthetic \
  --output_dir checkpoints/forward_weather_synthetic \
  --mixed_precision fp16 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --conditioning_dropout_prob 0.05 \
  --num_train_epochs 10 \
  --gradient_checkpointing
```

### 4. Train forward renderer LoRA

```bash
LATENT_DIR=latent/weatherSynthetic \
MODEL_NAME=sd3.5_medium \
NUM_PROCESSES=1 \
bash training/train_forward_lora.sh
```

Main script:

```bash
accelerate launch training/train_forward_lora.py \
  --pretrained_model_name_or_path sd3.5_medium \
  --latent_data_dir latent/weatherSynthetic \
  --output_dir checkpoints/forward_weather_synthetic_lora \
  --mixed_precision fp16 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate_lora 1e-4 \
  --lora_rank 32 \
  --conditioning_dropout_prob 0.05 \
  --num_train_epochs 10 \
  --gradient_checkpointing
```

See `training/README.md` for a shorter training-only checklist.

---

## Inverse rendering

Checkpoints are on Hugging Face:

| Model | Link |
|--------|------|
| InverseRenderer-512 | [GilgameshYX/InverseRenderer-512](https://huggingface.co/GilgameshYX/InverseRenderer-512) |
| InverseRenderer-1024 | [GilgameshYX/InverseRenderer-1024](https://huggingface.co/GilgameshYX/InverseRenderer-1024) |

Download:

```bash
hf download GilgameshYX/InverseRenderer-512 --local-dir checkpoints/InverseRenderer-512
hf download GilgameshYX/InverseRenderer-1024 --local-dir checkpoints/InverseRenderer-1024
```

### Benchmark script

Run from the repository root:

```bash
python test_inverse.py --output_dir inverse_output
```

### Gradio demo

```bash
python gradio_inverse_demo.py \
  --sd3_path sd3.5_medium \
  --transformer_ckpt0 checkpoints/InverseRenderer-512/pytorch_model-00001-of-00002.bin \
  --transformer_ckpt1 checkpoints/InverseRenderer-512/pytorch_model-00002-of-00002.bin \
  --imaa_path checkpoints/InverseRenderer-512/imaa.pth \
  --dino_path dinov2_base \
  --im_size 512 \
  --device cuda \
  --port 7860
```

---

## Forward rendering

Checkpoints are on Hugging Face:

| Model | Link |
|--------|------|
| ForwardRenderer | [GilgameshYX/ForwardRenderer](https://huggingface.co/GilgameshYX/ForwardRenderer) |

Download:

```bash
hf download GilgameshYX/ForwardRenderer --local-dir checkpoints/ForwardRenderer
```

### Gradio demo

```bash
python gradio_forward_demo.py \
  --sd3_path sd3.5_medium \
  --transformer_ckpt checkpoints/ForwardRenderer/pytorch_model.bin \
  --lora_path checkpoints/ForwardRenderer/pytorch_lora_weights.safetensors \
  --maps_dir assets/examples/intrinsic_maps \
  --device cuda \
  --port 7861
```

---

## Repository layout

```text
scripts/prepare_weather_synthetic_latents.py  # EXR → VAE/DINO HDF5 cache
training/train_inverse.py                     # inverse-renderer trainer
training/train_forward.py                     # forward-renderer trainer
training/train_forward_lora.py                # forward-renderer LoRA trainer
data/WeatherSynthetic.py                      # released EXR dataset loader
data/latent_datasets.py                       # HDF5 latent-cache dataset
```

Use `training/` for the WeatherSynthetic training workflow.

---

## Acknowledgements

We thank the authors of the following projects:

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [RGB↔X](https://github.com/zheng95z/rgbx)
- [DiffusionRenderer (origin)](https://github.com/nv-tlabs/diffusion-renderer), [DiffusionRenderer (cosmos)](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)

## Attribution and license status

The paper, authors, project page, acknowledgements, and file-level copyright/license notices are retained from the original repository. At the inspected upstream commit `94c4ae3439b4b1146b6f4524073dda2697f3fb18`, no repository-wide LICENSE, COPYING, or NOTICE file was present. Some files contain Apache-2.0 notices; these do not establish a license for the entire repository or for the model weights. This fork does not assign a new license to upstream material. Check upstream permission and the separate model terms before redistribution or commercial use.

The [InverseRenderer-512](https://huggingface.co/GilgameshYX/InverseRenderer-512), [ForwardRenderer](https://huggingface.co/GilgameshYX/ForwardRenderer) and [DINOv2-base](https://huggingface.co/facebook/dinov2-base) model cards list Apache-2.0. [SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) has separate Stability AI community terms and gated access conditions. This metadata does not grant a new repository-wide license or replace the actual terms for each pinned artifact. The unified-demo commit includes no weights.

The original project provides this citation in `docs/index.html`:

```bibtex
@InProceedings{Zhu_2026_CVPR,
    author    = {Zhu, Yixin and Zhu, Zuo-Liang and Yang, Jian and Ha\v{s}an, Milo\v{s} and Xie, Jin and Wang, Beibei},
    title     = {IntrinsicWeather: Controllable Weather Editing in Intrinsic Space},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {30772-30781}
}
```
