# Unified end-to-end Gradio

This fork adds a single-image workflow: upload an original RGB photograph → InverseRenderer-512 → inspect five intrinsic maps → ForwardRenderer → download one target-weather image and its experiment records. The official standalone demos and inference scripts are preserved.

## Start in the verified environment

Use the existing Linux environment and the local weight directories described in the [README](../README.md). The tested UI version is Gradio **5.44.1**; the entry refuses other versions and never installs dependencies automatically.

```bash
# CPU synthetic workflow; no real model is loaded (also the default mode).
python -B -u gradio_end2end_demo.py --mock

# Real Linux/CUDA workflow: explicitly enables model loading when Generate is clicked.
python -B -u gradio_end2end_demo.py --allow-real-models --results-root outputs/end2end
```

Use only one service. Both modes bind **127.0.0.1:7861**, with **share=False**. For a remote Linux host, forward port 7861 in VS Code Remote-SSH's Ports panel and open the displayed local address in the Windows browser. Check port ownership before starting; do not terminate unidentified listeners. The already running service does not need restarting to read this milestone.

The main service owns the UI, task scheduling and files, not resident CUDA models. The page provides an upload preview, a separate snapshot of the task's original image, Generate and Cancel, five named maps, the final image, stage/elapsed-time/errors, parameter JSON and an experiment ZIP. Defaults are `A rainy day.`, seed 42, 50 steps, Forward guidance 6.0 and image guidance 1.5. Steps applies to each of the five inverse predictions and the forward prediction. Inverse guidance remains 0/0 and strength remains 1.0. MOCK is explicit in page labels, images and records.

## Data and stage ordering

`end2end/controller.py` creates a unique task directory. `scripts/end2end_stage.py` runs a disposable supervisor/worker pair for each real stage. Inverse calls the existing official loader and `run_inference`, retaining its 512×512 bilinear preprocessing, DINO/IMAA behavior and attention masks. Diffusion stays CUDA FP16; DINO and IMAA stay FP32.

The recorder saves unmodified floating pipeline outputs before uint8 display conversion. All five arrays must be finite HWC RGB, 512×512×3, in [0,1]. These arrays are already postprocessed: official Albedo/Irradiance gamma correction and Roughness/Metallic channel replication have occurred. They are not pre-gamma linear physical measurements. No extra gamma, normal-sign remapping or RGB/BGR swap is applied.

The same official uint8 PNG conversion is used for visualization and the forward interface. Each raw/PNG pair is checked for exact conversion consistency, hashes, task identity and input identity. Forward receives PIL RGB maps through official `load_aov_image`: 512 LANCZOS, `/255`, normalization to [-1,1], NCHW and CUDA FP16. This retains the official 8-bit input convention instead of introducing an unvalidated float-direct path.

Albedo, Normal, Roughness and Metallic are actual forward conditions. Irradiance is displayed and saved but the pipeline argument remains **None**. The five-AOV list and **96-channel** transformer remain unchanged. A worker-local single weather prompt replaces the standalone multi-weather loop; the official seed/generator handling and inference parameters remain intact.

After the inverse supervisor and worker fully exit and cleanup passes, the page generator yields all five maps **before creating Forward**. The next stage retains those maps on screen and later returns the final image. A task's original, maps, final image and downloads share one manifest; failed/cancelled tasks never borrow another task's results.

## Protection and compatibility

Real workers preserve the established single-attempt policy: available RAM below **4 GiB**, worker RSS at/above **22 GiB**, or a stage exceeding **1200 seconds** terminates that stage. Automatic retries are **zero**. POSIX process groups, Linux subreaper and PID/creation-time tracking clean owned descendants on exit; GPU-query errors are isolated from the RAM/deadline policy loop. The controller adds a bounded outer cleanup window, not extra inference time.

Gradio uses `trigger_mode='once'`, one shared GPU concurrency group and a queue of size one. A process mutex and project-wide OS file lock also protect the worker chain. Each real stage rejects other GPU compute processes and fails closed if CUDA or the GPU query is unavailable. External scripts should still not be launched concurrently with Generate. If cleanup cannot be verified, the service rejects subsequent tasks.

Generate is a streaming generator. Cancel is an unqueued callback; it signals the supervisor and lets the stream report cleanup. The Gradio 5.44.1 unload callback receives an injected `gr.Request` and returns no output. `launch(allowed_paths=[str(controller.root)])` permits results outside the launch working directory while limiting the whitelist to the configured result root. These compatibility fixes were present during the successful real task and HTTP mock validation.

Outputs live below `outputs/end2end/<task>/`: original PNG, request/code/input hashes, raw `.npy`, five PNGs, final PNG, worker/supervisor logs and resource records, experiment JSON and ZIP. Keep this directory, weights, server logs, credentials and private input images outside Git.

## Verified milestone and limits

On 2026-10-12, one real A10 task used the original 1024×1024 `assets/examples/image/1.png`, official 512 preprocessing, `A rainy day.`, seed 42, steps 50, guidance 6.0 and image guidance 1.5. The live Gradio Generate API uploaded the image, returned five real maps before the final image, and exposed working downloads. A Windows browser opened the forwarded UI. The run was not mocked.

| Stage (including loading and cleanup) | Time | Worker RSS sampled peak | GPU device sampled peak | Minimum available RAM sampled |
|---|---:|---:|---:|---:|
| Inverse | 206.751 s | 14.555 GiB | 18501 MiB | 18.688 GiB |
| Forward | 199.362 s | 14.514 GiB | 16659 MiB | 18.766 GiB |

The page completed generation in about 407.6 s; the client observation including transfers was 410.158 s. Both supervisors passed, both workers exited 0, inverse exit preceded forward start, at most one GPU compute process was sampled, and final GPU usage was 0 MiB. Five float32 arrays passed finite/range/identity checks; four consumed-condition hashes matched. No model/environment changes, automatic retry, training or second full inference were performed.

The tested environment passed 21 new lightweight tests plus 17 existing Linux supervision regressions. Live HTTP mock validation covered upload, intermediate streaming, final output, downloads and cancellation without initializing CUDA. Source fingerprints are in [end2end_verified_sources.json](end2end_verified_sources.json); raw evidence and machine-specific deployment reports are deliberately excluded from Git.

This proves one runnable chain, not a benchmark or general quality guarantee. Real-night-road inputs, `A rainy night.`, other resolutions, batches and repeated-run stability are unvalidated. Pipeline floating arrays are not calibrated material measurements. Official DINO signed-input clipping, FP16 and PNG quantization remain unchanged. Sampled resource peaks can miss short spikes. Windows real GPU supervision is not supported. Browser disconnect/SIGTERM cleanup is bounded; machine shutdown or SIGKILL cannot guarantee Python cleanup. See the [proposed night-road test](night_road_test_plan.md) before a new experiment.

## Attribution and model terms

This integration retains upstream authorship and file-level notices and adds no blanket license to upstream material. No repository-wide LICENSE/COPYING/NOTICE is present in the inspected checkout. The [InverseRenderer-512](https://huggingface.co/GilgameshYX/InverseRenderer-512), [ForwardRenderer](https://huggingface.co/GilgameshYX/ForwardRenderer) and [DINOv2](https://huggingface.co/facebook/dinov2-base) model cards list Apache-2.0; [SD3.5 Medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) has separate Stability AI community terms and access conditions. Model-card metadata is not a new repository-wide license. Review the actual terms for the pinned artifacts independently; this commit includes no weights or new redistribution grant.
