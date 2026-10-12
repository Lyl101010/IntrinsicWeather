# Pretrained inference milestone — 2026-10-11

This fork completed two independent official examples on an NVIDIA A10. This is a reproducibility milestone for the tested inference paths, not a new algorithm or a quantitative reproduction claim. The original authors, implementation, and paper attribution remain in the README.

| Run | Load (s) | Inference stage (s) | Supervisor total (s) | Kernel CPU RSS peak (GiB) | Device VRAM sampled peak (MiB) |
| --- | --- | --- | --- | --- | --- |
| ForwardRenderer attempt_03 | 102.538 | 13.353 | 123.375 | 14.591 | 16659 |
| InverseRenderer inverse_attempt_01 | 141.072 | 22.280 | 170.361 | 14.585 | 18501 |

Forward used the official image_11 intrinsic-map example, `A rainy day.`, 512×512, seed 42, 50 steps, guidance 6.0, image guidance 1.5, and CUDA FP16. Inverse used official `assets/examples/image/1.png`, 512×512, seed 42, five sequential AOV calls of 50 steps each, guidance/image guidance 0, strength 1.0, CUDA FP16, and DINO/IMAA FP32. Inverse inference-stage timing includes preparation, conditioning and saving; the forward pipeline-call timing has a narrower scope.

All inverse raw outputs have shape 512×512×3 and dtype float32; NaN and Inf counts are zero, ranges lie in [0,1], and variances are nonzero. PNGs match the official uint8 conversion.

| AOV | Min | Max | Variance | Pipeline call (s) |
| --- | --- | --- | --- | --- |
| albedo | 0 | 1 | 0.0451453443 | 3.671 |
| normal | 0 | 1 | 0.0464943384 | 3.490 |
| roughness | 0 | 0.959960938 | 0.0539097568 | 3.489 |
| metallic | 0 | 1 | 0.0149090947 | 3.497 |
| irradiance | 0.354248047 | 0.952148438 | 0.0183090005 | 3.509 |

Both workers exited 0; supervisors passed with no residual worker, memory guard, deadline violation, CUDA OOM or monitoring error. GPU release to 0 MiB was checked after exit. Generated outputs and full private evidence are deliberately excluded from Git. The result-only viewer is separate from model-loading demos.

## Source and model identity

[pretrained_inference_versions.json](pretrained_inference_versions.json) contains source SHA256s, pinned model revisions, parameters and compact results without credentials, server aliases, process IDs or private machine paths. The base Git commit predates the loader/worker fixes; the SHA256s identify the actually verified uncommitted code. The six loader/worker/supervisor files match the successful run hashes byte for byte. The result viewer also matches the deployed source. Four unchanged pipeline/helper files differ only in Windows CRLF versus remote LF, verified by hashing the local bytes with CRLF replaced by LF. Git may normalize line endings; raw-file SHA256s should always be interpreted with their recorded line-ending convention.

Checkpoint loading retains the 96-channel forward and 32-channel inverse transformer conventions, strict checks and original inference/preprocessing behavior. Verified loaders, workers and supervisors are retained without source changes in this milestone. Only documentation, ignored-file policy, dependency completeness (`psutil`) and publication preparation have changed since the successful runs.

## Regression coverage and limits

The three checked-in regression test files cover tiny/mock loading interfaces, local offline LoRA, shard/index and dtype failures, worker outputs and parameters, GPU-probe failures, resource limits, normal/abnormal exits, and Linux child cleanup. These are CPU/tiny/mock tests; the full-model success came from the two GPU runs. The preparation round performed syntax and Git/publication checks and did not repeat GPU inference, rerun the remote tests, or change the server.

No real-road input experiment, paired reference metric, dataset benchmark, training run, 1024-model test, or joint inverse-to-forward experiment is established. The inverse normal map has noisy foliage/edge regions, metallic has isolated bright fragments, and roughness/irradiance retain some image structure; physically correct decomposition is not established. DINO clipping, FP16 precision, deprecated image-range conventions and dependency warnings remain documented. GPU polling can miss transient peaks; allocator and kernel RSS peaks supplement sampling.

## Attribution

No repository-wide license was found at the inspected original commit; file-level Apache-2.0 notices are retained and no new blanket license is assigned. Model licenses and access conditions must be reviewed independently. This compact milestone adds operational validation and safeguards to the original project; it does not claim authorship of the paper or models.
