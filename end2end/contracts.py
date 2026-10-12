"""Validated on-disk RGB/AOV contract. No CUDA or Gradio dependency."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

AOVS = ('albedo', 'normal', 'roughness', 'metallic', 'irradiance')
SIZE = 512


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters(prompt, seed, steps, guidance, image_guidance):
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1000:
        raise ValueError('天气 prompt 必须为 1–1000 个字符。')
    values = [float(x) for x in (seed, steps, guidance, image_guidance)]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('参数必须为有限数值。')
    seed, steps, guidance, image_guidance = values
    if seed != int(seed) or not 0 <= seed <= 2**31 - 1:
        raise ValueError('seed 必须是 0–2147483647 的整数。')
    if steps != int(steps) or not 1 <= steps <= 100:
        raise ValueError('steps 必须是 1–100 的整数。')
    if not 0 <= guidance <= 15 or not 0 <= image_guidance <= 5:
        raise ValueError('guidance_scale 范围为 0–15，image_guidance_scale 为 0–5。')
    return dict(prompt=prompt.strip(), seed=int(seed), steps=int(steps),
                guidance_scale=guidance, image_guidance_scale=image_guidance,
                im_size=SIZE, inverse_guidance_scale=0, inverse_image_guidance_scale=0,
                strength=1.0, irradiance_condition=None)


def display_uint8(array):
    # Exactly the official _aov_np_to_display_uint8 on validated float [0,1].
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def validate_array(array):
    if array.shape != (SIZE, SIZE, 3) or not np.issubdtype(array.dtype, np.floating):
        raise ValueError('AOV must be floating HWC RGB, 512×512×3.')
    if not np.isfinite(array).all() or array.min() < -1e-6 or array.max() > 1 + 1e-6:
        raise ValueError('AOV must be finite in [0,1].')


def save_aov(task, name, array, converter=display_uint8):
    if name not in AOVS:
        raise ValueError('Unknown AOV.')
    raw, png = task / 'inverse' / 'raw' / (name + '.npy'), task / 'inverse' / 'png' / (name + '.png')
    raw.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    # Preserve unmodified raw values even if validation subsequently fails.
    with raw.open('xb') as stream:
        np.save(stream, array, allow_pickle=False)
    validate_array(array)
    Image.fromarray(converter(array), 'RGB').save(png)
    return dict(name=name, raw=str(raw.relative_to(task)), png=str(png.relative_to(task)),
                raw_sha256=sha256(raw), png_sha256=sha256(png), shape=list(array.shape),
                dtype=str(array.dtype), min=float(array.min()), max=float(array.max()),
                variance=float(np.var(array, dtype=np.float64)),
                encoding='official postprocessed float output (not pre-gamma linear values)',
                official_gamma_correction=name in ('albedo', 'irradiance'),
                official_scalar_repeated_to_rgb=name in ('roughness', 'metallic'))


def checked_path(task, relative):
    path = (task / relative).resolve()
    if not path.is_relative_to(task.resolve()):
        raise ValueError('Artifact escapes task directory.')
    return path


def load_maps(task, request):
    manifest = read_json(task / 'inverse' / 'result.json')
    if any(manifest.get(k) != request[k] for k in ('task_id', 'input_sha256', 'backend')):
        raise ValueError('Inverse result belongs to another task/input/backend.')
    if [item['name'] for item in manifest['maps']] != list(AOVS):
        raise ValueError('Incomplete or reordered five-AOV manifest.')
    images = []
    for item in manifest['maps']:
        raw, png = [checked_path(task, item[k]) for k in ('raw', 'png')]
        if sha256(raw) != item['raw_sha256'] or sha256(png) != item['png_sha256']:
            raise ValueError('AOV hash mismatch.')
        array = np.load(raw, allow_pickle=False)
        validate_array(array)
        with Image.open(png) as image:
            if image.mode != 'RGB' or image.size != (SIZE, SIZE):
                raise ValueError('AOV PNG must be 512×512 RGB.')
            if not np.array_equal(np.asarray(image), display_uint8(array)):
                raise ValueError('AOV PNG does not correspond to raw float array.')
            images.append(image.copy())
    return images, manifest


def verify_input(task, request):
    if sha256(task / 'input.png') != request['input_sha256']:
        raise ValueError('Input hash mismatch.')
