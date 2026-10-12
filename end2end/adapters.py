"""Real adapters import official code only inside disposable GPU workers."""
import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .contracts import AOVS, SIZE, atomic_json, load_maps, save_aov, sha256

ROOT = Path(__file__).resolve().parents[1]


def status(task, phase):
    atomic_json(task / 'stage.json', dict(stage=phase, updated=time.time()))


def result_identity(request):
    return {k: request[k] for k in ('task_id', 'input_sha256', 'backend')}


def gpu_preflight():
    import torch
    from scripts.inverse_minimal_once import assert_exclusive_gpu
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback forbidden.')
    # Also fails closed on nvidia-smi error. Each stage checks after the prior exits.
    assert_exclusive_gpu()
    return torch


def inverse(task, request):
    torch = gpu_preflight()
    import gradio_inverse_demo as official
    p = request['parameters']
    random.seed(p['seed']); np.random.seed(p['seed'])
    torch.manual_seed(p['seed']); torch.cuda.manual_seed_all(p['seed'])
    ckpt = ROOT / 'checkpoints/InverseRenderer-512'
    status(task, 'inverse:loading')
    official.init_models(argparse.Namespace(device='cuda', im_size=SIZE,
        sd3_path=str(ROOT / 'sd3.5_medium'),
        transformer_ckpt0=str(ckpt / 'pytorch_model-00001-of-00002.bin'),
        transformer_ckpt1=str(ckpt / 'pytorch_model-00002-of-00002.bin'),
        imaa_path=str(ckpt / 'imaa.pth'), dino_path=str(ROOT / 'dinov2_base')))
    bundle = official._GLOBAL_BUNDLE
    if bundle['device'].type != 'cuda' or bundle['dtype'] != torch.float16:
        raise RuntimeError('Inverse loader fell back from CUDA FP16.')
    for key in ('imaa', 'dino_model'):
        if any(x.dtype != torch.float32 for x in bundle[key].parameters()):
            raise RuntimeError(key + ' must retain FP32.')
    original = bundle['pipe']
    maps = []

    class Recorder:
        def __getattr__(self, name):
            return getattr(original, name)

        def __call__(self, *args, **kwargs):
            name = AOVS[len(maps)]
            expected = dict(aov=[name], num_inference_steps=p['steps'], guidance_scale=0,
                            image_guidance_scale=0, strength=1.0, output_type='np')
            if any(kwargs.get(k) != v for k, v in expected.items()):
                raise RuntimeError('Official inverse contract changed.')
            status(task, 'inverse:inferring:' + name)
            started = time.monotonic()
            output = original(*args, **kwargs)
            torch.cuda.synchronize()
            item = save_aov(task, name, np.array(output[0][0], copy=True),
                            official._aov_np_to_display_uint8)
            item['seconds'] = time.monotonic() - started
            maps.append(item)
            return output

    bundle['pipe'] = Recorder()
    try:
        status(task, 'inverse:preprocessing')
        with Image.open(task / 'input.png') as image:
            gallery = official.run_inference(image.copy(), p['steps'])
        if len(maps) != 5 or len(gallery) != 5:
            raise RuntimeError('Incomplete inverse output.')
    finally:
        bundle['pipe'] = original
    atomic_json(task / 'inverse/result.json', dict(**result_identity(request), maps=maps,
        contract='float HWC RGB [0,1] -> official uint8 RGB PNG; no normal remapping'))


def forward(task, request):
    images, manifest = load_maps(task, request)
    torch = gpu_preflight()
    import gradio_forward_demo as official
    p = request['parameters']
    ckpt = ROOT / 'checkpoints/ForwardRenderer'
    status(task, 'forward:loading')
    official.init_pipeline(argparse.Namespace(device='cuda', im_size=SIZE,
        sd3_path=str(ROOT / 'sd3.5_medium'), transformer_ckpt=str(ckpt / 'pytorch_model.bin'),
        lora_path=str(ckpt / 'pytorch_lora_weights.safetensors')))
    if official._GLOBAL['device'].type != 'cuda' or official._GLOBAL['dtype'] != torch.float16:
        raise RuntimeError('Forward loader fell back from CUDA FP16.')
    # This global exists only in this disposable worker, never in the UI or another run.
    official.WEATHER_CONDITIONS = {'target': p['prompt']}
    status(task, 'forward:inferring')
    outputs = official.run_forward(*images, p['steps'], p['guidance_scale'],
                                   p['image_guidance_scale'], p['seed'])
    torch.cuda.synchronize()
    if len(outputs) != 1:
        raise RuntimeError('Expected one target weather.')
    array = outputs[0][0]
    if array.shape != (SIZE, SIZE, 3) or array.dtype != np.uint8:
        raise ValueError('Unexpected final image shape/dtype.')
    Image.fromarray(array, 'RGB').save(task / 'forward/final.png')
    save_forward_record(task, request, manifest)


def save_forward_record(task, request, manifest):
    atomic_json(task / 'forward/result.json', dict(**result_identity(request),
        final='forward/final.png', final_sha256=sha256(task / 'forward/final.png'),
        consumed_png_sha256={m['name']: m['png_sha256'] for m in manifest['maps'][:4]},
        irradiance_condition=None, normalization='RGB uint8 / 255 -> (x-0.5)/0.5; NCHW',
        condition_dtype='float32 CPU MOCK' if request['backend'] == 'mock' else 'float16 CUDA'))


def mock(task, request, stage):
    """Childless CPU worker; synthetic images visibly labelled, never real results."""
    status(task, stage + ':MOCK')
    controls = request.get('mock_controls', {})
    time.sleep(controls.get('delay', 0.1))
    if controls.get('fail_stage') == stage:
        raise RuntimeError('MOCK injected ' + stage + ' failure')
    if stage == 'inverse':
        with Image.open(task / 'input.png') as image:
            base = image.convert('RGB').resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        maps = []
        for index, name in enumerate(AOVS):
            image = Image.fromarray(np.roll(np.asarray(base), index * 7, axis=1))
            ImageDraw.Draw(image).text((12, 12), 'MOCK / ' + name, fill='white', stroke_width=2, stroke_fill='black')
            array = np.asarray(image).astype(np.float32) / 255.0
            maps.append(save_aov(task, name, array))
        atomic_json(task / 'inverse/result.json', dict(**result_identity(request), maps=maps, mock=True))
    else:
        images, manifest = load_maps(task, request)
        # Consume all FOUR active conditions with the official channel/range contract.
        conditions = [np.asarray(image).astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1
                      for image in images[:4]]
        array = np.clip((np.mean(conditions, axis=0)[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
        ImageDraw.Draw(image).text((12, 40), 'MOCK / NOT A RESEARCH RESULT', fill='white', stroke_width=2, stroke_fill='black')
        image.save(task / 'forward/final.png')
        save_forward_record(task, request, manifest)
