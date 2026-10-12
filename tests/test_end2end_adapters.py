"""Exercise official inference functions with NumPy tensor/pipe fixtures, no torch."""
import argparse
import ast
import contextlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from end2end import adapters
from end2end.contracts import AOVS, atomic_json, load_maps, parameters, read_json, sha256


def official_functions(filename, names, environment):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile('from __future__ import annotations\n' + '\n'.join(ast.unparse(n) for n in selected),
                 filename, 'exec'), environment)
    return environment


class Tensor:
    def __init__(self, array): self.array = array
    def unsqueeze(self, axis): return Tensor(np.expand_dims(self.array, axis))
    def to(self, device=None, dtype=None):
        return Tensor(self.array.astype(np.float16 if dtype == 'float16' else np.float32))


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.task = Path(self.tmp.name)
        for name in ('inverse', 'forward'): (self.task / name).mkdir()
        Image.new('RGB', (70, 40), (100, 140, 200)).save(self.task / 'input.png')
        self.request = dict(task_id='fixture', backend='real', input_sha256=sha256(self.task / 'input.png'),
                            parameters=parameters('A rainy night.', 123, 7, 5.5, 1.2))

    def tearDown(self): self.tmp.cleanup()

    def inverse_fixture(self):
        calls, seed_calls = [], []
        torch = SimpleNamespace(float16='float16', float32='float32',
            manual_seed=lambda v: seed_calls.append(('torch', v)),
            cuda=SimpleNamespace(manual_seed_all=lambda v: seed_calls.append(('cuda', v)), synchronize=lambda: None))
        module = ModuleType('gradio_inverse_demo')

        class Pipe:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                index = len(calls)
                return ([np.full((512, 512, 3), index / 6, np.float32)],)

        def init_models(args):
            self.assertEqual(args.device, 'cuda')
            self.assertEqual(args.im_size, 512)
            self.assertTrue(args.transformer_ckpt0.endswith('InverseRenderer-512/pytorch_model-00001-of-00002.bin')
                            or args.transformer_ckpt0.endswith('InverseRenderer-512\\pytorch_model-00001-of-00002.bin'))
            fp32 = SimpleNamespace(parameters=lambda: [SimpleNamespace(dtype='float32')])
            module._GLOBAL_BUNDLE = dict(device=SimpleNamespace(type='cuda'), dtype='float16',
                                        pipe=Pipe(), imaa=fp32, dino_model=fp32)

        module.init_models = init_models
        environment = official_functions('gradio_inverse_demo.py', ['_aov_np_to_display_uint8'], {'np': np})
        module._aov_np_to_display_uint8 = environment['_aov_np_to_display_uint8']

        def run_inference(image, steps):
            self.assertEqual(image.size, (70, 40))  # official preprocessing remains inside official run
            pipe = module._GLOBAL_BUNDLE['pipe']
            result = []
            for name in AOVS:
                raw = pipe(aov=[name], num_inference_steps=steps, guidance_scale=0,
                           image_guidance_scale=0, strength=1.0, output_type='np')[0][0]
                result.append((module._aov_np_to_display_uint8(raw), name))
            return result
        module.run_inference = run_inference
        with patch.dict(sys.modules, {'gradio_inverse_demo': module}), patch.object(adapters, 'gpu_preflight', return_value=torch):
            adapters.inverse(self.task, self.request)
        self.assertEqual(seed_calls, [('torch', 123), ('cuda', 123)])
        self.assertEqual([call['aov'][0] for call in calls], list(AOVS))
        self.assertTrue(all(call['num_inference_steps'] == 7 for call in calls))
        _, manifest = load_maps(self.task, self.request)
        self.assertTrue(all(m['dtype'] == 'float32' for m in manifest['maps']))
        return manifest

    def test_inverse_raw_recorder_preserves_official_api_and_seed(self):
        self.inverse_fixture()

    def test_actual_official_forward_function_normalization_four_conditions_and_seed(self):
        manifest = self.inverse_fixture()
        calls = []
        module = ModuleType('gradio_forward_demo')
        class Generator:
            def __init__(self, device): self.device = device
            def manual_seed(self, seed): self.seed = seed; return self
        torch = SimpleNamespace(float16='float16', no_grad=contextlib.nullcontext, Generator=Generator,
                                cuda=SimpleNamespace(synchronize=lambda: None))
        class Compose:
            def __init__(self, functions): self.functions = functions
            def __call__(self, image):
                value = image
                for function in self.functions: value = function(value)
                return value
        transforms = SimpleNamespace(Compose=Compose,
            ToTensor=lambda: lambda image: Tensor(np.asarray(image).astype(np.float32).transpose(2, 0, 1) / 255),
            Normalize=lambda mean, std: lambda value: Tensor((value.array - .5) / .5))
        def pipe(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[np.full((512, 512, 3), .75, np.float32)])
        def init_pipeline(args):
            self.assertEqual(args.im_size, 512)
            module._GLOBAL.update(device=SimpleNamespace(type='cuda'), dtype='float16',
                                  im_size=512, pipe=pipe)
        environment = official_functions('gradio_forward_demo.py', ['load_aov_image', 'run_forward'],
            dict(np=np, Image=Image, torch=torch, transforms=transforms, gr=SimpleNamespace(Error=RuntimeError),
                 _GLOBAL={}, REQUIRED_AOVS=list(AOVS), WEATHER_CONDITIONS={}))
        module._GLOBAL = environment['_GLOBAL']
        module.init_pipeline = init_pipeline
        def run_forward(*args):
            environment['WEATHER_CONDITIONS'] = module.WEATHER_CONDITIONS
            return environment['run_forward'](*args)
        module.run_forward = run_forward
        with patch.dict(sys.modules, {'gradio_forward_demo': module}), patch.object(adapters, 'gpu_preflight', return_value=torch):
            adapters.forward(self.task, self.request)
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertIsNone(call['irradiance'])
        self.assertEqual(call['prompt'], ['A rainy night.'])
        self.assertEqual(call['generator'].seed, 123)
        self.assertEqual(call['num_inference_steps'], 7)
        self.assertEqual(call['guidance_scale'], 5.5)
        self.assertEqual(call['image_guidance_scale'], 1.2)
        self.assertEqual(call['strength'], 1)
        self.assertEqual(call['required_aovs'], list(AOVS))
        for item in manifest['maps'][:4]:
            with Image.open(self.task / item['png']) as image:
                expected = ((np.asarray(image).astype(np.float32).transpose(2, 0, 1)[None] / 255 - .5) / .5).astype(np.float16)
            self.assertTrue(np.array_equal(call[item['name']].array, expected))
            self.assertEqual(call[item['name']].array.shape, (1, 3, 512, 512))
        with Image.open(self.task / 'forward/final.png') as image:
            self.assertEqual(image.getpixel((0, 0)), (191, 191, 191))

    def test_corrupt_maps_rejected_before_forward_gpu_preflight(self):
        self.inverse_fixture()
        (self.task / 'inverse/png/albedo.png').write_bytes(b'wrong')
        with patch.object(adapters, 'gpu_preflight') as gpu:
            with self.assertRaises(ValueError): adapters.forward(self.task, self.request)
            gpu.assert_not_called()


if __name__ == '__main__': unittest.main()
