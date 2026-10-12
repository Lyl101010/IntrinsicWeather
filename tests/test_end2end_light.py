"""CPU subprocess mock E2E and contract tests; no CUDA/model/Gradio dependency."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from end2end.contracts import (AOVS, atomic_json, display_uint8, load_maps,
                               parameters, read_json, save_aov, sha256, validate_array)
from end2end.controller import Controller, ProcessLock, BusyError


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.controller = Controller(self.root, poll_seconds=0.02)
        y, x = np.mgrid[:73, :91]
        self.image = Image.fromarray(np.stack((x * 2, y * 3, (x + y) % 256), axis=-1).astype(np.uint8))

    def tearDown(self):
        self.controller.close()
        self.temporary.cleanup()

    def stream(self, **kwargs):
        return self.controller.generate(self.image, 'A rainy day.', 42, 50, 6, 1.5, 'test', **kwargs)

    def collect(self, stream):
        return [copy.deepcopy(event) for event in stream]

    def test_upload_intermediate_update_forward_consumption_and_archive(self):
        stream = self.stream()
        seen = []
        for event in stream:
            seen.append(copy.deepcopy(event))
            if event.stage.startswith('inverse:complete'):
                self.assertTrue(all(Path(p).is_file() for p in event.maps))
                self.assertIsNone(event.final)
                task = self.root / event.task_id
                self.assertFalse((task / 'forward/launcher.log').exists())
                self.assertIsNone(self.controller.active['test'].process)
        event = seen[-1]
        self.assertEqual(event.stage, 'success')
        self.assertTrue(event.terminal)
        task = self.root / event.task_id
        request = read_json(task / 'request.json')
        self.assertTrue(request['mock'])
        with Image.open(event.input) as saved:
            self.assertTrue(np.array_equal(np.asarray(saved), np.asarray(self.image)))
        images, inverse = load_maps(task, request)
        forward = read_json(task / 'forward/result.json')
        self.assertEqual(forward['consumed_png_sha256'],
                         {item['name']: item['png_sha256'] for item in inverse['maps'][:4]})
        self.assertIsNone(forward['irradiance_condition'])
        # Verify the synthesized mock consumes the maps in RGB/NCHW [-1,1] order.
        conditions = [np.asarray(im).astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1
                      for im in images[:4]]
        expected = np.clip((np.mean(conditions, axis=0)[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255).astype(np.uint8)
        with Image.open(event.final) as final:
            self.assertTrue(np.array_equal(np.asarray(final)[100:], expected[100:]))
        self.assertEqual([read_json(task / stage / 'worker.json')['status']
                          for stage in ('inverse', 'forward')], ['success', 'success'])
        with zipfile.ZipFile(event.archive) as bundle:
            names = bundle.namelist()
            self.assertTrue(all('inverse/raw/' + name + '.npy' in names for name in AOVS))
            self.assertIn('forward/final.png', names)
            self.assertIn('request.json', names)
            self.assertIn('experiment.json', names)
        self.assertEqual(self.controller.active, {})

    def test_failures_preserve_partial_results_and_do_not_retry(self):
        for stage in ('inverse', 'forward'):
            with self.subTest(stage=stage):
                events = self.collect(self.stream(mock_controls={'fail_stage': stage}))
                last = events[-1]
                self.assertEqual(last.stage, 'failed')
                self.assertIn('MOCK injected', last.error)
                self.assertIsNone(last.final)
                self.assertEqual(all(last.maps), stage == 'forward')
                task = self.root / last.task_id
                self.assertTrue((task / stage / 'traceback.txt').is_file())
                if stage == 'inverse':
                    self.assertFalse((task / 'forward/launcher.log').exists())
                self.assertEqual(read_json(task / 'experiment.json')['automatic_retries'], 0)
                self.assertEqual(self.controller.active, {})

    def test_cancel_and_stream_close_exit_process_and_release_lock(self):
        for action in ('cancel', 'close'):
            with self.subTest(action=action):
                stream = self.stream(mock_controls={'delay': 5})
                next(stream)
                next(stream)
                process = self.controller.active['test'].process
                self.assertIsNone(process.poll())
                if action == 'cancel':
                    self.controller.cancel('test')
                    self.assertEqual(self.collect(stream)[-1].stage, 'cancelled')
                else:
                    stream.close()
                self.assertIsNotNone(process.poll())
                self.assertEqual(self.controller.active, {})
                self.assertEqual(self.collect(self.stream())[-1].stage, 'success')

    def test_cancel_between_stages_never_starts_forward(self):
        stream = self.stream()
        for event in stream:
            if event.stage.startswith('inverse:complete'):
                task = self.root / event.task_id
                self.controller.cancel('test')
        self.assertFalse((task / 'forward/launcher.log').exists())
        self.assertEqual(event.stage, 'cancelled')

    def test_busy_same_controller_and_second_service(self):
        stream = self.stream(mock_controls={'delay': 5})
        next(stream)
        other = Controller(self.root, poll_seconds=0.02)
        self.assertIn('已有', self.collect(self.stream())[-1].error)
        events = list(other.generate(self.image, 'rain', 42, 1, 6, 1, 'other'))
        self.assertIn('另一个', events[-1].error)
        stream.close()
        other.close()

    def test_timeout_reaps_worker(self):
        self.controller.stage_timeout = 0.08
        events = self.collect(self.stream(mock_controls={'delay': 5}))
        self.assertEqual(events[-1].stage, 'failed')
        self.assertIn('timeout', events[-1].error)
        self.assertEqual(self.controller.active, {})

    def test_invalid_inputs_clear_outputs_and_never_start_worker(self):
        for image, prompt, seed in [(None, 'rain', 42), (self.image, '', 42),
                                    (self.image, 'rain', float('nan'))]:
            events = list(self.controller.generate(image, prompt, seed, 50, 6, 1, 'test'))
            self.assertEqual(events[-1].stage, 'failed')
            self.assertEqual(events[-1].maps, [None] * 5)
            self.assertIsNone(events[-1].input)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_task_identity_hash_raw_png_and_path_checks(self):
        last = self.collect(self.stream())[-1]
        task = self.root / last.task_id
        request = read_json(task / 'request.json')
        bad = dict(request, task_id='different')
        with self.assertRaises(ValueError):
            load_maps(task, bad)
        manifest_path = task / 'inverse/result.json'
        manifest = read_json(manifest_path)
        original_manifest = copy.deepcopy(manifest)
        manifest['maps'][0]['png'] = '../escape.png'
        atomic_json(manifest_path, manifest)
        with self.assertRaises(ValueError):
            load_maps(task, request)
        atomic_json(manifest_path, original_manifest)
        raw = task / original_manifest['maps'][0]['raw']
        raw.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            load_maps(task, request)

    def test_raw_validation_and_no_hidden_nan_or_normal_remapping(self):
        for array in (np.full((512, 512, 3), np.nan, np.float32),
                      np.ones((512, 512, 3), np.uint8), np.full((512, 512, 3), 2, np.float32)):
            with self.assertRaises(ValueError):
                validate_array(array)
        array = np.zeros((512, 512, 3), np.float32)
        array[:, :, 0] = 0.25; array[:, :, 1] = 0.5; array[:, :, 2] = 1
        save_aov(self.root, 'normal', array)
        with Image.open(self.root / 'inverse/png/normal.png') as image:
            self.assertEqual(image.getpixel((0, 0)), (63, 127, 255))

    def test_real_mode_fails_closed_before_model_import(self):
        with self.assertRaises(ValueError):
            Controller(self.root, backend='real')
        self.assertNotIn('torch', sys.modules)


class PageBindingTests(unittest.TestCase):
    def test_page_generator_event_binding_and_intermediate_values(self):
        # Wiring fixture only: never report this as a live Gradio/browser test.
        events, components = [], []

        class Component:
            def __init__(self, *args, **kwargs):
                self.kwargs = kwargs
                components.append(self)
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def click(self, fn, inputs=None, outputs=None, **kwargs):
                events.append(dict(fn=fn, inputs=inputs, outputs=outputs, **kwargs))
            def unload(self, fn): self.unload_fn = fn
            def queue(self, **kwargs): self.queue_kwargs = kwargs

        gr = SimpleNamespace(__version__='5.44.1', Request=type('Request', (), {}),
            themes=SimpleNamespace(Soft=lambda: None), update=lambda **kw: kw)
        for name in ('Blocks', 'Markdown', 'Row', 'Column', 'Image', 'Textbox', 'Number',
                     'Slider', 'Button', 'File'):
            setattr(gr, name, Component)
        import gradio_end2end_demo
        with tempfile.TemporaryDirectory() as folder, patch.dict(sys.modules, {'gradio': gr}):
            controller = Controller(folder, poll_seconds=0.02)
            demo = gradio_end2end_demo.build_demo(controller)
            generate, cancel = events
            self.assertEqual(generate['concurrency_limit'], 1)
            self.assertEqual(generate['trigger_mode'], 'once')
            self.assertFalse(cancel['queue'])
            self.assertEqual(demo.queue_kwargs['max_size'], 1)
            values = list(generate['fn'](Image.new('RGB', (50, 40), 'red'), 'A rainy day.',
                                        42, 1, 6, 1.5, SimpleNamespace(session_hash='page')))
            self.assertTrue(any(all(v[1:6]) and v[6] is None for v in values))
            self.assertEqual(len(values[-1]), len(generate['outputs']))
            self.assertIsNotNone(values[-1][6])
            self.assertIn('MOCK', values[-1][7])
            self.assertEqual(values[-1][-2:], ({'interactive': True}, {'interactive': False}))
            controller.close()

    def test_real_gradio_build_if_preinstalled(self):
        try:
            import gradio
        except ImportError:
            self.skipTest('Gradio not installed; no dependencies changed.')
        if gradio.__version__ != '5.44.1':
            self.skipTest('Installed Gradio is not the targeted 5.44.1.')
        import gradio_end2end_demo
        with tempfile.TemporaryDirectory() as folder:
            controller = Controller(folder)
            demo = gradio_end2end_demo.build_demo(controller)
            self.assertTrue(demo.config['enable_queue'])
            demo.close(); controller.close()


if __name__ == '__main__':
    unittest.main()
