"""CPU-only tests: tiny weights/mocked loader and disposable process trees."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import psutil
import torch
import gradio_forward_demo as demo
import forward_minimal_once
import forward_result_viewer
import forward_minimal_supervisor as supervisor
from diffusers import SD3Transformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler


class LoadingTests(unittest.TestCase):
    def test_real_tiny_model_and_pipeline_api(self):
        with tempfile.TemporaryDirectory() as folder:
            model = SD3Transformer2DModel(sample_size=4, patch_size=2, in_channels=16,
                out_channels=16, num_layers=1, attention_head_dim=8, num_attention_heads=1,
                joint_attention_dim=8, caption_projection_dim=8, pooled_projection_dim=4,
                pos_embed_max_size=4)
            model.save_pretrained(Path(folder) / 'transformer')
            loaded = SD3Transformer2DModel.from_pretrained(folder, subfolder='transformer',
                torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map=None,
                local_files_only=True)
            self.assertTrue(all(p.dtype == torch.float16 for p in loaded.parameters()))
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v.half(), loaded.state_dict()[k]))
            vae = AutoencoderKL(block_out_channels=(32,),
                down_block_types=('DownEncoderBlock2D',), up_block_types=('UpDecoderBlock2D',),
                layers_per_block=1, latent_channels=16, norm_num_groups=8, sample_size=4)
            pipe = demo.StableDiffusion3InstructPix2PixPipeline(transformer=model, vae=vae,
                scheduler=FlowMatchEulerDiscreteScheduler(), text_encoder=None,
                text_encoder_2=None, text_encoder_3=None, tokenizer=None, tokenizer_2=None,
                tokenizer_3=None)
            pipe.save_pretrained(Path(folder) / 'pipeline')
            restored = demo.StableDiffusion3InstructPix2PixPipeline.from_pretrained(
                Path(folder) / 'pipeline', transformer=loaded, use_safetensors=True,
                torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True,
                text_encoder=None, text_encoder_2=None, text_encoder_3=None,
                tokenizer=None, tokenizer_2=None, tokenizer_3=None)
            self.assertTrue(all(p.dtype == torch.float16 for p in restored.vae.parameters()))
            self.assertFalse(torch.cuda.is_initialized())

    def check_mocked_loader(self, invalid=False):
        calls = {}; references = []
        class TinyTransformer(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.pos_embed = torch.nn.Module()
                self.pos_embed.proj = torch.nn.Conv2d(16, 8, 2, 2, dtype=dtype)
            def register_to_config(self, **kw): calls['config'] = kw
            def to(self, device=None, dtype=None):
                calls['to'] = (str(device), dtype)
                return super().to(dtype=dtype)  # deliberately CPU-only
        def transformer_factory(*a, **kw):
            calls['transformer'] = kw
            return TinyTransformer(kw['torch_dtype'])
        class TinyPipe:
            def to(self, device, dtype): return self
            def load_lora_weights(self, path, **kwargs): calls['lora'] = (path, kwargs)
        def pipe_factory(*a, **kw):
            calls['pipeline'] = kw
            return TinyPipe()
        real_load = torch.load
        def load(*a, **kw):
            calls['checkpoint'] = kw
            state = real_load(*a, **kw)
            references.extend(weakref.ref(t) for t in state.values())
            return state
        with tempfile.TemporaryDirectory() as folder:
            conv = torch.nn.Conv2d(96, 8, 2, 2)
            expected = {'pos_embed.proj.' + k: v for k, v in conv.state_dict().items()}
            checkpoint = Path(folder) / 'weights.bin'
            torch.save({'wrong_key': torch.zeros(1)} if invalid else expected, checkpoint)
            args = argparse.Namespace(device='cuda', im_size=512, sd3_path=folder,
                transformer_ckpt=str(checkpoint), lora_path='original_lora.safetensors')
            try:
                with patch.object(torch.cuda, 'is_available', return_value=True), \
                     patch.object(demo.SD3Transformer2DModel, 'from_pretrained', transformer_factory), \
                     patch.object(demo.StableDiffusion3InstructPix2PixPipeline, 'from_pretrained', pipe_factory), \
                     patch.object(torch, 'load', load):
                    if invalid:
                        with self.assertRaises(RuntimeError): demo.init_pipeline(args)
                    else:
                        demo.init_pipeline(args)
                if invalid:
                    self.assertNotIn('pipeline', calls)
                else:
                    self.assertEqual(calls['config'], {'in_channels': 96})
                    self.assertEqual(calls['transformer']['torch_dtype'], torch.float16)
                    self.assertTrue(calls['transformer']['low_cpu_mem_usage'])
                    self.assertEqual(calls['pipeline']['torch_dtype'], torch.float16)
                    self.assertTrue(calls['pipeline']['low_cpu_mem_usage'])
                    actual = calls['pipeline']['transformer'].state_dict()
                    for k, v in expected.items(): self.assertTrue(torch.equal(actual[k], v.half()))
                    self.assertEqual(calls['lora'], (os.path.dirname(os.path.abspath(args.lora_path)),
                        {'weight_name': 'pytorch_lora_weights.safetensors', 'local_files_only': True}))
                self.assertEqual(calls['checkpoint'], {'map_location': 'cpu', 'weights_only': True, 'mmap': True})
                gc.collect()
                self.assertTrue(all(r() is None for r in references), 'CPU checkpoint references retained')
                self.assertFalse(torch.cuda.is_initialized())
            finally:
                demo._GLOBAL.clear()

    def test_half_loading_strict_weights_and_release(self): self.check_mocked_loader()
    def test_release_on_checkpoint_mismatch(self): self.check_mocked_loader(invalid=True)

    def test_worker_saves_metrics_during_mocked_loading(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            def fake_load(args):
                time.sleep(1.15)
                live = json.loads((out/'metrics.json').read_text())
                self.assertEqual(live['status'], 'loading')
                self.assertGreater(live['cpu_rss_peak_bytes'], 0)
            array = np.tile(np.arange(512, dtype=np.uint8)[None, :, None], (512, 1, 3))
            images = tuple(Image.new('RGB', (2, 2)) for _ in range(5))
            with patch.object(sys, 'argv', ['test-worker', '--confirmed', '--output-dir', str(out)]), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                 patch.object(torch.cuda, 'synchronize'), \
                 patch.object(torch.cuda, 'memory_allocated', return_value=0), \
                 patch.object(torch.cuda, 'memory_reserved', return_value=0), \
                 patch.object(torch.cuda, 'max_memory_allocated', return_value=0), \
                 patch.object(torch.cuda, 'max_memory_reserved', return_value=0), \
                 patch.object(demo, 'init_pipeline', fake_load), \
                 patch.object(demo, 'load_example_set', return_value=images), \
                 patch.object(demo, 'run_forward', return_value=[(array, 'rainy')]), \
                 patch.object(demo, 'WEATHER_CONDITIONS', {'rainy': 'A rainy day.'}):
                self.assertEqual(forward_minimal_once.main(), 0)
            self.assertEqual(json.loads((out/'metrics.json').read_text())['status'], 'success')
            self.assertFalse(torch.cuda.is_initialized())


class OfflineLoraTests(unittest.TestCase):
    def test_actual_offline_local_fixture_and_production_call(self):
        import ast
        import inspect
        import socket
        import requests
        import diffusers
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict
        self.assertEqual(diffusers.__version__, '0.35.1')
        self.assertEqual(os.environ.get('HF_HUB_OFFLINE'), '1')
        network_attempts = []
        def forbidden_network(*args, **kwargs):
            network_attempts.append('attempted')
            raise AssertionError('Network access forbidden in offline LoRA fixture test')
        def tiny_model():
            return SD3Transformer2DModel(sample_size=4, patch_size=2, in_channels=16,
                out_channels=16, num_layers=1, attention_head_dim=8, num_attention_heads=1,
                joint_attention_dim=8, caption_projection_dim=8, pooled_projection_dim=4,
                pos_embed_max_size=4)
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(requests.sessions.Session, 'request', forbidden_network), \
             patch.object(socket.socket, 'connect', forbidden_network), \
             patch.object(socket, 'create_connection', forbidden_network):
            source = tiny_model()
            source.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=['to_q', 'to_v']))
            with torch.no_grad():
                for name, parameter in source.named_parameters():
                    if 'lora_' in name:
                        parameter.fill_(0.125 if 'lora_A' in name else 0.25)
            expected = get_peft_model_state_dict(source)
            self.assertTrue(expected)
            demo.StableDiffusion3InstructPix2PixPipeline.save_lora_weights(
                folder, transformer_lora_layers=expected,
                weight_name='pytorch_lora_weights.safetensors', safe_serialization=True)
            path = Path(folder)/'pytorch_lora_weights.safetensors'
            self.assertLess(path.stat().st_size, 100_000)
            target = tiny_model()
            pipe = demo.StableDiffusion3InstructPix2PixPipeline(transformer=target,
                vae=None, scheduler=FlowMatchEulerDiscreteScheduler(), text_encoder=None,
                text_encoder_2=None, text_encoder_3=None, tokenizer=None,
                tokenizer_2=None, tokenizer_3=None)
            # Verify the exact old failure, then execute the actual edited call via AST.
            with self.assertRaisesRegex(ValueError, 'must specify a `weight_name`'):
                pipe.load_lora_weights(str(path))
            tree = ast.parse(inspect.getsource(demo.init_pipeline))
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == 'load_lora_weights']
            self.assertEqual(len(calls), 1)
            expression = ast.Expression(body=calls[0])
            eval(compile(ast.fix_missing_locations(expression), '<actual-LoRA-call>', 'eval'),
                 {'pipe': pipe, 'os': os, 'args': argparse.Namespace(lora_path=str(path))})
            adapters = list(target.peft_config)
            self.assertEqual(len(adapters), 1)
            actual = get_peft_model_state_dict(target, adapter_name=adapters[0])
            self.assertEqual(set(actual), set(expected))
            for name in expected:
                self.assertTrue(torch.equal(actual[name], expected[name]), name)
            self.assertTrue(all(p.device.type == 'cpu' for p in target.parameters()))
            self.assertFalse(torch.cuda.is_initialized())
            self.assertEqual(network_attempts, [])
            print(json.dumps({'diffusers': diffusers.__version__, 'fixture_bytes': path.stat().st_size,
                'adapter_tensors_verified': len(actual), 'network_attempts': len(network_attempts),
                'cuda_initialized': torch.cuda.is_initialized(), 'old_error_reproduced': True,
                'edited_call_loaded_successfully': True}), flush=True)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def run_case(self, mode='sleep', timeout=.35, gpu=None, memory=None):
        # Child detaches into its own session and ignores TERM: requires orphan
        # discovery plus SIGKILL, rather than killing only the initial worker PID.
        child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"
        code = ("import subprocess,sys,time,signal; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"p=subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True); "
                "print('CHILD_PID='+str(p.pid),flush=True); "
                + ("time.sleep(.12)" if mode == 'normal' else "time.sleep(30)"))
        policy = supervisor.Policy(timeout_seconds=timeout, min_available_bytes=1,
            max_worker_rss_bytes=1024**3, sample_seconds=.025, gpu_interval_seconds=.02,
            terminate_grace_seconds=.15, kill_grace_seconds=.6)
        out = self.base / mode
        result = supervisor.supervise([sys.executable, '-u', '-c', code], out, policy,
            gpu_query=gpu or (lambda: 0), memory_reader=memory)
        self.assertEqual(result['cleanup']['remaining_pids'], [])
        self.assertTrue((out/'resource_samples.csv').read_text().count('\n') >= 2)
        self.assertEqual(json.loads((out/'supervisor.json').read_text())['status'], result['status'])
        text = (out/'full.log').read_text()
        if 'CHILD_PID=' in text:
            pid = int(text.split('CHILD_PID=')[1].splitlines()[0])
            self.assertFalse(psutil.pid_exists(pid), 'Detached child survived cleanup')
        return result

    def test_timeout_and_detached_child_cleanup(self):
        self.assertEqual(self.run_case()['status'], 'time_limit_exceeded')
    def test_normal_exit_also_cleans_child(self):
        r = self.run_case(mode='normal', timeout=1)
        self.assertTrue(r['passed'])
    def test_gpu_query_timeout_is_nonfatal(self):
        def query(): raise subprocess.TimeoutExpired('simulated-nvidia-smi', .02)
        r = self.run_case(gpu=query)
        self.assertEqual(r['status'], 'time_limit_exceeded')
        self.assertEqual(r['gpu_monitor_errors'][0]['type'], 'TimeoutExpired')
    def test_gpu_query_exception_is_nonfatal(self):
        def query(): raise OSError('simulated GPU query failure')
        r = self.run_case(gpu=query)
        self.assertEqual(r['status'], 'time_limit_exceeded')
        self.assertEqual(r['gpu_monitor_errors'][0]['type'], 'OSError')
    def test_blocked_gpu_query_does_not_disable_deadline(self):
        def query(): time.sleep(.8); return 0
        start = time.monotonic()
        r = self.run_case(timeout=.2, gpu=query)
        self.assertEqual(r['status'], 'time_limit_exceeded')
        self.assertLess(time.monotonic()-start, 2.5)
    def test_system_available_guard(self):
        self.assertEqual(self.run_case(memory=lambda pid: (1, 0))['status'], 'system_memory_guard')
    def test_worker_rss_guard(self):
        self.assertEqual(self.run_case(memory=lambda pid: (1024**3, 10*1024**3))['status'], 'worker_rss_guard')
    def test_unexpected_monitor_exception_cleanup(self):
        count = 0
        def memory(pid):
            nonlocal count
            count += 1
            if count >= 4: raise RuntimeError('simulated monitor exception')
            return (psutil.Process(pid).memory_info().rss, 10*1024**3)
        self.assertEqual(self.run_case(memory=memory)['status'], 'supervisor_error')
    def test_user_interrupt_cleanup(self):
        def interrupt(): os.kill(os.getpid(), signal.SIGINT)
        timer = threading.Timer(.2, interrupt)
        timer.start()
        try: self.assertEqual(self.run_case(timeout=2)['status'], 'user_interrupted')
        finally: timer.cancel(); timer.join()
    def test_ssh_hangup_cleanup(self):
        timer = threading.Timer(.2, lambda: os.kill(os.getpid(), signal.SIGHUP))
        timer.start()
        try: self.assertEqual(self.run_case(timeout=2)['status'], 'user_interrupted')
        finally: timer.cancel(); timer.join()
    def test_actual_gpu_probe_timeout_reaps_probe(self):
        real_popen = subprocess.Popen
        pids = []
        probes = []
        def fake_probe(command, **kwargs):
            self.assertEqual(command[0], 'nvidia-smi')
            p = real_popen([sys.executable, '-c', 'import time; time.sleep(30)'], **kwargs)
            pids.append(p.pid)
            probes.append(p)
            return p
        sampler = supervisor.GPUSampler(supervisor.Policy(gpu_timeout_seconds=.03,
            gpu_interval_seconds=.1))
        with patch.object(supervisor.subprocess, 'Popen', fake_probe):
            sampler.thread.start()
            end = time.monotonic()+2
            try:
                while not sampler.snapshot()['errors'] and time.monotonic()<end:
                    time.sleep(.01)
            finally:
                result = sampler.close()
        self.assertEqual(sampler.snapshot()['errors'][0]['type'], 'TimeoutExpired')
        self.assertTrue(result['query_thread_stopped'])
        self.assertIsNone(result['remaining_query_pid'])
        self.assertTrue(pids)
        self.assertTrue(all(not psutil.pid_exists(pid) for pid in pids))
        self.assertTrue(all(p.stdout.closed and p.stderr.closed for p in probes))


class SupervisorSafetyTests(unittest.TestCase):
    def check_cleanup_failure(self, target):
        with tempfile.TemporaryDirectory() as folder:
            policy = supervisor.Policy(timeout_seconds=.12, min_available_bytes=1,
                max_worker_rss_bytes=1024**3, sample_seconds=.02,
                terminate_grace_seconds=.1, kill_grace_seconds=.5)
            with patch.object(target, 'close' if target is supervisor.GPUSampler else 'cleanup',
                              side_effect=RuntimeError('simulated cleanup helper failure')):
                result = supervisor.supervise([sys.executable, '-c', 'import time; time.sleep(30)'],
                    Path(folder)/'out', policy, gpu_query=lambda: 0)
            self.assertTrue(result['cleanup_errors'])
            self.assertFalse(result['passed'])
            self.assertFalse(psutil.pid_exists(result['worker_pid']))
            self.assertEqual(result['cleanup']['remaining_pids'], [])
            self.assertTrue((Path(folder)/'out/supervisor.json').exists())
    def test_gpu_cleanup_exception_still_cleans_worker(self):
        self.check_cleanup_failure(supervisor.GPUSampler)
    def test_worker_cleanup_exception_uses_bounded_fallback(self):
        self.check_cleanup_failure(supervisor.OwnedWorkers)
    def test_low_memory_preflight_does_not_spawn_worker(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(supervisor.psutil, 'virtual_memory', return_value=SimpleNamespace(available=0)), \
             patch.object(supervisor.subprocess, 'Popen') as spawn:
            result = supervisor.supervise(['must-not-start'], Path(folder)/'out',
                gpu_query=lambda: 0)
            self.assertEqual(result['status'], 'preflight_memory_guard')
            self.assertFalse(result['passed'])
            spawn.assert_not_called()
    def test_reused_root_pid_is_not_adopted(self):
        owner = supervisor.OwnedWorkers()
        owner.subreaper = False
        owner.root_identity = (12345, 1.0)
        impostor = SimpleNamespace(pid=12345, create_time=lambda: 2.0,
            children=lambda **kwargs: self.fail('Traversed an unrelated reused PID'))
        with patch.object(supervisor.psutil, 'Process', return_value=impostor):
            owner.refresh(SimpleNamespace(pid=12345), set())
        self.assertEqual(owner.known, {})


if __name__ == '__main__':
    unittest.main(verbosity=2)
