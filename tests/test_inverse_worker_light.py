"""Mocks only: one inverse worker, raw output diagnostics, and routing/cleanup."""
import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from PIL import Image
import psutil
import torch
import gradio_inverse_demo as official
import inverse_minimal_once as worker
import inverse_minimal_supervisor as entry
import forward_minimal_supervisor as supervisor

class WorkerTests(unittest.TestCase):
    def fixture(self,folder):
        root=Path(folder)/'project'
        image=root/'assets/examples/image/1.png';image.parent.mkdir(parents=True)
        Image.new('RGB',(9,8),(128,120,100)).save(image)
        for name in ['sd3.5_medium/model_index.json','checkpoints/InverseRenderer-512/pytorch_model-00001-of-00002.bin',
            'checkpoints/InverseRenderer-512/pytorch_model-00002-of-00002.bin','checkpoints/InverseRenderer-512/pytorch_model.bin.index.json',
            'checkpoints/InverseRenderer-512/imaa.pth','dinov2_base/config.json','dinov2_base/preprocessor_config.json','dinov2_base/model.safetensors']:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture only, never load')
        for name in ['gradio_inverse_demo.py','scripts/inverse_minimal_once.py','scripts/inverse_minimal_supervisor.py']:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
        return root,Path(folder)/'result'

    def run_mock(self,root,out,*,cuda=True,invalid=None,pipe_failure=False,cpu_fallback=False,resource_error=False):
        calls=[];seed_draws=[];models=[]
        good=np.broadcast_to(np.linspace(0,1,512,dtype=np.float32)[None,:,None],(512,512,3)).copy()
        class Pipe:
            def encode_prompt(self,**kwargs):return (torch.zeros(1),None,torch.zeros(1),None)
            def __call__(self,**kwargs):
                calls.append(kwargs)
                if pipe_failure and len(calls)==2:raise RuntimeError('simulated second AOV failure')
                array=good.copy()
                if invalid=='nan':array[0,0,0]=np.nan
                if invalid=='inf':array[0,0,0]=np.inf
                if invalid=='shape':array=array[:1]
                if invalid=='range':array[0,0,0]=2
                if invalid=='integer':array=array.astype(np.uint8)
                if invalid=='constant':array.fill(0)
                return ([array],)
        def init(args):
            self.assertEqual(args.device,'cuda');self.assertEqual(args.im_size,512)
            self.assertTrue(args.dino_path.endswith('dinov2_base'))
            seed_draws.append((random.random(),np.random.rand(),torch.rand(1).item()))
            models.append(args)
            official._GLOBAL_BUNDLE={'device':torch.device('cpu' if cpu_fallback else 'cuda'),'dtype':torch.float16,
                'pipe':Pipe(),'imaa':torch.nn.Linear(1,1),'dino_model':torch.nn.Linear(1,1)}
            time.sleep(.55)  # permit at least two periodic diagnostic samples
        def inference(image,steps):
            self.assertEqual(steps,50)
            pipe=official._GLOBAL_BUNDLE['pipe'];result=[]
            for aov in worker.AOVS:
                pipe.encode_prompt(prompt=aov)
                call=dict(image=None,prompt_embeds=torch.zeros(1),pooled_prompt_embeds=torch.zeros(1),
                    guidance_scale=0,image_guidance_scale=0,num_inference_steps=50,strength=1.0,
                    output_type='np',aov=[aov],map_aware_mask=torch.zeros(1))
                array=pipe(**call)[0][0]
                result.append((official._aov_np_to_display_uint8(array),aov))
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(worker,'ROOT',root))
            stack.enter_context(patch.object(sys,'argv',['worker','--confirmed','--output-dir',str(out)]))
            stack.enter_context(patch.object(worker,'assert_exclusive_gpu',return_value=''))
            stack.enter_context(patch.object(official,'init_models',init))
            stack.enter_context(patch.object(official,'run_inference',inference))
            stack.enter_context(patch.object(torch.cuda,'is_available',return_value=cuda))
            stack.enter_context(patch.object(torch.cuda,'is_initialized',return_value=cuda))
            seed=stack.enter_context(patch.object(torch.cuda,'manual_seed_all'))
            stack.enter_context(patch.object(torch.cuda,'reset_peak_memory_stats'))
            stack.enter_context(patch.object(torch.cuda,'synchronize'))
            if resource_error:stack.enter_context(patch.object(psutil.Process,'memory_info',side_effect=OSError('simulated resource monitor failure')))
            for name in ['memory_allocated','memory_reserved','max_memory_allocated','max_memory_reserved']:
                stack.enter_context(patch.object(torch.cuda,name,return_value=123))
            code=worker.main()
            if cuda and models:seed.assert_any_call(42)
        official._GLOBAL_BUNDLE=None
        self.assertFalse(torch.cuda.is_initialized(),'Actual CUDA must stay uninitialized outside mocks')
        return code,json.loads((out/'metrics.json').read_text()),calls,seed_draws

    def test_success_order_parameters_seed_raw_png_and_periodic_resources(self):
        with tempfile.TemporaryDirectory() as folder:
            root,out=self.fixture(folder);code,record,calls,draws=self.run_mock(root,out)
            self.assertEqual(code,0);self.assertEqual(record['status'],'success')
            self.assertEqual([call['aov'][0] for call in calls],list(worker.AOVS))
            for call in calls:
                self.assertEqual((call['num_inference_steps'],call['guidance_scale'],call['image_guidance_scale'],call['strength']),(50,0,0,1.0))
            self.assertEqual(draws[0][0],random.Random(42).random())
            self.assertEqual(draws[0][1],np.random.RandomState(42).rand())
            generator=torch.Generator().manual_seed(42)
            self.assertEqual(draws[0][2],torch.rand(1,generator=generator).item())
            for output in record['outputs']:
                raw=np.load(output['raw_path'],allow_pickle=False)
                self.assertEqual(raw.dtype,np.float32);self.assertEqual(raw.shape,(512,512,3));self.assertTrue(np.isfinite(raw).all())
                with Image.open(output['png_path']) as im:self.assertEqual(im.size,(512,512))
                self.assertTrue(output['valid']);self.assertGreater(output['variance'],0)
            self.assertGreaterEqual(len((out/'worker_resources.csv').read_text().splitlines()),3)
            self.assertEqual(record['gpu_allocated_peak_bytes'],123)
            self.assertEqual(record['automatic_retries'],0)

    def test_cuda_unavailable_saves_traceback_and_never_initializes_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root,out=self.fixture(folder);code,record,calls,draws=self.run_mock(root,out,cuda=False)
            self.assertEqual(code,1);self.assertEqual(calls,[]);self.assertEqual(draws,[])
            self.assertIn('CPU fallback forbidden',(out/'traceback.txt').read_text())

    def test_loader_cpu_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,out=self.fixture(folder);code,record,calls,_=self.run_mock(root,out,cpu_fallback=True)
            self.assertEqual(code,1);self.assertEqual(calls,[])
            self.assertIn('fell back',(out/'traceback.txt').read_text())

    def test_invalid_raw_preserved_and_not_hidden_by_png_conversion(self):
        for reason in ['nan','inf','shape','range','integer']:
            with self.subTest(reason=reason),tempfile.TemporaryDirectory() as folder:
                root,out=self.fixture(folder);code,record,calls,_=self.run_mock(root,out,invalid=reason)
                self.assertEqual(code,1);self.assertEqual(len(calls),1)
                self.assertTrue((out/'raw/albedo.npy').exists())
                self.assertFalse((out/'png/albedo.png').exists())
                self.assertFalse(record['outputs'][0]['valid']);self.assertTrue((out/'traceback.txt').exists())

    def test_second_aov_failure_keeps_completed_output_no_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root,out=self.fixture(folder);code,record,calls,_=self.run_mock(root,out,pipe_failure=True)
            self.assertEqual(code,1);self.assertEqual(len(calls),2)
            self.assertTrue((out/'png/albedo.png').exists());self.assertFalse((out/'raw/normal.npy').exists())
            self.assertEqual(record['failed_stage'],'inferring:normal')
            self.assertIn('simulated second AOV failure',(out/'traceback.txt').read_text())

    def test_confirmation_and_existing_output_refused_before_loading(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(sys,'argv',['worker','--output-dir',folder]),patch.object(torch.cuda,'is_available') as cuda:
            with self.assertRaises(SystemExit):worker.main()
            cuda.assert_not_called()
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'metrics.json').write_text('preserve')
            with patch.object(sys,'argv',['worker','--confirmed','--output-dir',folder]),patch.object(torch.cuda,'is_available') as cuda:
                with self.assertRaises(SystemExit):worker.main()
                cuda.assert_not_called();self.assertEqual((out/'metrics.json').read_text(),'preserve')

    def test_other_gpu_process_and_query_error_rejected(self):
        with patch.object(worker.subprocess,'run',return_value=argparse.Namespace(stdout='999999, ForwardRenderer\n')):
            with self.assertRaisesRegex(RuntimeError,'never co-load'):worker.assert_exclusive_gpu()
        with patch.object(worker.subprocess,'run',side_effect=subprocess.TimeoutExpired('nvidia-smi',3)):
            with self.assertRaises(subprocess.TimeoutExpired):worker.assert_exclusive_gpu()
        with patch.object(worker.subprocess,'run',return_value=argparse.Namespace(stdout='')):
            self.assertEqual(worker.assert_exclusive_gpu(),'')

    def test_constant_maps_are_recorded_without_false_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root,out=self.fixture(folder);code,record,_,_=self.run_mock(root,out,invalid='constant')
            self.assertEqual(code,0)
            self.assertTrue(all(v['constant_map'] and v['variance']==0 for v in record['outputs']))

    def test_worker_sampling_error_retained_and_success_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,out=self.fixture(folder);code,record,_,_=self.run_mock(root,out,resource_error=True)
            self.assertEqual(code,1);self.assertTrue(record['resource_monitor_errors'])
            self.assertIn('resource monitor failed',(out/'traceback.txt').read_text())

class EntryTests(unittest.TestCase):
    def test_confirmed_entry_routes_only_inverse_with_default_policy(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(sys,'argv',['entry','--confirmed','--output-dir',folder]), \
             patch.object(entry,'supervise',return_value={'passed':True}) as call:
            self.assertEqual(entry.main(),0)
            command,out,policy=call.call_args.args
            self.assertTrue(command[3].endswith('inverse_minimal_once.py'))
            self.assertEqual((policy.timeout_seconds,policy.min_available_bytes,policy.max_worker_rss_bytes),(1200,4*1024**3,22*1024**3))
        with patch.object(sys,'argv',['entry','--output-dir','unused']),patch.object(entry,'supervise') as call:
            with self.assertRaises(SystemExit):entry.main()
            call.assert_not_called()

    def test_inverse_entry_timeout_ram_rss_monitor_exception_and_hangup_cleanup(self):
        for mode in ['normal','timeout','available','rss','exception','hangup']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                folder=Path(folder)
                child='import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)'
                text='import subprocess,sys,time,signal\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\n'+f'p=subprocess.Popen([sys.executable,"-c",{child!r}],start_new_session=True)\nprint("CHILD_PID="+str(p.pid),flush=True)\n'+('time.sleep(.1)' if mode=='normal' else 'time.sleep(20)')
                (folder/'inverse_minimal_once.py').write_text(text)
                policy=replace(supervisor.Policy(),timeout_seconds=.35,sample_seconds=.025,
                    gpu_interval_seconds=.02,terminate_grace_seconds=.1,kill_grace_seconds=.5)
                records=[];count=0
                def memory(pid):
                    nonlocal count
                    count+=1
                    if count>3 and mode=='exception':raise RuntimeError('simulated monitor error')
                    if count>3 and mode=='available':return (1,4*1024**3-1)
                    if count>3 and mode=='rss':return (22*1024**3,10*1024**3)
                    return (1,10*1024**3)
                def run(command,out,policy):
                    result=supervisor.supervise(command,out,policy,gpu_query=lambda:0,memory_reader=memory)
                    records.append(result);return result
                timer=threading.Timer(.15,lambda:os.kill(os.getpid(),signal.SIGHUP)) if mode=='hangup' else None
                try:
                    if timer:timer.start()
                    with patch.object(entry,'__file__',str(folder/'inverse_minimal_supervisor.py')),patch.object(entry,'Policy',return_value=policy),patch.object(entry,'supervise',run),patch.object(sys,'argv',['entry','--confirmed','--output-dir',str(folder/'out')]):entry.main()
                finally:
                    if timer:timer.cancel();timer.join()
                result=records[0]
                expected={'normal':'completed','timeout':'time_limit_exceeded','available':'system_memory_guard','rss':'worker_rss_guard','exception':'supervisor_error','hangup':'user_interrupted'}[mode]
                self.assertEqual(result['status'],expected);self.assertEqual(result['cleanup']['remaining_pids'],[])
                log=(folder/'out/full.log').read_text()
                if 'CHILD_PID=' in log:
                    pid=int(log.split('CHILD_PID=')[1].splitlines()[0]);self.assertFalse(psutil.pid_exists(pid))
                self.assertTrue((folder/'out/resource_samples.csv').exists())
                self.assertEqual(result['automatic_retries'],0)

if __name__=='__main__':unittest.main(verbosity=2)
