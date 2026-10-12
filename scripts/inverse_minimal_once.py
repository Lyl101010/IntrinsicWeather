"""One offline RGB-to-five-AOV run; use inverse_minimal_supervisor.py for real runs."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GRADIO_ANALYTICS_ENABLED='False')
AOVS=('albedo','normal','roughness','metallic','irradiance')

def assert_exclusive_gpu():
    # A result-only Gradio viewer has no CUDA context and is not rejected here.
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=3)
    other=[]
    for row in csv.reader(result.stdout.splitlines()):
        if row and row[0].strip():
            pid=int(row[0].strip())
            if pid!=os.getpid():other.append({'pid':pid,'process':','.join(row[1:]).strip()})
    if other:raise RuntimeError('Other GPU processes exist; never co-load renderers: '+str(other))
    return result.stdout

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirmed',action='store_true')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if not args.confirmed:parser.error('Explicit confirmation required before model/GPU loading.')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    # The supervisor owns full.log/resource_samples.csv/supervisor.json; never overwrite a worker run.
    if any((out/name).exists() for name in ['metrics.json','raw','png','worker_resources.csv']):
        parser.error('Worker results already exist; no overwrite or automatic retry.')
    lock=threading.RLock()
    report={'started_utc':datetime.now(timezone.utc).isoformat(),'status':'preflight','stage':'preflight',
        'attempt_limit':1,'automatic_retries':0,'seed':42,'im_size':512,'num_inference_steps_per_aov':50,
        'guidance_scale':0,'image_guidance_scale':0,'strength':1.0,'aov_order':list(AOVS),
        'diffusion_dtype':'float16','dino_dtype':'float32','imaa_dtype':'float32',
        'forward_renderer_loaded':False,'training_started':False,'cpu_rss_peak_bytes':0,
        'resource_monitor_errors':[],'outputs':[],
        'quality_followup':'Official DINO signed-input clipping behavior preserved; assess separately.'}
    started=time.monotonic();stop=threading.Event();thread=None;torch=None;original_pipe=None;official=None
    def save():
        with lock:
            tmp=out/'metrics.json.tmp'
            tmp.write_text(json.dumps(copy.deepcopy(report),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            os.replace(tmp,out/'metrics.json')
    def update(**values):
        with lock:report.update(values);save()
    def stage(name):
        print('INVERSE_STAGE '+name,flush=True);update(stage=name)
    save()
    try:
        import numpy as np
        import psutil
        import torch
        from PIL import Image
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; CPU fallback forbidden.')
        update(gpu_process_preflight=assert_exclusive_gpu())
        sd3=ROOT/'sd3.5_medium';ckpt=ROOT/'checkpoints/InverseRenderer-512';dino=ROOT/'dinov2_base'
        image=ROOT/'assets/examples/image/1.png'
        required=[image,sd3/'model_index.json',ckpt/'pytorch_model-00001-of-00002.bin',
            ckpt/'pytorch_model-00002-of-00002.bin',ckpt/'pytorch_model.bin.index.json',ckpt/'imaa.pth',
            dino/'config.json',dino/'preprocessor_config.json',dino/'model.safetensors']
        if not all(path.is_file() for path in required):raise FileNotFoundError('Missing required offline files: '+str([str(p) for p in required if not p.is_file()]))
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        import gradio_inverse_demo as official
        update(input_path=str(image),input_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            sd3_path=str(sd3),inverse_path=str(ckpt),dino_path=str(dino),
            code_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                ['gradio_inverse_demo.py','scripts/inverse_minimal_once.py','scripts/inverse_minimal_supervisor.py']})
        process=psutil.Process()
        def sample():
            try:
                with (out/'worker_resources.csv').open('x',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['elapsed_seconds','stage','rss_bytes',
                        'available_bytes','gpu_allocated_bytes','gpu_reserved_bytes'])
                    while not stop.is_set():
                        rss=process.memory_info().rss;available=psutil.virtual_memory().available
                        allocated=reserved=None
                        if torch.cuda.is_initialized():
                            allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
                        with lock:
                            report['cpu_rss_peak_bytes']=max(report['cpu_rss_peak_bytes'],rss)
                            report['system_available_min_bytes']=min(report.get('system_available_min_bytes',available),available)
                            report['gpu_allocated_current_bytes']=allocated;report['gpu_reserved_current_bytes']=reserved
                            writer.writerow([time.monotonic()-started,report['stage'],rss,available,allocated,reserved])
                            stream.flush();save()
                        stop.wait(.5)
            except Exception:
                with lock:report['resource_monitor_errors'].append(traceback.format_exc());save()
        torch.cuda.reset_peak_memory_stats()
        thread=threading.Thread(target=sample,daemon=True);thread.start()
        update(status='loading');stage('loading_models');phase=time.perf_counter()
        official.init_models(argparse.Namespace(device='cuda',im_size=512,sd3_path=str(sd3),
            transformer_ckpt0=str(ckpt/'pytorch_model-00001-of-00002.bin'),
            transformer_ckpt1=str(ckpt/'pytorch_model-00002-of-00002.bin'),imaa_path=str(ckpt/'imaa.pth'),dino_path=str(dino)))
        torch.cuda.synchronize()
        bundle=official._GLOBAL_BUNDLE
        if bundle['device'].type!='cuda' or bundle['dtype']!=torch.float16:
            raise RuntimeError('Official loader fell back from CUDA FP16; abort.')
        for key in ['imaa','dino_model']:
            if any(p.dtype!=torch.float32 for p in bundle[key].parameters()):
                raise RuntimeError(key+' dtype changed from verified FP32.')
        update(model_load_seconds=time.perf_counter()-phase,status='inferring',
            gpu_allocated_after_load_bytes=torch.cuda.memory_allocated(),gpu_reserved_after_load_bytes=torch.cuda.memory_reserved())
        (out/'raw').mkdir();(out/'png').mkdir()
        original_pipe=bundle['pipe']
        class Recorder:
            def __init__(self):self.index=0
            def __getattr__(self,name):return getattr(original_pipe,name)
            def encode_prompt(self,*a,**kw):
                if self.index>=len(AOVS):raise RuntimeError('Unexpected extra AOV conditioning.')
                stage('conditioning:'+AOVS[self.index]);return original_pipe.encode_prompt(*a,**kw)
            def __call__(self,*a,**kw):
                if self.index>=len(AOVS):raise RuntimeError('Unexpected extra AOV inference.')
                name=AOVS[self.index]
                if kw.get('aov')!=[name] or kw.get('num_inference_steps')!=50 or kw.get('guidance_scale')!=0 or kw.get('image_guidance_scale')!=0 or kw.get('strength')!=1.0 or kw.get('output_type')!='np':
                    raise RuntimeError('Official AOV sequence/parameters differ from approved plan.')
                stage('inferring:'+name);t=time.perf_counter()
                result=original_pipe(*a,**kw);torch.cuda.synchronize()
                seconds=time.perf_counter()-t
                array=np.array(result[0][0],copy=True)
                stage('saving:'+name)
                raw=out/'raw'/f'{name}.npy'
                with raw.open('xb') as stream:np.save(stream,array,allow_pickle=False)
                finite=np.isfinite(array);values=array[finite]
                stats={'aov':name,'raw_path':str(raw),'shape':list(array.shape),'dtype':str(array.dtype),
                    'nan_count':int(np.isnan(array).sum()),'inf_count':int(np.isinf(array).sum()),
                    'min':float(values.min()) if values.size else None,'max':float(values.max()) if values.size else None,
                    'variance':float(np.var(values,dtype=np.float64)) if values.size else None,
                    'inference_seconds':seconds,'valid':False}
                with lock:report['outputs'].append(stats);save()
                if array.shape!=(512,512,3) or not np.issubdtype(array.dtype,np.floating) or not finite.all():
                    raise ValueError('Invalid raw AOV (shape/dtype/NaN/Inf): '+name)
                with lock:stats['in_expected_0_1']=stats['min']>=-1e-6 and stats['max']<=1+1e-6;save()
                if not stats['in_expected_0_1']:raise ValueError('Raw AOV outside expected [0,1]: '+name)
                png=out/'png'/f'{name}.png'
                with png.open('xb') as stream:Image.fromarray(official._aov_np_to_display_uint8(array)).save(stream,format='PNG')
                with lock:
                    stats.update(png_path=str(png),valid=True,constant_map=stats['variance']==0);save()
                self.index+=1;stage('completed:'+name)
                return result
        recorder=Recorder();bundle['pipe']=recorder
        stage('dino_and_aov_preparation');phase=time.perf_counter()
        with Image.open(image) as source:
            gallery=official.run_inference(source.copy(),50)
        torch.cuda.synchronize()
        if recorder.index!=5 or len(gallery)!=5:raise RuntimeError('Incomplete five-AOV result.')
        update(inference_total_seconds=time.perf_counter()-phase)
        if report['resource_monitor_errors']:raise RuntimeError('Worker resource monitor failed; diagnostics retained.')
        update(status='success');stage('success')
    except Exception as exc:
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');traceback.print_exc()
        update(status='failed',failed_stage=report['stage'],exception_type=type(exc).__name__,
            cuda_oom=bool(torch is not None and isinstance(exc,torch.cuda.OutOfMemoryError)))
    finally:
        stop.set()
        if thread is not None:thread.join(timeout=2)
        if official is not None and original_pipe is not None:official._GLOBAL_BUNDLE['pipe']=original_pipe
        if torch is not None and torch.cuda.is_initialized():
            try:update(gpu_allocated_peak_bytes=torch.cuda.max_memory_allocated(),gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved())
            except Exception:
                with lock:report['resource_monitor_errors'].append(traceback.format_exc())
        update(cpu_maxrss_kernel_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if report['status']=='success' else 1

if __name__=='__main__':raise SystemExit(main())
