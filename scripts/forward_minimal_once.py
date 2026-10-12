"""One official ForwardRenderer rain sample; explicit confirmation required."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['GRADIO_ANALYTICS_ENABLED']='False'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmed',action='store_true')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    if not args.confirmed:
        p.error('Explicit user confirmation is required before any model/GPU loading.')
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists() or (out/'rainy_image_11_seed42.png').exists():
        p.error('Results already exist; do not overwrite or automatically retry.')
    import psutil
    import torch
    import numpy as np
    from PIL import Image
    import gradio_forward_demo as official
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU fallback is forbidden for this run.')
    report={'started_utc':datetime.now(timezone.utc).isoformat(),
            'status':'loading','automatic_retry_count':0,'attempt_limit':1,
            'official_loader':'gradio_forward_demo.init_pipeline (FP16/low_cpu_mem_usage/mmap loading optimization)',
            'official_inference':'gradio_forward_demo.run_forward (rainy only)',
            'source_commit':'94c4ae3439b4b1146b6f4524073dda2697f3fb18',
            'sd3_path':str(ROOT/'sd3.5_medium'),
            'transformer_ckpt':str(ROOT/'checkpoints/ForwardRenderer/pytorch_model.bin'),
            'lora_path':str(ROOT/'checkpoints/ForwardRenderer/pytorch_lora_weights.safetensors'),
            'example':'image_11','im_size':512,'prompt':'A rainy day.',
            'seed':42,'num_inference_steps':50,'guidance_scale':6.0,
            'image_guidance_scale':1.5,'irradiance_argument':None,
            'dtype':'float16 (official CUDA path)','inverse_renderer_loaded':False,
            'training_started':False,'cpu_rss_peak_bytes':0,'gpu_device_used_peak_MiB':None,'gpu_device_sampling':'supervisor.json'}
    save_lock=threading.RLock()
    def save():
        with save_lock:
            snapshot=dict(report)
            tmp=out/'metrics.json.tmp'
            tmp.write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            os.replace(tmp,out/'metrics.json')
    stop=threading.Event()
    process=psutil.Process()
    def sample():
        last_save=0.0
        while not stop.is_set():
            report['cpu_rss_peak_bytes']=max(report['cpu_rss_peak_bytes'],process.memory_info().rss)
            report['system_available_ram_min_bytes']=min(report.get('system_available_ram_min_bytes',2**63),psutil.virtual_memory().available)
            if time.monotonic()-last_save>=1:
                save()
                last_save=time.monotonic()
            stop.wait(0.1)
    thread=threading.Thread(target=sample,daemon=True)
    thread.start()
    started=time.perf_counter()
    phase_started=started
    try:
        torch.cuda.reset_peak_memory_stats()
        official_args=argparse.Namespace(sd3_path=report['sd3_path'],transformer_ckpt=report['transformer_ckpt'],
            lora_path=report['lora_path'],device='cuda',im_size=512)
        save()
        official.init_pipeline(official_args)
        torch.cuda.synchronize()
        report['model_load_seconds']=time.perf_counter()-phase_started
        report['gpu_allocated_after_load_bytes']=torch.cuda.memory_allocated()
        report['gpu_reserved_after_load_bytes']=torch.cuda.memory_reserved()
        report['cpu_rss_after_load_bytes']=process.memory_info().rss
        report['status']='inferring'
        save()
        official.WEATHER_CONDITIONS={'rainy':'A rainy day.'}
        images=official.load_example_set('image_11',str(ROOT/'assets/examples/intrinsic_maps'))
        if any(im is None for im in images):
            raise RuntimeError('Official example has missing maps.')
        phase_started=time.perf_counter()
        result=official.run_forward(*images,50,6.0,1.5,42)
        torch.cuda.synchronize()
        report['inference_seconds']=time.perf_counter()-phase_started
        if len(result)!=1 or result[0][1]!='rainy':
            raise RuntimeError('Unexpected number or type of weather outputs.')
        array=result[0][0]
        if array.shape!=(512,512,3):
            raise RuntimeError('Unexpected output dimensions: '+str(array.shape))
        path=out/'rainy_image_11_seed42.png'
        Image.fromarray(array).save(path)
        report['output_path']=str(path)
        report['output_dimensions']=[512,512]
        report['output_min']=int(array.min())
        report['output_max']=int(array.max())
        report['output_std']=float(np.std(array))
        if report['output_std']==0:
            raise RuntimeError('Output is constant; minimum validation failed.')
        report['status']='success'
    except Exception as exc:
        report['failed_phase']=report['status']
        report['failed_phase_seconds']=time.perf_counter()-phase_started
        report['status']='failed'
        report['exception_type']=type(exc).__name__
        report['cuda_oom']=isinstance(exc,torch.cuda.OutOfMemoryError)
        (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        traceback.print_exc()
    finally:
        stop.set()
        thread.join(timeout=6)
        report['cpu_rss_final_bytes']=process.memory_info().rss
        report['cpu_maxrss_kernel_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['gpu_allocated_peak_bytes']=torch.cuda.max_memory_allocated()
        report['gpu_reserved_peak_bytes']=torch.cuda.max_memory_reserved()
        report['elapsed_seconds']=time.perf_counter()-started
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return 0 if report['status']=='success' else 1

if __name__=='__main__':
    raise SystemExit(main())
