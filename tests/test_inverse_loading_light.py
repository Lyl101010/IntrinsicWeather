"""CPU fixtures only: indexed loading, offline APIs, and unchanged inverse interfaces."""
import argparse
import ast
import gc
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import requests
import torch
from PIL import Image
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import Dinov2Config, Dinov2Model, BitImageProcessor
import gradio_inverse_demo as demo
from IMAA import IMAA, build_attn_mask
from extract_dino_feature import extract_patch_tokens_min_windows

def tiny_transformer():
    return demo.SD3Transformer2DModel(sample_size=4,patch_size=2,in_channels=16,
        out_channels=16,num_layers=1,attention_head_dim=8,num_attention_heads=1,
        joint_attention_dim=8,caption_projection_dim=8,pooled_projection_dim=4,pos_embed_max_size=4)

def fixture(folder, corruption=None):
    folder=Path(folder)
    base=tiny_transformer()
    base.save_pretrained(folder/'transformer')
    base.register_to_config(in_channels=32)
    base.pos_embed.proj=torch.nn.Conv2d(32,8,2,2)
    state=base.state_dict()
    names=list(state)
    parts=[{k:state[k] for k in names[::2]},{k:state[k] for k in names[1::2]}]
    files=['pytorch_model-00001-of-00002.bin','pytorch_model-00002-of-00002.bin']
    if corruption=='shape':parts[0][names[0]]=torch.zeros(1)
    if corruption=='model_missing':parts[0].pop(names[0])
    if corruption=='model_extra':parts[0]['unexpected']=torch.zeros(1)
    if corruption=='duplicate':parts[1][names[0]]=state[names[0]]
    index={k:filename for filename,part in zip(files,parts) for k in part}
    if corruption=='index_missing':index['not_in_shard']=files[0]
    for filename,part in zip(files,parts):torch.save(part,folder/filename)
    (folder/'pytorch_model.bin.index.json').write_text(json.dumps({'weight_map':index}))
    return state,[str(folder/name) for name in files]

class InverseLoadingTests(unittest.TestCase):
    def test_real_half_low_cpu_strict_indexed_weights_and_release(self):
        refs=[];real=demo._load_torch
        def load(*args,**kwargs):
            state=real(*args,**kwargs);refs.extend(weakref.ref(v) for v in state.values());return state
        with tempfile.TemporaryDirectory() as folder:
            expected,paths=fixture(folder)
            with patch.object(demo,'_load_torch',load):
                model=demo.build_transformer(folder,*paths,torch.device('cpu'),torch.float16)
            self.assertEqual(model.config.in_channels,32)
            for k,v in expected.items():self.assertTrue(torch.equal(model.state_dict()[k],v.half()),k)
            gc.collect();self.assertTrue(all(ref() is None for ref in refs))
            self.assertFalse(torch.cuda.is_initialized())

    def test_missing_unexpected_shape_duplicate_and_index_fail_closed(self):
        for reason in ['shape','model_missing','model_extra','duplicate','index_missing']:
            with self.subTest(reason=reason),tempfile.TemporaryDirectory() as folder:
                _,paths=fixture(folder,reason)
                with self.assertRaises((RuntimeError,ValueError)):
                    demo.build_transformer(folder,*paths,torch.device('cpu'),torch.float16)
        with tempfile.TemporaryDirectory() as folder:
            bad=Path(folder)/'legacy.bin';bad.write_bytes(b'not a ZIP checkpoint')
            with self.assertRaisesRegex(ValueError,'ZIP'):demo._load_torch(str(bad),'cpu')

    def test_checkpoint_release_on_strict_failure(self):
        refs=[];real=demo._load_torch
        def load(*args,**kwargs):
            state=real(*args,**kwargs);refs.extend(weakref.ref(v) for v in state.values());return state
        with tempfile.TemporaryDirectory() as folder:
            _,paths=fixture(folder,'model_missing')
            with patch.object(demo,'_load_torch',load):
                try:demo.build_transformer(folder,*paths,torch.device('cpu'),torch.float16)
                except RuntimeError:pass
                else:self.fail('Missing model weights accepted')
            gc.collect();self.assertTrue(all(ref() is None for ref in refs))

    def test_real_tiny_inverse_pipeline_fp16_api(self):
        with tempfile.TemporaryDirectory() as folder:
            model=tiny_transformer()
            vae=AutoencoderKL(block_out_channels=(32,),down_block_types=('DownEncoderBlock2D',),
                up_block_types=('UpDecoderBlock2D',),layers_per_block=1,latent_channels=16,
                norm_num_groups=8,sample_size=4)
            none={name:None for name in ['text_encoder','text_encoder_2','text_encoder_3','tokenizer','tokenizer_2','tokenizer_3']}
            pipe=demo.StableDiffusion3InstructPix2PixPipeline(transformer=model,vae=vae,
                scheduler=FlowMatchEulerDiscreteScheduler(),**none)
            pipe.save_pretrained(folder)
            restored=demo.StableDiffusion3InstructPix2PixPipeline.from_pretrained(folder,
                transformer=model.to(dtype=torch.float16),use_safetensors=True,torch_dtype=torch.float16,
                low_cpu_mem_usage=True,local_files_only=True,**none)
            self.assertTrue(all(p.dtype==torch.float16 for p in restored.vae.parameters()))
            self.assertFalse(torch.cuda.is_initialized())

    def test_offline_local_dino_imaa_and_init_api(self):
        attempts=[]
        def forbidden(*args,**kwargs):attempts.append(1);raise AssertionError('Network request attempted')
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(requests.sessions.Session,'request',forbidden), \
             patch.object(socket.socket,'connect',forbidden),patch.object(socket,'create_connection',forbidden):
            folder=Path(folder);expected,paths=fixture(folder)
            imaa=IMAA();torch.save({'epoch':1,'model_state_dict':imaa.state_dict(),'val_loss':0.1},folder/'imaa.pth')
            dino=Dinov2Model(Dinov2Config(image_size=224,patch_size=14,hidden_size=768,
                num_hidden_layers=0,num_attention_heads=12,intermediate_size=16))
            dino.save_pretrained(folder/'dino')
            processor=BitImageProcessor();processor.save_pretrained(folder/'dino')
            class Pipe:
                def to(self,device,dtype):self.dtype=dtype;return self
            def pipeline_factory(*args,**kwargs):
                self.assertTrue(kwargs['low_cpu_mem_usage']);self.assertTrue(kwargs['local_files_only'])
                self.assertEqual(kwargs['torch_dtype'],torch.float32);return Pipe()
            args=argparse.Namespace(device='cpu',im_size=512,sd3_path=str(folder),
                transformer_ckpt0=paths[0],transformer_ckpt1=paths[1],imaa_path=str(folder/'imaa.pth'),dino_path=str(folder/'dino'))
            try:
                with patch.object(demo.StableDiffusion3InstructPix2PixPipeline,'from_pretrained',pipeline_factory):
                    demo.init_models(args)
                bundle=demo._GLOBAL_BUNDLE
                self.assertEqual(bundle['dino_model'].dtype,torch.float32)
                self.assertEqual(next(bundle['imaa'].parameters()).dtype,torch.float32)
                for k,v in imaa.state_dict().items():self.assertTrue(torch.equal(bundle['imaa'].state_dict()[k],v))
                inputs=bundle['dino_processor'](images=np.zeros((224,224,3),dtype=np.uint8),return_tensors='pt')
                self.assertEqual(inputs['pixel_values'].dtype,torch.float32)
                with torch.no_grad():tokens=bundle['dino_model'](**inputs).last_hidden_state[:,1:]
                self.assertEqual(tuple(tokens.shape),(1,256,768))
                self.assertEqual(attempts,[]);self.assertFalse(torch.cuda.is_initialized())
            finally:demo._GLOBAL_BUNDLE=None

class InverseInterfaceTests(unittest.TestCase):
    def test_supervisor_defaults_and_exact_boundary_guards(self):
        from dataclasses import replace
        from scripts import forward_minimal_supervisor as supervisor
        default=supervisor.Policy()
        self.assertEqual(default.timeout_seconds,1200)
        self.assertEqual(default.min_available_bytes,4*1024**3)
        self.assertEqual(default.max_worker_rss_bytes,22*1024**3)
        policy=replace(default,timeout_seconds=.12,sample_seconds=.01,
            gpu_interval_seconds=.01,terminate_grace_seconds=.1,kill_grace_seconds=.3)
        boundaries=[(22*1024**3-1,4*1024**3,'time_limit_exceeded'),
            (1,4*1024**3-1,'system_memory_guard'),(22*1024**3,4*1024**3,'worker_rss_guard')]
        for rss,available,status in boundaries:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as folder:
                result=supervisor.supervise([sys.executable,'-c','import time;time.sleep(10)'],
                    Path(folder)/'out',policy,gpu_query=lambda:0,memory_reader=lambda pid:(rss,available))
                self.assertEqual(result['status'],status)
                self.assertEqual(result['cleanup']['remaining_pids'],[])
                self.assertEqual(result['automatic_retries'],0)

    def test_original_preprocessing_patch_coverage_imaa_and_attention(self):
        image=Image.fromarray(np.full((8,9,3),128,dtype=np.uint8))
        original=demo.pil_to_model_input(image,512,torch.device('cpu'),torch.float16)
        self.assertEqual(tuple(original.shape),(1,3,512,512));self.assertAlmostEqual(float(original[0,0,0,0]),128/255*2-1,places=4)
        pixels=[]
        class Processor:
            def __call__(self,images,return_tensors):
                pixels.append(images.copy())
                class Inputs(dict):
                    def to(self,device):return self
                return Inputs(pixel_values=torch.zeros(1,3,224,224))
        class Dino:
            config=argparse.Namespace(hidden_size=768,patch_size=14)
            def __call__(self,**kwargs):return argparse.Namespace(last_hidden_state=torch.ones(1,257,768))
        tokens=extract_patch_tokens_min_windows(original,Dino(),Processor(),device='cpu')
        self.assertEqual(tuple(tokens.shape),(1,36,36,768));self.assertTrue(torch.equal(tokens,torch.ones_like(tokens)))
        self.assertEqual(len(pixels),9);self.assertEqual(int(pixels[0][0,0,0]),1)
        # Preserve and explicitly document the existing signed-input clipping behavior.
        negative=original.fill_(-0.5);pixels.clear()
        extract_patch_tokens_min_windows(negative,Dino(),Processor(),device='cpu')
        self.assertEqual(int(pixels[0][0,0,0]),0)
        imaa=IMAA().eval()
        with torch.no_grad():gating=imaa(patch_tokens=tokens,output_size=(32,32),map_ids=torch.tensor([0]))
        mask=build_attn_mask(gating,154,1024,0.7)
        self.assertEqual(tuple(mask.shape),(1,1,1,1178));self.assertEqual(mask.dtype,torch.float32)
        self.assertTrue(torch.isfinite(mask).all())
        model=tiny_transformer();model.register_to_config(in_channels=32);model.pos_embed.proj=torch.nn.Conv2d(32,8,2,2)
        with torch.no_grad():result=model(hidden_states=torch.zeros(1,32,4,4),encoder_hidden_states=torch.zeros(1,2,8),
            pooled_projections=torch.zeros(1,4),timestep=torch.ones(1),map_aware_mask=torch.zeros(1,1,1,6)).sample
        self.assertEqual(tuple(result.shape),(1,16,4,4));self.assertTrue(torch.isfinite(result).all())

    def test_five_aovs_order_and_call_arguments_unchanged(self):
        calls=[]
        class Pipe:
            def encode_prompt(self,**kwargs):return (torch.zeros(1),None,torch.zeros(1),None)
            def __call__(self,**kwargs):calls.append(kwargs);return ([np.full((1,512,512,3),0.5,dtype=np.float32)],)
        class Mask:
            def __call__(self,**kwargs):return torch.ones(1,1,32,32)
        demo._GLOBAL_BUNDLE={'device':torch.device('cpu'),'dtype':torch.float16,'pipe':Pipe(),
            'imaa':Mask(),'dino_model':None,'dino_processor':None,'im_size':512}
        try:
            with patch.object(demo,'extract_patch_tokens_min_windows',return_value=torch.ones(1,36,36,768)):
                result=demo.run_inference(Image.new('RGB',(2,2)),50)
            self.assertEqual([c['aov'][0] for c in calls],['albedo','normal','roughness','metallic','irradiance'])
            self.assertEqual(len(result),5)
            for call in calls:
                self.assertEqual(call['num_inference_steps'],50);self.assertEqual(call['guidance_scale'],0)
                self.assertEqual(call['image_guidance_scale'],0);self.assertEqual(call['strength'],1.0)
                self.assertEqual(tuple(call['map_aware_mask'].shape),(1,1,1,1178))
        finally:demo._GLOBAL_BUNDLE=None

if __name__=='__main__':unittest.main(verbosity=2)
