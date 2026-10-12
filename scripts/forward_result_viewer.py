"""Read-only Gradio result viewer. Does not import torch or load models."""
import argparse
import json
from pathlib import Path
import os
os.environ['GRADIO_ANALYTICS_ENABLED']='False'
import gradio as gr

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result-dir',type=Path,required=True)
    args=p.parse_args()
    result=json.loads((args.result_dir/'metrics.json').read_text(encoding='utf-8'))
    if result['status']!='success':
        raise RuntimeError('No successful result to display.')
    image=Path(result['output_path']).resolve()
    with gr.Blocks(title='IntrinsicWeather ForwardRenderer 最小验证') as demo:
        gr.Markdown('## 官方 ForwardRenderer 雨天结果\n512×512 · A rainy day. · seed 42 · 50 steps')
        gr.Image(value=str(image),label='已生成的雨天图像',interactive=False)
    demo.launch(server_name='127.0.0.1',server_port=7861,share=False,
                allowed_paths=[str(args.result_dir.resolve())])

if __name__=='__main__':
    main()
