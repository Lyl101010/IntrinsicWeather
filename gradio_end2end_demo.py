"""Unified RGB -> Inverse -> Forward UI, with no CUDA model in the main process."""
import argparse
import os
from pathlib import Path
import signal

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

from end2end.controller import Controller


def build_demo(controller):
    import gradio as gr
    if gr.__version__ != '5.44.1':
        raise RuntimeError('This entry targets the verified Gradio 5.44.1; dependency changes are not automatic.')
    mock = controller.backend == 'mock'
    with gr.Blocks(title='IntrinsicWeather · End-to-end', theme=gr.themes.Soft(),
                   css='.gradio-container {max-width: 1280px !important; margin: auto;}') as demo:
        gr.Markdown('# IntrinsicWeather\n单张 RGB → 本征属性 → 目标天气 · 512 × 512')
        gr.Markdown('**MOCK 模式：全部输出为模拟图，不是科研结果。**' if mock else
                    'GPU 模式：两阶段依次加载、退出。自定义夜间天气尚未验证生成质量。')
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.Image(type='pil', image_mode='RGB', sources=['upload'],
                                  label='上传原始 RGB 照片', height=310)
                prompt = gr.Textbox(value='A rainy day.', label='目标天气 prompt', max_lines=3)
                with gr.Row():
                    seed = gr.Number(value=42, precision=0, label='seed', minimum=0, maximum=2**31-1)
                    steps = gr.Slider(1, 100, value=50, step=1, label='steps（两阶段共用）')
                guidance = gr.Slider(0, 15, value=6, step=0.1, label='guidance_scale（Forward）')
                image_guidance = gr.Slider(0, 5, value=1.5, step=0.1, label='image_guidance_scale（Forward）')
                with gr.Row():
                    generate = gr.Button('Generate', variant='primary')
                    cancel = gr.Button('取消当前任务', interactive=False)
                cancel_message = gr.Markdown()
            with gr.Column(scale=2):
                with gr.Row():
                    original = gr.Image(label='本次任务原图', interactive=False, height=330)
                    final = gr.Image(label='最终天气图像' + (' · MOCK' if mock else ''),
                                     interactive=False, height=330)
                status = gr.Textbox(label='当前阶段 / 耗时 / 错误', value='等待上传图片', interactive=False)
                with gr.Row():
                    archive = gr.File(label='下载本次实验（原图、五图、浮点数组、最终图、记录）')
                    params = gr.File(label='下载参数 JSON')
        gr.Markdown('### 本征属性图\nInverse 完成并退出后立即显示；随后开始 Forward。Irradiance 保留展示与记录，目前不参与 Forward 条件。')
        with gr.Row():
            maps = [gr.Image(label=name.title() + (' · MOCK' if mock else ''),
                             interactive=False, height=220) for name in
                    ('albedo', 'normal', 'roughness', 'metallic', 'irradiance')]

        def run(image, weather, random_seed, num_steps, gs, igs, request: gr.Request):
            for event in controller.generate(image, weather, random_seed, num_steps, gs, igs, request.session_hash):
                label = ('MOCK | ' if mock else '') + event.stage + f' | {event.elapsed:.1f} s'
                if event.task_id:
                    label += ' | task ' + event.task_id
                if event.error:
                    label += '\n' + event.error
                yield (event.input, *event.maps, event.final, label, event.archive, event.parameters,
                       gr.update(interactive=event.terminal), gr.update(interactive=not event.terminal))

        outputs = [original, *maps, final, status, archive, params, generate, cancel]
        generate.click(run, inputs=[upload, prompt, seed, steps, guidance, image_guidance],
                       outputs=outputs, concurrency_limit=1, concurrency_id='gpu_task',
                       trigger_mode='once', show_progress='minimal', api_name='generate')
        # Do not use Gradio cancels here: let the generator display the cleanup result.
        def cancel_request(request: gr.Request):
            return controller.cancel(request.session_hash)

        cancel.click(cancel_request, inputs=None, outputs=cancel_message,
                     queue=False, api_name=False)
        def unload_request(request: gr.Request):
            controller.cancel(request.session_hash)

        demo.unload(unload_request)
    demo.queue(default_concurrency_limit=1, max_size=1)
    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--mock', action='store_true', help='CPU synthetic demo (default).')
    mode.add_argument('--allow-real-models', action='store_true', help='Explicitly enable real workers on Linux.')
    parser.add_argument('--results-root', type=Path, default=Path(__file__).resolve().parent / 'outputs/end2end')
    args = parser.parse_args()
    controller = Controller(args.results_root, backend='real' if args.allow_real_models else 'mock',
                            allow_real_models=args.allow_real_models)
    demo = build_demo(controller)
    def stop_service(signum, frame):
        controller.close()
        raise SystemExit(128 + signum)
    if os.name == 'posix':
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop_service)
    try:
        demo.launch(server_name='127.0.0.1', server_port=7861, share=False, show_error=True,
                    allowed_paths=[str(controller.root)])
    finally:
        controller.close()


if __name__ == '__main__':
    main()
