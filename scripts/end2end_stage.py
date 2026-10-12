"""Disposable stage supervisor or worker; real mode is Linux/CUDA/offline only."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', GRADIO_ANALYTICS_ENABLED='False')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=['inverse', 'forward'])
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--allow-real-models', action='store_true')
    args = parser.parse_args()
    from end2end.contracts import read_json, atomic_json, verify_input
    task = args.task.resolve()
    request = read_json(task / 'request.json')
    real = request['backend'] == 'real'
    if real and (not args.allow_real_models or not sys.platform.startswith('linux')):
        parser.error('Real loading requires explicit --allow-real-models on Linux.')
    if args.worker or not real:
        # Mock runs directly as a childless process: no Gradio, torch or psutil import.
        from end2end import adapters
        started = time.monotonic()
        try:
            verify_input(task, request)
            if real:
                getattr(adapters, args.stage)(task, request)
            else:
                adapters.mock(task, request, args.stage)
            atomic_json(task / args.stage / 'worker.json', dict(status='success',
                        backend=request['backend'], seconds=time.monotonic() - started))
            return 0
        except Exception as exc:
            (task / args.stage / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
            atomic_json(task / args.stage / 'worker.json', dict(status='failed', error=str(exc),
                        backend=request['backend'], seconds=time.monotonic() - started))
            traceback.print_exc()
            return 1
    from scripts.forward_minimal_supervisor import supervise, Policy
    command = [sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--task', str(task),
               '--stage', args.stage, '--worker', '--allow-real-models']
    record = supervise(command, task / (args.stage + '_supervision'), Policy(),
                       cancel_requested=lambda: (task / 'cancel.flag').exists())
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
