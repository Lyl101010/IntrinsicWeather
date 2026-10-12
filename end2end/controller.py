"""One task at a time, two fully exited processes, immutable per-task artifacts."""
import atexit
from dataclasses import dataclass, field
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid
import zipfile

from PIL import Image, ImageOps

from .contracts import (AOVS, atomic_json, load_maps, parameters, read_json,
                        sha256, checked_path, verify_input)

ROOT = Path(__file__).resolve().parents[1]


class BusyError(RuntimeError):
    pass


class TaskCancelled(RuntimeError):
    pass


class ProcessLock:
    """OS releases the persistent lock on exit; never unlink a locked inode."""
    def __init__(self, path):
        self.path, self.stream, self.locked = path, None, False

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        try:
            if os.fstat(self.stream.fileno()).st_size == 0:
                self.stream.write(b'0'); self.stream.flush()
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.locked = True
        except OSError:
            self.stream.close(); self.stream = None
            raise BusyError('另一个服务正在生成，请等待当前任务结束。')

    def release(self):
        if self.stream is not None:
            if self.locked and os.name == 'nt':
                import msvcrt
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            elif self.locked:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close(); self.stream = None; self.locked = False


@dataclass
class Event:
    stage: str
    task_id: str = ''
    backend: str = 'mock'
    elapsed: float = 0
    input: str | None = None
    maps: list = field(default_factory=lambda: [None] * 5)
    final: str | None = None
    archive: str | None = None
    parameters: str | None = None
    error: str | None = None
    terminal: bool = False


@dataclass
class ActiveTask:
    task: Path
    cancel: threading.Event = field(default_factory=threading.Event)
    process: object = None
    owner: object = None
    log: object = None
    stop_lock: threading.RLock = field(default_factory=threading.RLock)


class Controller:
    def __init__(self, results_root=ROOT / 'outputs/end2end', backend='mock',
                 allow_real_models=False, poll_seconds=0.5, stage_timeout=1200):
        if backend not in ('mock', 'real'):
            raise ValueError('Unknown backend.')
        if backend == 'real' and (not allow_real_models or not sys.platform.startswith('linux')):
            raise ValueError('Real backend requires explicit authorization and Linux.')
        self.root = Path(results_root).resolve()
        self.backend = backend
        self.poll_seconds = poll_seconds
        self.stage_timeout = stage_timeout
        self.lock = threading.Lock()
        self.registry_lock = threading.Lock()
        self.active = {}
        self.poisoned = False
        atexit.register(self.close)

    def cancel(self, session):
        with self.registry_lock:
            active = self.active.get(session)
            if active is None:
                return '当前会话没有运行中的任务。'
            active.cancel.set()
            (active.task / 'cancel.flag').touch()
        return '已请求取消；正在等待进程树清理，清理完成前不能开始新任务。'

    def close(self):
        with self.registry_lock:
            tasks = list(self.active.values())
        for active in tasks:
            active.cancel.set()
            (active.task / 'cancel.flag').touch()
            self._stop(active)

    def _stop(self, active):
        with active.stop_lock:
            return self._stop_locked(active)

    def _stop_locked(self, active):
        process = active.process
        if process is None:
            if active.log is not None:
                active.log.close(); active.log = None
            return
        (active.task / 'cancel.flag').touch()
        if self.backend == 'real':
            # Give the supervisor time to observe the flag and perform bounded cleanup.
            try:
                process.wait(timeout=16)
            except subprocess.TimeoutExpired:
                pass
            from scripts.forward_minimal_supervisor import Policy
            try:
                cleanup = active.owner.cleanup(process, set(), Policy())
            except Exception:
                # Do not release the GPU lock if cleanup itself could not be verified.
                self.poisoned = True
                for proc in active.owner.alive():
                    try:
                        proc.kill()
                    except Exception:
                        pass
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=3)
                finally:
                    if active.log is not None:
                        active.log.close(); active.log = None
                raise
            atomic_json(active.task / 'outer_cleanup.json', cleanup)
            if cleanup['remaining_pids']:
                self.poisoned = True
                raise RuntimeError('进程树清理未完成；服务已禁止新任务，请检查日志。')
        elif process.poll() is None:
            # The mock CLI is deliberately childless; it cannot load any CUDA model.
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=3)
        if process.poll() is None:
            self.poisoned = True
            raise RuntimeError('Stage process did not exit.')
        active.process = None
        if active.log is not None:
            active.log.close(); active.log = None

    def _stage(self, active, request, stage):
        task = active.task
        if active.cancel.is_set():
            raise TaskCancelled('任务已取消。')
        command = [sys.executable, '-B', '-u', str(ROOT / 'scripts/end2end_stage.py'),
                   '--task', str(task), '--stage', stage]
        if self.backend == 'real':
            command.append('--allow-real-models')
            from scripts.forward_minimal_supervisor import OwnedWorkers
            active.owner = OwnedWorkers()
        active.log = (task / stage / 'launcher.log').open('x', encoding='utf-8')
        started = time.monotonic()
        active.process = subprocess.Popen(command, stdout=active.log, stderr=subprocess.STDOUT,
                                           start_new_session=(os.name == 'posix'))
        process = active.process
        if self.backend == 'real':
            import psutil
            root = psutil.Process(process.pid)
            active.owner.root_identity = (root.pid, root.create_time())
            active.owner.remember(root)
        try:
            while process.poll() is None:
                if self.backend == 'real':
                    active.owner.refresh(process, set())
                if active.cancel.is_set():
                    raise TaskCancelled('任务已取消。')
                # Extra outer deadline includes the supervisor's own bounded cleanup.
                if time.monotonic() - started > self.stage_timeout + (20 if self.backend == 'real' else 0):
                    raise TimeoutError(stage + ' exceeded stage timeout.')
                yield stage
                active.cancel.wait(self.poll_seconds)
            code = process.returncode
            if active.cancel.is_set():
                raise TaskCancelled('任务已取消。')
            if code != 0:
                detail = ''
                if (task / stage / 'worker.json').exists():
                    detail = read_json(task / stage / 'worker.json').get('error', '')
                elif (task / (stage + '_supervision') / 'supervisor.json').exists():
                    detail = read_json(task / (stage + '_supervision') / 'supervisor.json')['status']
                raise RuntimeError(stage + ' failed (exit ' + str(code) + '): ' + detail)
            if self.backend == 'real':
                record = read_json(task / (stage + '_supervision') / 'supervisor.json')
                if not record['passed'] or record['cleanup']['remaining_pids']:
                    raise RuntimeError('Stage cleanup not confirmed.')
            if read_json(task / stage / 'worker.json')['status'] != 'success':
                raise RuntimeError('Stage worker did not report success.')
        finally:
            # The lock is held until the stage and its descendants have exited.
            self._stop(active)
            # Normal completion does not cancel the next stage.
            if not active.cancel.is_set() and (task / 'cancel.flag').exists():
                (task / 'cancel.flag').unlink()

    def generate(self, image, prompt, seed, steps, guidance, image_guidance, session,
                 mock_controls=None):
        event = Event(stage='validating', backend=self.backend)
        active = None
        acquired = False
        process_lock = ProcessLock((ROOT / 'outputs/.intrinsicweather-gpu.lock') if self.backend == 'real'
                                   else self.root / '.gpu_task.lock')
        started = time.monotonic()
        try:
            p = parameters(prompt, seed, steps, guidance, image_guidance)
            if not isinstance(image, Image.Image) or image.width * image.height > 25_000_000:
                raise ValueError('请上传一张 RGB 图片（最多 2500 万像素）。')
            if self.poisoned or not self.lock.acquire(blocking=False):
                raise BusyError('已有生成任务或清理未完成，请稍后再试。')
            acquired = True
            process_lock.acquire()
            task_id = uuid.uuid4().hex
            task = self.root / task_id
            task.mkdir(parents=True, exist_ok=False)
            for stage in ('inverse', 'forward'):
                (task / stage).mkdir()
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            rgb.save(task / 'input.png')
            request = dict(schema_version=1, task_id=task_id, backend=self.backend,
                mock=self.backend == 'mock', input_sha256=sha256(task / 'input.png'),
                parameters=p, created_unix=time.time(),
                python=sys.version, automatic_retries=0,
                model_paths=dict(sd3='sd3.5_medium', inverse='checkpoints/InverseRenderer-512',
                                 dino='dinov2_base', forward='checkpoints/ForwardRenderer'),
                code_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
                    [ROOT / 'gradio_end2end_demo.py', ROOT / 'gradio_inverse_demo.py',
                     ROOT / 'gradio_forward_demo.py', ROOT / 'scripts/forward_minimal_supervisor.py',
                     ROOT / 'inverse_renderer_pipeline.py', ROOT / 'forward_renderer_pipeline.py',
                     ROOT / 'extract_dino_feature.py', ROOT / 'IMAA.py', ROOT / 'custom_model/transformer.py',
                     ROOT / 'scripts/end2end_stage.py', *sorted((ROOT / 'end2end').glob('*.py'))]})
            if mock_controls:
                if self.backend != 'mock':
                    raise ValueError('Mock controls forbidden for real backend.')
                request['mock_controls'] = mock_controls
            atomic_json(task / 'request.json', request)
            active = ActiveTask(task)
            with self.registry_lock:
                self.active[session] = active
            event.task_id = task_id
            event.input = str(task / 'input.png')
            event.parameters = str(task / 'request.json')
            event.stage = 'inverse:starting'
            yield event
            for stage in ('inverse', 'forward'):
                verify_input(task, request)
                for _ in self._stage(active, request, stage):
                    event.stage = stage + ':starting'
                    if (task / 'stage.json').exists():
                        reported = read_json(task / 'stage.json')['stage']
                        if reported.startswith(stage + ':'):
                            event.stage = reported
                    event.elapsed = time.monotonic() - started
                    yield event
                if stage == 'inverse':
                    _, manifest = load_maps(task, request)
                    event.maps = [str(checked_path(task, m['png'])) for m in manifest['maps']]
                    event.stage = 'inverse:complete / GPU worker exited'
                    event.elapsed = time.monotonic() - started
                    # This yield happens BEFORE Forward is even started.
                    yield event
                else:
                    manifest = read_json(task / 'forward/result.json')
                    if any(manifest.get(k) != request[k] for k in ('task_id', 'input_sha256', 'backend')):
                        raise ValueError('Forward result belongs to another task.')
                    path = checked_path(task, manifest['final'])
                    if sha256(path) != manifest['final_sha256']:
                        raise ValueError('Final hash mismatch.')
                    with Image.open(path) as final:
                        if final.mode != 'RGB' or final.size != (512, 512):
                            raise ValueError('Invalid final image.')
                    event.final = str(path)
            event.stage = 'success'
            event.terminal = True
            event.elapsed = time.monotonic() - started
            self._record_and_archive(active, event)
            yield event
        except Exception as exc:
            event.stage = 'cancelled' if isinstance(exc, TaskCancelled) else 'failed'
            event.error = str(exc)
            event.terminal = True
            event.elapsed = time.monotonic() - started
            if active is not None:
                self._stop(active)
                self._record_and_archive(active, event)
            yield event
        finally:
            try:
                if active is not None:
                    try:
                        interrupted_stream = not event.terminal
                        if not event.terminal:
                            active.cancel.set()
                            event.stage = 'cancelled'; event.terminal = True
                            event.error = '事件流关闭；已取消任务并清理进程。'
                        self._stop(active)
                        if interrupted_stream:
                            event.elapsed = time.monotonic() - started
                            self._record_and_archive(active, event)
                    finally:
                        with self.registry_lock:
                            self.active.pop(session, None)
            finally:
                if acquired:
                    # If cleanup failed, keep the OS lock and reject all further work.
                    try:
                        if not self.poisoned:
                            process_lock.release()
                        else:
                            self._quarantine_lock = process_lock
                    finally:
                        self.lock.release()

    def _record_and_archive(self, active, event):
        atomic_json(active.task / 'experiment.json', dict(task_id=event.task_id,
            backend=self.backend, mock=self.backend == 'mock', stage=event.stage,
            elapsed_seconds=event.elapsed, error=event.error, automatic_retries=0,
            final=event.final, maps=event.maps))
        archive = active.task / ('MOCK_results.zip' if self.backend == 'mock' else 'results.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(active.task.rglob('*')):
                if path.is_file() and path != archive and not path.name.endswith(('.tmp', '.flag')):
                    bundle.write(path, str(path.relative_to(active.task)))
        event.archive = str(archive)
