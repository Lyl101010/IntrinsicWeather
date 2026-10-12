"""Single-attempt supervisor with independent time/RAM guards and owned-process cleanup."""
import argparse
import csv
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import psutil

GIB = 1024 ** 3

@dataclass(frozen=True)
class Policy:
    timeout_seconds: float = 1200
    min_available_bytes: int = 4 * GIB
    max_worker_rss_bytes: int = 22 * GIB
    sample_seconds: float = 0.5
    gpu_timeout_seconds: float = 2
    gpu_interval_seconds: float = 2
    terminate_grace_seconds: float = 3
    kill_grace_seconds: float = 3

def atomic_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)

class GPUSampler:
    """Only this daemon thread waits for NVML; the policy loop never waits for it."""
    def __init__(self, policy, query=None):
        self.policy, self.query = policy, query
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.process = None
        self.identities = set()
        self.state = {"value_MiB": None, "peak_MiB": None, "errors": [], "updated_monotonic": None}
        self.thread = threading.Thread(target=self.run, daemon=True)

    def default_query(self):
        proc = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with self.lock:
            self.process = proc
            self.identities.add((proc.pid, psutil.Process(proc.pid).create_time()))
        try:
            out, err = proc.communicate(timeout=self.policy.gpu_timeout_seconds)
            if proc.returncode:
                raise RuntimeError("nvidia-smi exit " + str(proc.returncode) + ": " + err.strip())
            return int(out.strip().splitlines()[0])
        finally:
            if proc.poll() is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            # communicate() timed out before closing its pipes. Do not leak two
            # descriptors on every failed query during a long supervised run.
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
            with self.lock:
                if self.process is proc:
                    self.process = None if proc.poll() is not None else proc

    def run(self):
        while not self.stop_event.is_set():
            try:
                value = (self.query or self.default_query)()
                with self.lock:
                    self.state["value_MiB"] = value
                    if value is not None:
                        self.state["peak_MiB"] = max(self.state["peak_MiB"] or 0, value)
                    self.state["updated_monotonic"] = time.monotonic()
            except Exception as exc:
                with self.lock:
                    self.state["value_MiB"] = None
                    self.state["errors"].append({"type": type(exc).__name__, "message": str(exc)})
            self.stop_event.wait(self.policy.gpu_interval_seconds)

    def snapshot(self):
        with self.lock:
            return dict(self.state, errors=list(self.state["errors"]))

    def excluded(self):
        with self.lock:
            return set(self.identities)

    def close(self):
        self.stop_event.set()
        with self.lock:
            proc = self.process
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        self.thread.join(timeout=1.5)
        return {"query_thread_stopped": not self.thread.is_alive(),
                "remaining_query_pid": self.process.pid if self.process is not None and self.process.poll() is None else None}

class OwnedWorkers:
    def __init__(self):
        self.parent = psutil.Process()
        self.baseline = {(p.pid, p.create_time()) for p in self.parent.children(recursive=True)}
        self.known = {}
        self.root_identity = None
        self.subreaper = False
        if sys.platform.startswith("linux"):
            # Process-local setting: adopted orphans remain discoverable after a worker exits.
            self.subreaper = ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0

    def remember(self, proc):
        try:
            self.known[(proc.pid, proc.create_time())] = proc
        except psutil.NoSuchProcess:
            pass

    def refresh(self, worker, excluded):
        try:
            root = psutil.Process(worker.pid)
            if (root.pid, root.create_time()) == self.root_identity:
                self.remember(root)
                for p in root.children(recursive=True):
                    self.remember(p)
        except psutil.NoSuchProcess:
            pass
        if self.subreaper:
            for p in self.parent.children(recursive=True):
                try:
                    identity = (p.pid, p.create_time())
                    if identity not in self.baseline and identity not in excluded:
                        self.remember(p)
                except psutil.NoSuchProcess:
                    pass
        for identity in excluded:
            self.known.pop(identity, None)

    def alive(self):
        result = []
        for (pid, created), proc in self.known.items():
            try:
                if proc.create_time() == created and proc.is_running():
                    result.append(proc)
            except psutil.NoSuchProcess:
                pass
        return result

    def cleanup(self, worker, excluded, policy):
        self.refresh(worker, excluded)
        initial = [p.pid for p in self.alive()]
        # The worker owns a new POSIX session. Signal inherited descendants even
        # if the parent exits; separately tracked detached descendants are handled below.
        if os.name == "posix" and self.alive():
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in self.alive():
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        end = time.monotonic() + policy.terminate_grace_seconds
        while time.monotonic() < end and self.alive():
            worker.poll()
            self.refresh(worker, excluded)
            for p in self.alive():
                try:
                    if p.status() == psutil.STATUS_ZOMBIE:
                        try:
                            os.waitpid(p.pid, os.WNOHANG) if p.pid != worker.pid else None
                        except ChildProcessError:
                            pass
                except psutil.NoSuchProcess:
                    pass
            time.sleep(0.05)
        if os.name == "posix" and self.alive():
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for p in self.alive():
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        end = time.monotonic() + policy.kill_grace_seconds
        while time.monotonic() < end and self.alive():
            worker.poll()
            for p in self.alive():
                try:
                    os.waitpid(p.pid, os.WNOHANG) if p.pid != worker.pid else None
                except (ChildProcessError, ProcessLookupError):
                    pass
            time.sleep(0.05)
        return {"tracked_pids": sorted(set(initial + [pid for pid, _ in self.known])),
                "remaining_pids": [p.pid for p in self.alive()], "linux_subreaper": self.subreaper}

class StopRequested(BaseException):
    pass

def supervise(command, output_dir, policy=Policy(), gpu_query=None, memory_reader=None,
              cancel_requested=None):
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Output directory is not empty: no overwrite or automatic retry.")
    started = time.monotonic()
    record = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "starting",
        "attempt_limit": 1, "automatic_retries": 0, "policy": vars(policy),
        "cpu_rss_peak_bytes": 0, "system_available_ram_min_bytes": None,
        "gpu_device_used_peak_MiB": None, "command": list(command)}
    sampler = GPUSampler(policy, gpu_query)
    owner = OwnedWorkers()
    worker = None
    old_handlers = {}
    def stop_signal(signum, frame):
        raise StopRequested(signum)
    try:
        for sig in (signal.SIGINT, signal.SIGTERM) + ((signal.SIGHUP,) if hasattr(signal, "SIGHUP") else ()):
            old_handlers[sig] = signal.signal(sig, stop_signal)
        with (out / "full.log").open("x", encoding="utf-8") as log, (out / "resource_samples.csv").open("x", newline="", encoding="utf-8") as stream:
            samples = csv.writer(stream)
            samples.writerow(["elapsed_seconds", "worker_rss_bytes", "system_available_ram_bytes", "gpu_used_MiB", "gpu_sample_age_seconds", "gpu_query_errors"])
            if cancel_requested is not None and cancel_requested():
                raise StopRequested("cancel_requested")
            if memory_reader is None and psutil.virtual_memory().available < policy.min_available_bytes:
                record["status"] = "preflight_memory_guard"
                raise StopRequested("preflight_memory_guard")
            worker = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"))
            root_process = psutil.Process(worker.pid)
            owner.root_identity = (root_process.pid, root_process.create_time())
            owner.remember(root_process)
            record.update(worker_pid=worker.pid, status="running")
            atomic_json(out / "supervisor.json", record)
            sampler.thread.start()
            last_message = 0.0
            while True:
                elapsed = time.monotonic() - started
                owner.refresh(worker, sampler.excluded())
                if cancel_requested is not None and cancel_requested():
                    raise StopRequested("cancel_requested")
                code = worker.poll()
                if code is not None:
                    record.update(worker_exit_code=code, status="completed" if code == 0 else "worker_failed")
                    break
                if elapsed >= policy.timeout_seconds:
                    record["status"] = "time_limit_exceeded"
                    break
                try:
                    rss, available = memory_reader(worker.pid) if memory_reader else (psutil.Process(worker.pid).memory_info().rss, psutil.virtual_memory().available)
                except psutil.NoSuchProcess:
                    continue
                gpu = sampler.snapshot()
                age = None if gpu["updated_monotonic"] is None else time.monotonic() - gpu["updated_monotonic"]
                record["cpu_rss_peak_bytes"] = max(record["cpu_rss_peak_bytes"], rss)
                record["system_available_ram_min_bytes"] = min(record["system_available_ram_min_bytes"] or available, available)
                record["gpu_device_used_peak_MiB"] = gpu["peak_MiB"]
                record["gpu_monitor_errors"] = gpu["errors"]
                record["elapsed_seconds"] = elapsed
                samples.writerow([round(elapsed, 3), rss, available, gpu["value_MiB"], age, len(gpu["errors"])])
                stream.flush()
                atomic_json(out / "supervisor.json", record)
                if available < policy.min_available_bytes:
                    record["status"] = "system_memory_guard"
                    break
                if rss >= policy.max_worker_rss_bytes:
                    record["status"] = "worker_rss_guard"
                    break
                if elapsed - last_message >= 30:
                    print(f"Single attempt: {elapsed:.0f}s; RSS {rss/GIB:.2f} GiB; available {available/GIB:.2f} GiB; GPU {gpu['value_MiB']} MiB", flush=True)
                    last_message = elapsed
                time.sleep(policy.sample_seconds)
    except StopRequested as exc:
        if record["status"] != "preflight_memory_guard":
            record["status"] = "user_interrupted"
            record["interrupt_signal"] = exc.args[0]
    except KeyboardInterrupt:
        record["status"] = "user_interrupted"
        record["interrupt_signal"] = int(signal.SIGINT)
    except Exception:
        record["status"] = "supervisor_error"
        (out / "supervisor_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        # Ignore repeated interrupts during bounded cleanup; do not strand owned workers.
        for sig in old_handlers:
            signal.signal(sig, signal.SIG_IGN)
        record["gpu_monitor_errors"] = sampler.snapshot()["errors"]
        record["gpu_device_used_peak_MiB"] = sampler.snapshot()["peak_MiB"]
        record["cleanup_errors"] = []
        if sampler.thread.ident is not None:
            try:
                record["gpu_cleanup"] = sampler.close()
            except Exception:
                record["cleanup_errors"].append(traceback.format_exc())
                sampler.stop_event.set()
                if sampler.process is not None and sampler.process.poll() is None:
                    sampler.process.kill()
                sampler.thread.join(timeout=1.5)
        if worker is not None:
            try:
                record["cleanup"] = owner.cleanup(worker, sampler.excluded(), policy)
            except Exception:
                record["cleanup_errors"].append(traceback.format_exc())
                # A cleanup helper failure must not bypass terminating the actual
                # worker or the descendants already identified by creation time.
                for proc in owner.alive():
                    try:
                        proc.kill()
                    except psutil.NoSuchProcess:
                        pass
                if worker.poll() is None:
                    worker.kill()
                try:
                    worker.wait(timeout=policy.kill_grace_seconds)
                except subprocess.TimeoutExpired:
                    pass
                for proc in owner.alive():
                    if proc.pid != worker.pid:
                        try:
                            os.waitpid(proc.pid, os.WNOHANG)
                        except (ChildProcessError, ProcessLookupError):
                            pass
                record["cleanup"] = {"remaining_pids": [p.pid for p in owner.alive()]}
            record["worker_exit_code"] = worker.poll()
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["passed"] = record["status"] == "completed" and not record["cleanup_errors"] and not record.get("cleanup", {}).get("remaining_pids") and not record.get("gpu_cleanup", {}).get("remaining_query_pid")
        atomic_json(out / "supervisor.json", record)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    return record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmed", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not args.confirmed:
        parser.error("A new user confirmation is required before model/GPU loading.")
    root = Path(__file__).resolve().parents[1]
    record = supervise([sys.executable, "-u", str(root / "scripts/forward_minimal_once.py"),
                        "--confirmed", "--output-dir", str(args.output_dir.resolve())], args.output_dir)
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    return 0 if record["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
