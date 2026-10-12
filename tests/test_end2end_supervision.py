"""Supervisor control-flow tests with mocked psutil metadata and real childless processes.

This fixture does NOT claim OS RSS accuracy or Linux descendant cleanup validation.
It allows local cancellation/guard regression without installing psutil or torch.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class SupervisorControlTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / 'stage'
        self.processes = {}
        processes = self.processes
        fake = ModuleType('psutil')
        class NoSuchProcess(Exception): pass
        class Process:
            def __init__(self, pid=None): self.pid = pid or os.getpid()
            def create_time(self): return float(self.pid)
            def children(self, recursive=False): return []
            def is_running(self): return self.pid not in processes or processes[self.pid].poll() is None
            def memory_info(self): return SimpleNamespace(rss=1024)
            def status(self): return 'running'
            def terminate(self): processes[self.pid].terminate()
            def kill(self): processes[self.pid].kill()
        fake.Process, fake.NoSuchProcess, fake.STATUS_ZOMBIE = Process, NoSuchProcess, 'zombie'
        fake.virtual_memory = lambda: SimpleNamespace(available=8 * 1024**3)
        spec = importlib.util.spec_from_file_location('e2e_test_supervisor', ROOT / 'scripts/forward_minimal_supervisor.py')
        self.supervisor = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'psutil': fake, 'e2e_test_supervisor': self.supervisor}):
            spec.loader.exec_module(self.supervisor)
        original = subprocess.Popen
        def tracked(*args, **kwargs):
            process = original(*args, **kwargs)
            self.processes[process.pid] = process
            return process
        self.popen_patch = patch.object(self.supervisor.subprocess, 'Popen', tracked)
        self.popen_patch.start()

    def tearDown(self):
        self.popen_patch.stop()
        for process in self.processes.values():
            if process.poll() is None: process.kill()
            process.wait(timeout=3)
        self.tmp.cleanup()

    def run_supervised(self, **kwargs):
        policy = self.supervisor.Policy(timeout_seconds=1, sample_seconds=.01,
            terminate_grace_seconds=.1, kill_grace_seconds=.1, gpu_interval_seconds=.01)
        return self.supervisor.supervise([sys.executable, '-c', 'import time; time.sleep(5)'],
            self.out, policy, gpu_query=kwargs.pop('gpu_query', lambda: None), **kwargs)

    def test_default_resource_policy_preserved(self):
        policy = self.supervisor.Policy()
        self.assertEqual((policy.timeout_seconds, policy.min_available_bytes, policy.max_worker_rss_bytes),
                         (1200, 4 * 1024**3, 22 * 1024**3))

    def test_cancel_before_spawn(self):
        record = self.run_supervised(cancel_requested=lambda: True)
        self.assertEqual(record['status'], 'user_interrupted')
        self.assertEqual(self.processes, {})

    def test_cancel_during_run_reaps_worker(self):
        started = time.monotonic()
        record = self.run_supervised(cancel_requested=lambda: time.monotonic() - started > .08)
        self.assertEqual(record['status'], 'user_interrupted')
        self.assertEqual(record['cleanup']['remaining_pids'], [])
        self.assertTrue(all(process.poll() is not None for process in self.processes.values()))

    def test_cancel_callback_exception_cleans_worker(self):
        def check():
            if self.processes: raise RuntimeError('injected callback failure')
            return False
        record = self.run_supervised(cancel_requested=check)
        self.assertEqual(record['status'], 'supervisor_error')
        self.assertEqual(record['cleanup']['remaining_pids'], [])

    def test_ram_rss_and_gpu_query_exception_isolation(self):
        def query(): raise TimeoutError('mock GPU query timeout')
        record = self.run_supervised(memory_reader=lambda pid: (23 * 1024**3, 8 * 1024**3), gpu_query=query)
        self.assertEqual(record['status'], 'worker_rss_guard')
        self.assertEqual(record['cleanup']['remaining_pids'], [])
        self.assertTrue(record['gpu_monitor_errors'])
        self.out = Path(self.tmp.name) / 'ram'
        record = self.run_supervised(memory_reader=lambda pid: (1024, 3 * 1024**3))
        self.assertEqual(record['status'], 'system_memory_guard')
        self.assertEqual(record['cleanup']['remaining_pids'], [])

    def test_normal_exit_and_worker_failure(self):
        for code in (0, 7):
            self.out = Path(self.tmp.name) / str(code)
            record = self.supervisor.supervise([sys.executable, '-c', 'raise SystemExit(' + str(code) + ')'],
                self.out, self.supervisor.Policy(sample_seconds=.01), gpu_query=lambda: None)
            self.assertEqual(record['passed'], code == 0)
            self.assertEqual(record['cleanup']['remaining_pids'], [])


if __name__ == '__main__': unittest.main()
