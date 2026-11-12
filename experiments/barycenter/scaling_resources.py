"""Isolated benchmark jobs with sampled RSS limits and solver-only memory peaks.

Memory caps are watchdog limits, NOT simulated OOMs. A SIGKILL with no known
cause is reported as killed_unknown, never assumed to be an OOM.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def rss_bytes(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return 0


class MemoryMeter:
    """Current process RSS sampled only inside the measured solver window."""
    def __init__(self, device='cpu', interval=.01):
        self.device, self.interval = device, interval
        self.done = threading.Event()

    def __enter__(self):
        import torch
        self.baseline = self.peak = rss_bytes(os.getpid())
        self.cuda_baseline = 0
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.cuda_baseline = torch.cuda.memory_allocated()
        def monitor():
            while not self.done.wait(self.interval):
                self.peak = max(self.peak, rss_bytes(os.getpid()))
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        import torch
        if self.device == 'cuda':
            torch.cuda.synchronize()
        self.peak = max(self.peak, rss_bytes(os.getpid()))
        self.done.set()
        self.thread.join()
        self.stats = dict(cpu_rss_baseline_bytes=self.baseline,
                          cpu_peak_rss_bytes=self.peak,
                          cpu_peak_extra_bytes=max(0, self.peak-self.baseline),
                          cuda_baseline_allocated_bytes=self.cuda_baseline,
                          cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                                     if self.device == 'cuda' else 0),
                          cuda_peak_reserved_bytes=(torch.cuda.max_memory_reserved()
                                                    if self.device == 'cuda' else 0))


def supervise(command, folder, *, timeout_s, memory_limit_gb, env=None,
              startup_timeout_s=180., poll_s=.05):
    """Timing starts at worker ready.json; watchdog includes serialization tail.

    The measured successful total_ms is recorded separately by the worker.
    Watchdog memory cap is absolute child RSS, including Python/library baseline.
    """
    folder = Path(folder)
    for name in ('ready.json', 'result.json'):
        (folder / name).unlink(missing_ok=True)
    launched = time.monotonic()
    peak = 0
    reason = None
    ready = None
    with (folder / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            while process.poll() is None:
                peak = max(peak, rss_bytes(process.pid))
                marker = folder / 'ready.json'
                if ready is None and marker.exists():
                    ready = json.loads(marker.read_text())['monotonic']
                now = time.monotonic()
                if memory_limit_gb and peak > memory_limit_gb * 1024**3:
                    reason = 'memory_limit'
                elif ready is not None and timeout_s and now-ready > timeout_s:
                    reason = 'timeout'
                elif ready is None and now-launched > startup_timeout_s:
                    reason = 'startup_timeout'
                if reason:
                    process.kill()
                    break
                time.sleep(poll_s)
            process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    path = folder / 'result.json'
    if reason is None and path.exists():
        result = json.loads(path.read_text())
    else:
        result = dict(status=reason or ('killed_unknown' if process.returncode < 0 else 'worker_error'),
                      total_ms=None)
    result.update(watchdog_peak_rss_bytes=peak, worker_exit_code=process.returncode,
                  watchdog_elapsed_ms=1000*(time.monotonic()-(ready or launched)),
                  timeout_s=timeout_s, memory_limit_gb=memory_limit_gb)
    write_json(path, result)
    return result


def run_worker(spec_path, timeout_s, memory_limit_gb, threads):
    env = dict(os.environ)
    env.update({name: str(threads) for name in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    return supervise([sys.executable, '-m', 'experiments.barycenter.scaling',
                      '--worker', str(spec_path)], Path(spec_path).parent,
                     timeout_s=timeout_s, memory_limit_gb=memory_limit_gb, env=env)
