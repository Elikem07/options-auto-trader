"""process_manager.py — subprocess lifecycle helpers for auto_trader/backtest/monte_carlo.

Owns start/stop/status/log-tailing for scripts launched as separate OS
processes. Kept dependency-free and free of any trading logic.
"""
import subprocess
import sys
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_procs = {}       # name -> {"popen": Popen, "started_at": float}
ROOT = Path(__file__).parent


def _external_pids(script: str, exclude_pid=None):
    """PIDs of any OTHER process running this script, whether or not we started it."""
    try:
        result = subprocess.run(["pgrep", "-f", script], capture_output=True, text=True, timeout=3)
    except Exception:
        return []
    pids = [int(p) for p in result.stdout.split() if p.strip()]
    if exclude_pid is not None and exclude_pid in pids:
        pids.remove(exclude_pid)
    return pids


def start(name: str, script: str):
    with _lock:
        entry = _procs.get(name)
        if entry and entry["popen"].poll() is None:
            return {"started": False, "reason": "already running", "pid": entry["popen"].pid}

        known_pid = entry["popen"].pid if entry else None
        external = _external_pids(script, exclude_pid=known_pid)
        if external:
            return {
                "started": False,
                "reason": "an instance of this script is already running outside this app (not started here, refusing to start a second one)",
                "pid": external[0],
            }

        popen = subprocess.Popen(
            [sys.executable, script],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _procs[name] = {"popen": popen, "started_at": time.time()}
        return {"started": True, "pid": popen.pid}


def stop(name: str, timeout: float = 5):
    with _lock:
        entry = _procs.get(name)
        if not entry or entry["popen"].poll() is not None:
            return {"stopped": False, "reason": "not running (or running externally — this app can only stop processes it started)"}

        popen = entry["popen"]
        popen.terminate()
        try:
            popen.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            popen.kill()
            popen.wait(timeout=timeout)
        return {"stopped": True}


def status(name: str, script: str):
    with _lock:
        entry = _procs.get(name)
        if entry and entry["popen"].poll() is None:
            return {"running": True, "pid": entry["popen"].pid, "started_at": entry["started_at"], "external": False}
        known_pid = entry["popen"].pid if entry else None

    external = _external_pids(script, exclude_pid=known_pid)
    if external:
        return {"running": True, "pid": external[0], "started_at": None, "external": True}
    return {"running": False, "pid": None, "started_at": None, "external": False}


def tail_log(path: str, n: int = 50):
    p = Path(path)
    if not p.exists():
        return []
    with p.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = 4096
        data = b""
        while size > 0 and data.count(b"\n") <= n:
            step = min(block, size)
            size -= step
            f.seek(size)
            data = f.read(step) + data
    lines = data.decode(errors="replace").splitlines()
    return lines[-n:]
