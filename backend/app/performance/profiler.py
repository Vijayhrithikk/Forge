"""Runtime Profiler — minimal-overhead CPU, RAM, IO sampling."""
import time, os, threading, json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class RuntimeProfiler:
    """Collects CPU, RAM, and timing metrics with minimal overhead."""
    def __init__(self):
        self._samples: list = []
        self._start_time = time.time()
        self._peak_ram = 0.0
        self._thread = None
        self._running = False

    def start(self, interval: float = 0.5):
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, args=(interval,), daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._running = False
        if self._thread: self._thread.join(timeout=2)
        elapsed = time.time() - self._start_time
        ram_samples = [s["ram_mb"] for s in self._samples if s.get("ram_mb")]
        return {
            "duration_seconds": round(elapsed, 2),
            "sample_count": len(self._samples),
            "peak_ram_mb": round(max(ram_samples) if ram_samples else 0, 1),
            "avg_ram_mb": round(sum(ram_samples) / len(ram_samples), 1) if ram_samples else 0,
            "samples": self._samples[-10:],  # Last 10 for detail
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _sample_loop(self, interval: float):
        while self._running:
            try:
                import psutil
                proc = psutil.Process()
                ram = proc.memory_info().rss / (1024 * 1024)
                cpu = proc.cpu_percent()
                self._samples.append({
                    "time": round(time.time() - self._start_time, 2),
                    "ram_mb": round(ram, 1),
                    "cpu_percent": cpu,
                })
            except ImportError:
                self._samples.append({"time": round(time.time() - self._start_time, 2), "ram_mb": 0, "cpu_percent": 0})
            time.sleep(interval)
