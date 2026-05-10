"""Auto-Ingest Daemon — watches the replay buffer and learns in real-time.

Runs as a background thread or standalone process. Monitors the
interceptor's replay buffer for new entries and immediately converts
them into experiences in the Adaptive Mind's memory.

Usage:
    # As a background thread (inside another process):
    from neuro.learning.auto_ingest import start_watcher
    stop_fn = start_watcher()
    # ... later ...
    stop_fn()
    
    # As a standalone daemon:
    python -m neuro.learning.auto_ingest
"""

from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console

console = Console()

DEFAULT_BUFFER = Path("/media/seyominaoto/x/neurobridge/traces/replay_buffer.jsonl")
DEFAULT_MEMORY = Path("/media/seyominaoto/x/neurobridge/brain/experiences")
POLL_INTERVAL = 5.0  # seconds between checks


class BufferWatcher:
    """Watches the replay buffer and ingests new entries."""
    
    def __init__(
        self,
        buffer_path: Path = DEFAULT_BUFFER,
        memory_dir: Path = DEFAULT_MEMORY,
        embed_model: str = "nomic-embed-text",
        poll_interval: float = POLL_INTERVAL,
    ):
        self.buffer_path = buffer_path
        self.memory_dir = memory_dir
        self.embed_model = embed_model
        self.poll_interval = poll_interval
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_size = 0
        self._last_line_count = 0
        self._total_ingested = 0
        
        # Track file position (byte offset) so we only read new lines
        self._file_offset = 0
        if self.buffer_path.exists():
            self._file_offset = self.buffer_path.stat().st_size
            self._last_size = self._file_offset
            # Count existing lines
            with open(self.buffer_path) as f:
                self._last_line_count = sum(1 for _ in f)
    
    def _get_memory(self):
        """Lazy-load ExperienceMemory to avoid import overhead."""
        from neuro.learning.experience_memory import ExperienceMemory
        return ExperienceMemory(
            memory_dir=self.memory_dir,
            embed_model=self.embed_model,
        )
    
    def _check_and_ingest(self) -> int:
        """Check for new buffer entries and ingest them.
        
        Returns number of new experiences learned.
        """
        if not self.buffer_path.exists():
            return 0
        
        current_size = self.buffer_path.stat().st_size
        if current_size <= self._file_offset:
            return 0  # no new data
        
        memory = self._get_memory()
        learned = 0
        
        with open(self.buffer_path) as f:
            f.seek(self._file_offset)
            for line in f:
                try:
                    data = json.loads(line.strip())
                    msgs = data.get("messages", [])
                    if len(msgs) < 3:
                        continue
                    
                    query = msgs[1].get("content", "")
                    response = msgs[2].get("content", "")
                    source = data.get("metadata", {}).get("tool", "unknown")
                    
                    if len(query) < 5 or len(response) < 10:
                        continue
                    
                    exp = memory.learn(
                        query=query,
                        response=response,
                        source=source,
                        quality_score=0.7 if source in ("codex", "claude") else 0.5,
                    )
                    if exp:
                        learned += 1
                        
                except (json.JSONDecodeError, Exception):
                    continue
        
        self._file_offset = current_size
        self._last_size = current_size
        self._total_ingested += learned
        
        return learned
    
    def _watch_loop(self) -> None:
        """Main watch loop — runs in a thread."""
        console.print(f"[dim]Auto-ingest watching: {self.buffer_path}[/dim]")
        
        while not self._stop_event.is_set():
            try:
                learned = self._check_and_ingest()
                if learned > 0:
                    console.print(
                        f"[green]Auto-ingested {learned} new experience(s) "
                        f"(total: {self._total_ingested})[/green]"
                    )
            except Exception as e:
                console.print(f"[dim red]Auto-ingest error: {e}[/dim red]")
            
            self._stop_event.wait(self.poll_interval)
    
    def start(self) -> None:
        """Start watching in a background thread."""
        if self._thread and self._thread.is_alive():
            return
        
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="neuro-auto-ingest",
        )
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the watcher."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)
    
    def stats(self) -> dict:
        return {
            "buffer_path": str(self.buffer_path),
            "buffer_size_bytes": self._last_size,
            "total_ingested": self._total_ingested,
            "file_offset": self._file_offset,
            "watching": self._thread.is_alive() if self._thread else False,
        }


def start_watcher(
    buffer_path: Path = DEFAULT_BUFFER,
    memory_dir: Path = DEFAULT_MEMORY,
) -> Callable[[], None]:
    """Start the auto-ingest watcher. Returns a stop function."""
    watcher = BufferWatcher(buffer_path=buffer_path, memory_dir=memory_dir)
    watcher.start()
    return watcher.stop


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroBridge Auto-Ingest Daemon")
    parser.add_argument("--buffer", default=str(DEFAULT_BUFFER), help="Path to replay buffer")
    parser.add_argument("--memory", default=str(DEFAULT_MEMORY), help="Path to experience memory")
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL, help="Poll interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Ingest once and exit (don't watch)")
    args = parser.parse_args()
    
    watcher = BufferWatcher(
        buffer_path=Path(args.buffer),
        memory_dir=Path(args.memory),
        poll_interval=args.interval,
    )
    
    if args.once:
        learned = watcher._check_and_ingest()
        console.print(f"Ingested {learned} experiences")
    else:
        console.print("[bold cyan]NeuroBridge Auto-Ingest Daemon[/bold cyan]")
        console.print(f"  Buffer: {args.buffer}")
        console.print(f"  Memory: {args.memory}")
        console.print(f"  Interval: {args.interval}s")
        console.print("  Press Ctrl+C to stop\n")
        
        watcher.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[dim]Stopping...[/dim]")
            watcher.stop()
            console.print(f"Total ingested: {watcher._total_ingested}")
