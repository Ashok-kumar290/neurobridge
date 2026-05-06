"""Phantom Network Skill Sharing (Adapter Sync)."""

import os
import json
import socket
import hashlib
from pathlib import Path
from typing import Any
from rich.console import Console

console = Console()
CHUNK_SIZE = 1024 * 1024  # 1MB chunks

def compute_hash(filepath: Path) -> str:
    """Compute SHA256 (acting as BLAKE3 for simplicity here) of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def handle_sync_req(message: dict[str, Any], addr: tuple[str, int], daemon) -> None:
    """Handle an incoming request for an adapter."""
    adapter_id = message.get("adapter_id")
    console.print(f"[cyan]Peer {addr[0]} requested adapter: {adapter_id}[/cyan]")
    
    # Check if we have it (mocking adapter path for now)
    adapter_path = Path.home() / ".neurobridge" / "adapters" / f"{adapter_id}.gguf"
    
    if not adapter_path.exists():
        console.print(f"[yellow]Adapter {adapter_id} not found. Sending NACK.[/yellow]")
        daemon.send_packet(addr[0], {"type": "sync_nack", "adapter_id": adapter_id, "reason": "not_found"})
        return
        
    file_hash = compute_hash(adapter_path)
    file_size = adapter_path.stat().st_size
    
    # Send ACK with metadata
    daemon.send_packet(addr[0], {
        "type": "sync_ack",
        "adapter_id": adapter_id,
        "hash": file_hash,
        "size": file_size
    })
    
    # Stream chunks over raw socket
    console.print(f"[green]Streaming {adapter_id} to {addr[0]}...[/green]")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((addr[0], daemon.port + 1)) # File stream port
            with open(adapter_path, "rb") as f:
                while chunk := f.read(CHUNK_SIZE):
                    s.sendall(chunk)
        console.print("[bold green]Stream complete![/bold green]")
    except Exception as e:
        console.print(f"[red]Failed to stream to {addr[0]}: {e}[/red]")
