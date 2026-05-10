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

def receive_adapter(message: dict[str, Any], daemon) -> None:
    """Open a listener to receive an adapter stream from a peer."""
    adapter_id = message.get("adapter_id")
    expected_hash = message.get("hash")
    file_size = message.get("size")
    
    save_path = Path.home() / ".neurobridge" / "adapters" / f"{adapter_id}.gguf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[cyan]Opening stream port for {adapter_id} ({file_size / 1024 / 1024:.1f} MB)...[/cyan]")
    
    # Open listener on Port + 1
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", daemon.port + 1))
        s.listen(1)
        s.settimeout(30) # Wait 30s for the peer to connect
        
        try:
            conn, addr = s.accept()
            with conn:
                console.print(f"[green]Receiving data from {addr[0]}...[/green]")
                with open(save_path, "wb") as f:
                    received_bytes = 0
                    while received_bytes < file_size:
                        chunk = conn.recv(CHUNK_SIZE)
                        if not chunk: break
                        f.write(chunk)
                        received_bytes += len(chunk)
            
            # Verify Integrity
            actual_hash = compute_hash(save_path)
            if actual_hash == expected_hash:
                console.print(f"[bold green]✓ Adapter {adapter_id} synced and verified![/bold green]")
            else:
                console.print(f"[bold red]✗ Hash mismatch! Corruption detected.[/bold red]")
                save_path.unlink(missing_ok=True)
                
        except socket.timeout:
            console.print("[red]Stream timed out. Peer failed to connect.[/red]")
        except Exception as e:
            console.print(f"[red]Error during sync: {e}[/red]")
