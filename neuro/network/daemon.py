"""Phantom Network Daemon — decentralized communication for NeuroBridge nodes."""

import json
import socket
import threading
from typing import Any, Optional
from rich.console import Console
import nacl.secret
import nacl.utils

console = Console()

class NodeDaemon:
    """A background daemon for P2P node communication."""

    def __init__(self, port: int = 9999, secret_key: bytes | None = None):
        self.port = port
        self.secret_key = secret_key or nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)
        self.box = nacl.secret.SecretBox(self.secret_key)
        self.running = False
        self._server_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the node server."""
        self.running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        console.print(f"[bold green]✓ Phantom Node Daemon active on port {self.port}[/bold green]")

    def _run_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('0.0.0.0', self.port))
            s.listen()
            while self.running:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024 * 10)
                    if data:
                        try:
                            decrypted = self.box.decrypt(data)
                            message = json.loads(decrypted)
                            self._handle_message(message, addr)
                        except Exception as e:
                            console.print(f"[red]Failed to decrypt message from {addr}: {e}[/red]")

    def _handle_message(self, message: dict[str, Any], addr: tuple[str, int]):
        """Handle incoming node messages (e.g., skill sharing, trace sync)."""
        msg_type = message.get("type")
        if msg_type == "ping":
            console.print(f"[dim]Peer {addr[0]} pinged.[/dim]")
        elif msg_type == "trace_packet":
            console.print(f"[cyan]Received expert trace from {addr[0]}. Distilling...[/cyan]")
            # In Phase 11, this would trigger the training loop

    def send_packet(self, target_ip: str, message: dict[str, Any]):
        """Encrypt and send a packet to another node."""
        encrypted = self.box.encrypt(json.dumps(message).encode())
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((target_ip, self.port))
                s.sendall(encrypted)
        except Exception as e:
            console.print(f"[red]Failed to reach node {target_ip}: {e}[/red]")
