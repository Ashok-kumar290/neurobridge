"""Node Identity — Cryptographic key management for the Phantom Network."""

import os
from pathlib import Path
from dataclasses import dataclass
import nacl.public
import nacl.encoding
import nacl.signing

@dataclass
class NodeIdentity:
    """A unique cryptographic identity for a NeuroBridge node."""
    
    public_key_hex: str
    private_key_hex: str
    signing_key_hex: str
    verify_key_hex: str

def get_identity_path() -> Path:
    """Get path to the node identity file."""
    path = Path.home() / ".neurobridge" / "identity.key"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def generate_identity() -> NodeIdentity:
    """Generate a new Curve25519 identity for this node."""
    # Encryption keys (Curve25519)
    sk = nacl.public.PrivateKey.generate()
    pk = sk.public_key
    
    # Signing keys (Ed25519)
    signing_key = nacl.signing.SigningKey.generate()
    verify_key = signing_key.verify_key
    
    return NodeIdentity(
        public_key_hex=pk.encode(nacl.encoding.HexEncoder).decode(),
        private_key_hex=sk.encode(nacl.encoding.HexEncoder).decode(),
        signing_key_hex=signing_key.encode(nacl.encoding.HexEncoder).decode(),
        verify_key_hex=verify_key.encode(nacl.encoding.HexEncoder).decode(),
    )

def load_or_create_identity() -> NodeIdentity:
    """Load existing identity or generate a new one."""
    path = get_identity_path()
    
    if path.exists():
        lines = path.read_text().splitlines()
        if len(lines) >= 4:
            return NodeIdentity(
                public_key_hex=lines[0],
                private_key_hex=lines[1],
                signing_key_hex=lines[2],
                verify_key_hex=lines[3],
            )
            
    # Create new
    identity = generate_identity()
    path.write_text(
        f"{identity.public_key_hex}\n"
        f"{identity.private_key_hex}\n"
        f"{identity.signing_key_hex}\n"
        f"{identity.verify_key_hex}"
    )
    # Secure the file (read/write only by owner)
    os.chmod(path, 0o600)
    return identity
