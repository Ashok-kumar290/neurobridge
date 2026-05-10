# Phantom Network Protocol (v1.0)

The Phantom Network is the decentralized communication layer for NeuroBridge. It allows nodes (PCs, phones, embedded devices) to sync specialized coding adapters, share expert traces, and form a private mesh of intelligence.

## 1. Identity & Security
- **Encryption**: All P2P traffic is encrypted using `PyNaCl` (libsodium).
- **Node Keys**: Every node generates a unique Curve25519 keypair on first boot.
- **Authentication**: Handshakes involve a signed challenge-response to prevent impersonation.
- **Privacy**: No telemetry or metadata is sent to any central server.

## 2. Peer Discovery
- **Phase 1 (Manual)**: Nodes are linked via static IP/Port (e.g., `neuro network connect 192.168.1.50`).
- **Phase 2 (Mesh)**: Future implementation of a Lightweight DHT for automated local discovery.

## 3. Skill Sharing (Adapter Sync)
- **Problem**: QLoRA adapters can be 50MB - 200MB. Sending them over constrained networks requires chunking.
- **Protocol**:
    1. `SYNC_REQ`: Node A requests a specific adapter by ID.
    2. `SYNC_ACK`: Node B confirms it has the adapter and provides a BLAKE3 hash.
    3. `STREAM_CHUNKS`: The adapter is sent in 1MB encrypted blocks.
    4. `VERIFY`: Node A verifies the final file against the hash before registering it.

## 4. Trace Propagation
- **Decentralized Distillation**: When a node solves a difficult task using an Expert (Claude/Codex), it can broadcast the resulting **Expert Trace** to other nodes.
- **Impact**: One node pays for the Claude usage; the entire network learns from the result.

## 5. CLI Interface
```bash
neuro network start      # Activate the Node Daemon
neuro network ping <IP>  # Verify connection
neuro network status     # List connected peers
neuro network sync all   # Pull latest adapters from peers
```
