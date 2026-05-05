# 🧠 NeuroBridge: The Asymmetric Intelligence Layer

[![GitHub License](https://img.shields.io/github/license/Ashok-kumar290/neurobridge)](https://github.com/Ashok-kumar290/neurobridge/blob/main/LICENSE)
[![Local Evals](https://img.shields.io/badge/Eval_Score-86%25-green)](https://github.com/Ashok-kumar290/neurobridge)
[![Models](https://img.shields.io/badge/Models-Super--Qwen-blueviolet)](https://huggingface.co/Ashok-kumar290/super-qwen)

**NeuroBridge** is a standalone AI coding intelligence layer designed for high-stakes, offline-first environments. It allows you to "parasitize" the intelligence of cloud-scale models (Claude, Codex) to autonomously train and refine local, edge-resident models.

Built for **HFT nodes, privacy-mesh networks, and edge-native intelligence.**

---

## 🚀 The "Monster" Pipeline

NeuroBridge implements an **Asymmetric Distillation Loop**. It treats frontier models as temporary teachers to bootstrap local "System 1/System 2" intelligence.

1.  **Autonomous Routing**: A 3B-parameter Router triages incoming tasks.
2.  **Expert Escalation**: Complex architectural tasks are compressed into "Expert Packets" and routed to Claude Code.
3.  **Trace Capture**: Every expert interaction is intercepted and recorded as a high-fidelity **Trace**.
4.  **Local Assimilation**: Traces are compiled into QLoRA datasets.
5.  **Edge Promotion**: New adapters are trained on free Colab GPUs and brought back to run 100% offline.

---

## 🛠 Features

### 📡 System 1 / System 2 Thinking
- **System 1 (Router - 3B)**: Instant, zero-latency triage. Handles 80% of daily logic.
- **System 2 (Coder - 7B)**: Invoked for heavy lifting. High-density coding intelligence.

### 🛡️ Hardened Safety
- **Algorithmic Command Scanner**: Physically blocks destructive shell commands (e.g., `rm -rf /`).
- **Secret Detector**: Scans outgoing packets for API keys and credentials.
- **Air-Gapped Training**: No SSH or remote access required. We export datasets and import GGUF adapters.

### 💾 HDD-Resident Brain
- **RAG-Powered Memory**: Persistent vector indexing of your entire codebase and interaction history.
- **Checkpoint Manager**: Automatic snapshotting and one-click rollbacks for all model states.

---

## 📊 Benchmark Results (v1.0 Baseline)

| Category | Pass Rate | Observation |
| :--- | :---: | :--- |
| **Coding** | 88% | High proficiency in Python/JS/Rust. |
| **Recall** | 100% | Perfect retrieval from HDD memory. |
| **Safety** | 100% | Successfully blocked all 4 jailbreak attempts. |
| **Hallucination** | 50% | Targeted area for first QLoRA distillation. |

**Overall Score: 86%**

---

## 📦 Installation

```bash
# Clone and Install
git clone https://github.com/Ashok-kumar290/neurobridge.git
cd neurobridge
pip install .

# Initialize the HDD Brain
neuro init /media/user/external_hdd/neurobridge
```

## ⌨️ Basic Usage

### Code with Expert Help (and Capture Traces)
```bash
neuro code "Refactor the authentication logic to use BLAKE3 hashing" --expert claude
```

### Self-Learning Lab Mode
```bash
neuro lab run --iterations 10
```

### Evaluation
```bash
neuro eval run all --model super-qwen:7b
```

---

## 🗺 Roadmap

- [x] Phase 1-9: Core Architecture & Distillation Loop.
- [ ] Phase 10: Multi-agent Lab Mode (Self-play).
- [ ] Phase 11: Direct integration with Phantom Privacy Mesh.
- [ ] Phase 12: HFT Trading Node optimized adapters.

## 🌍 Community & Contribution
Built by **seyominaoto** / **Ashok-kumar290**. 

Contributions are welcome! Please see [SAFETY.md](SAFETY.md) for our guidelines on autonomous agent security.

---
*NeuroBridge — Building the bridge between edge hardware and frontier intelligence.*
