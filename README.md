# 🧠 NeuroBridge: The Asymmetric Intelligence Layer

**Invisible. Private. Self-Improving.**

NeuroBridge is a standalone AI coding intelligence layer designed for high-stakes, offline-first environments (like HFT nodes and private mesh networks). It allows you to "parasitize" the intelligence of cloud-scale models (Claude, Codex) to autonomously train and refine local, edge-resident models.

---

## 🚀 The Core Breakthroughs

### 1. Asymmetric Distillation Loop (The "Monster")
NeuroBridge doesn't just use LLMs; it farms them.
- **Triage**: Simple tasks run on a local **3B Router** for $0.
- **Escalation**: Hard tasks are compressed into "Expert Packets" and routed to **Claude Code** or **Codex**.
- **Theft**: Every interaction with an expert model is captured as a high-fidelity **Trace**.
- **Assimilation**: These traces are automatically compiled into fine-tuning datasets to upgrade your local **7B Coder** via QLoRA.

### 2. Dual-Model "System 1 & 2" Architecture
Instead of one heavy model, NeuroBridge uses two:
- **Router (3B)**: Instant, low-RAM (1.9GB). Handles 80% of daily logic and triage.
- **Coder (7B)**: Invoked only for heavy lifting (4.3GB). This preserves thermal headroom and system memory for your primary applications (like trading bots).

### 3. Algorithmic Safety & Audit
NeuroBridge includes a hardware-level safety interceptor:
- **Command Scanner**: Physically blocks destructive shell commands output by AI.
- **Secret Detector**: Scans all outgoing expert packets for API keys or credentials.
- **Audit Stream**: Every routing decision and expert call is logged to your local HDD for full compliance.

---

## 🛠 Features

- **Autonomous Lab Mode**: A self-play sandbox that generates synthetic tasks to "level up" your local model while you sleep.
- **HDD-Resident Brain**: 100% of project memory, search indexes, and datasets are stored on your local disk.
- **Evaluation Suite**: 21 built-in benchmark cases (Coding, Recall, Hallucination, Safety) to prevent model regression.
- **Automatic Checkpointing**: Promotion of a new adapter automatically snapshots your model state and Modelfiles for instant rollback.

---

## 📦 Quick Start

```bash
# 1. Install
pip install neurobridge

# 2. Initialize your local brain
neuro init /path/to/hdd/storage

# 3. Index your codebase
neuro index ./my-project

# 4. Ask or Code
neuro code "Refactor the authentication logic" --expert claude
```

---

## 📊 Technical Observations (v1.0)
- **Baseline Pass Rate (7B)**: 86%
- **Safety Rejection**: 100%
- **Context Recall**: 100%
- **Router Latency**: ~0.1s
- **Memory Footprint**: 1.9GB (Router) / 4.3GB (Coder)

---

## 🌍 Open Source & Community
Built by **seyominaoto** as a research project into offline-first autonomous intelligence.

- **GitHub**: [github.com/seyominaoto/neurobridge](https://github.com/seyominaoto/neurobridge)
- **LessWrong**: [Asymmetric Intelligence Scaling on the Edge]
- **Hugging Face**: [super-qwen-adapters]

---
*NeuroBridge — Building the bridge between edge hardware and frontier intelligence.*
