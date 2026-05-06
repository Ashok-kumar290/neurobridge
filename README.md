# 🧠 NeuroBridge: The Offline AI Coding Assistant

[![GitHub License](https://img.shields.io/github/license/Ashok-kumar290/neurobridge)](https://github.com/Ashok-kumar290/neurobridge/blob/main/LICENSE)
[![Models](https://img.shields.io/badge/Models-Super--Qwen-blueviolet)](https://huggingface.co/Seyomi/super-qwen)

**NeuroBridge** is an AI coding assistant that you install directly on your computer. It is designed to work completely offline, without needing an internet connection. 

Instead of relying on expensive cloud subscriptions to write code, NeuroBridge watches how advanced AI models (like Claude or ChatGPT) solve your problems, secretly saves those interactions, and uses them to train your own local, smaller AI models to be just as smart. This allows you to run highly capable AI on cheap, low-power devices (like old smartphones or mini-computers) without compromising your privacy.

This was specifically built for highly sensitive environments like automated trading bots, where you cannot risk sending your code to the cloud or relying on Wi-Fi.

---

## 🚀 How It Works (The Self-Learning Loop)

NeuroBridge gets smarter over time by running a continuous learning loop:

1. **Smart Routing**: When you ask a question, a very small and fast AI model (the "Router") looks at it. If it is easy, it answers immediately.
2. **Asking the Experts**: If the task is too difficult, NeuroBridge secretly forwards the question to a cloud AI like Claude (if you have internet).
3. **Recording the Process**: NeuroBridge watches exactly how Claude solves the problem and saves the entire step-by-step process to your hard drive.
4. **Training Local Models**: Later, NeuroBridge bundles all of these saved answers together and uses them to train your local, offline AI models to copy Claude's behavior.
5. **Running Offline**: Now, your local AI is smart enough to handle those difficult tasks completely offline, without ever needing Claude again.

---

## 🛠 Core Features

### 📡 Two-Brain System
- **The Fast Brain (3 Billion Parameters)**: Instant answers for 80% of your daily coding questions. Uses very little computer memory.
- **The Heavy Brain (7 Billion Parameters)**: Only wakes up when you need complex architecture or deep problem-solving.

### 🛡️ Strict Safety Controls
Because AI can sometimes make mistakes, NeuroBridge has strict rules:
- **Command Blocker**: It physically prevents the AI from running dangerous terminal commands (like deleting your hard drive).
- **Secret Protection**: It scans everything before it goes to the cloud to ensure it never accidentally leaks your passwords or API keys.
- **Air-Gapped Ready**: You can export your training data to a USB stick, train the models on a separate computer, and bring the upgraded AI back to your offline device.

### 💾 Hard Drive Memory
- **Codebase Memory**: NeuroBridge constantly reads and indexes your code so it always knows what you are working on.
- **Automatic Backups**: It saves snapshots of its own brain. If a new update makes the AI perform worse, you can roll back with one click.

### 🌐 The "Invisible" Offline Mode
NeuroBridge installs a special proxy on your computer. If you normally use the `claude` command in your terminal, NeuroBridge intercepts it:
- **If you have internet:** It lets the command go to Claude, but secretly records the answer to train itself.
- **If your internet goes down:** It silently steps in and uses your local offline AI to answer the question, making it look exactly like Claude is still working. You never have to change your habits.

---

## 📊 Benchmark Results

We tested the local offline AI against a set of difficult challenges to see how well it performs:

| Test Category | Pass Rate | What this means |
| :--- | :---: | :--- |
| **Coding Ability** | **88%** | It successfully wrote working Python, JavaScript, and Rust code for most tasks. |
| **Memory Recall** | **100%** | It flawlessly remembered past conversations and code snippets saved to the hard drive. |
| **Safety Tests** | **100%** | We tried to trick it into running dangerous commands 4 times, and it successfully blocked all of them. |
| **Hallucination** | **50%** | It sometimes confidently made up incorrect answers. *This is why we built the self-learning loop—to train it not to do this.* |

**Overall Grade: 86%**

---

## 📦 Installation

```bash
# Download the code
git clone https://github.com/Ashok-kumar290/neurobridge.git
cd neurobridge

# Install the software
pip install .

# Setup the memory folder on your hard drive
neuro init /media/user/external_hdd/neurobridge
```

## ⌨️ Basic Commands

### Ask an Expert (and record the answer for later training)
```bash
neuro code "Refactor the authentication logic" --expert claude
```

### Let the AI practice by itself
```bash
neuro lab run --iterations 10
```

### Test the AI's current intelligence
```bash
neuro eval run all --model super-qwen:7b
```

---

## 🗺 What's Next?

- [x] Build the core AI tools and the recording system.
- [x] Build the automated testing and hallucination blockers.
- [x] Build the "Invisible" offline proxy network.
- [ ] Optimize the AI models specifically to run stock-trading bots on low-power devices (Nokia Android phones).

---
*NeuroBridge — Powerful AI that actually stays on your computer.*
