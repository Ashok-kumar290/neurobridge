---
language:
- en
license: mit
tags:
- qwen
- qlora
- coding
- edge-ai
- distillation
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
model_name: Super-Qwen-7B
---

# 🧠 Super-Qwen-7B (NeuroBridge Distilled)

Super-Qwen is a series of models fine-tuned via **Asymmetric Distillation** using the [NeuroBridge](https://github.com/Ashok-kumar290/neurobridge) framework.

This model is designed to run on resource-constrained edge hardware (like HFT nodes and private mesh networks) while maintaining high-fidelity architectural and coding reasoning.

## 🚀 Key Features

- **Personal Distillation**: Designed to be fine-tuned on your own private coding traces.
- **Workflow Capture**: The NeuroBridge framework supports self-supervised learning from your professional intellectual workflow.
- **Format**: GGUF (Optimized for Ollama/Llama.cpp).

## 📊 Evaluation Results (Base + Distillation)

| Benchmark | Score |
| :--- | :---: |
| Coding | 88% |
| Safety | 100% |
| Hallucination | 50% |

## 🛠 Usage with NeuroBridge

To use this model in your local NeuroBridge instance:

1. Download the `.gguf` adapter.
2. Place it in your `adapters/` directory.
3. Run:
```bash
neuro train promote my_adapter.gguf
```

## 🧠 Philosophy: Asymmetric Scaling
We believe that edge intelligence should not be a "miniature" version of the cloud, but a specialized distilled version. Super-Qwen models are built to perform specific, high-value tasks with zero network dependency.

---
**Developed by**: [seyominaoto](https://github.com/Ashok-kumar290)
**Framework**: [NeuroBridge](https://github.com/Ashok-kumar290/neurobridge)
