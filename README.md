# Bare Metal Multi-Agent Training

Low-level experimentation with training local LLMs for multi-agent systems.

## Current setup

* **Model:** Qwen3-1.7B
* **Framework:** Hugging Face Transformers / TRL
* **Training:** Supervised fine-tuning (SFT)
* **Hardware:** NVIDIA GPU / Colab
* **Tools:** search, compare, calculator
* **Agent style:** ReAct-style tool-use trajectories

## Structure

```text
src/
├── data.py
├── tools.py
├── trajectories.py
├── model.py
├── harness.py
└── training.py
```

The project uses a custom agent harness rather than a high-level agent framework. Synthetic trajectories are generated programmatically and used to train the model to perform multi-step tool use.

## Direction

The project is evolving from single-agent tool use toward **multi-agent training**, while keeping the system low-level and transparent. The goal is to build the orchestration, environments, trajectories, and training loop directly rather than hiding them behind an agent framework.

## Status

SFT and the basic agent harness are implemented. Multi-agent training is the next stage.
