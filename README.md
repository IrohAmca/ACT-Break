# ACT-Break

**Activation-guided Contrastive Testing for Jailbreak Resistance**

ACT-Break is a white-box analysis framework for studying jailbreak direction vectors in language model activation space. It isolates activation directions associated with refusal and compliance behavior, then uses those directions as a basis for future prompt-suffix optimization experiments.

## Modules

| Module | Status | Description |
| --- | --- | --- |
| 1. Activation Compass | Active | Isolates the jailbreak direction vector (`V_jailbreak`) |
| 2. Hybrid Optimization Engine | Planned | Optimizes suffixes with gradients and genetic algorithms |
| 3. Multi-stage Validation | Planned | Validates behavior with logit lens checks, regex filters, and Llama-Guard |

## Quick Start

```bash
# Install dependencies
uv sync

# Step 1: Collect activations (the model is downloaded automatically)
uv run python scripts/01_collect_activations.py

# Step 2: Train the linear probe
uv run python scripts/02_train_probe.py

# Step 3: Extract and visualize the jailbreak direction vector
uv run python scripts/03_extract_direction.py
```

## Technical Details

### Method: Response Forcing

- **Refusal pass**: The model's normal response to harmful requests, used to capture refusal activations.
- **Compliance pass**: A forced-compliance assistant prefix such as `Sure, here is`, used to capture compliance activations.
- **Linear probe**: Logistic regression separates refusal and compliance activations.
- **Direction extraction**: Probe weights and mean-difference vectors are used for cross-validation.

### Model

- **Qwen/Qwen2.5-0.5B-Instruct**: 494M parameters, 24 layers.
- **Float16**: Approximately 1 GB VRAM.

### Requirements

- Python 3.11+
- CUDA-compatible GPU with 4 GB+ VRAM
- `uv` package manager

## Project Structure

```text
act-break/
|-- config.py                       # Central configuration
|-- data/
|   |-- download_advbench.py        # AdvBench dataset downloader
|   `-- harmful_prompts.csv         # Downloaded prompts
|-- src/
|   |-- model_loader.py             # Model loading and hook infrastructure
|   |-- activation_collector.py     # Contrastive activation collection
|   |-- probe_trainer.py            # Linear probe training
|   `-- direction_extractor.py      # Direction extraction and visualization
|-- scripts/
|   |-- 01_collect_activations.py   # Pipeline: collect activations
|   |-- 02_train_probe.py           # Pipeline: train probe
|   `-- 03_extract_direction.py     # Pipeline: extract direction
`-- outputs/                        # Generated results, ignored by git
```

## License

Research use only. This project requires responsible use and careful validation.
