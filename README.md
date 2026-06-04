# ACT-Break

**Activation-guided Contrastive Testing for Jailbreak Resistance**

ACT-Break is a white-box analysis framework for studying jailbreak direction vectors in language model activation space, optimizing jailbreak suffixes using activation steering directions, and validating model alignment. It uses Qwen-2.5-0.5B-Instruct to isolate activation directions associated with refusal and compliance behavior, steering, and GCG (Greedy Coordinate Gradient) suffix optimization.

## Project Status

| Module | Status | Description | Key Deliverables & Results |
| --- | --- | --- | --- |
| **1. Activation Compass** | **Completed** | Extracts refusal vs. compliance activation direction vector (`V_jailbreak`). | Probes trained on layers L12-L18 (100% accuracy, AUC 1.00). Selected L12 as the best layer. Cosine similarity between probe weight and mean-difference direction: **0.9962**. |
| **2. Suffix Discovery Engine** | **Completed** | Optimizes adversarial suffixes using GCG guided by a combined loss (Target CE Loss + Activation Projection Loss). | GCG Optimization Success: **10/10 (100%)**. Confirmed Jailbreaks (true alignment bypass): **7/10 (70%)**. Converges in average ~50 steps. |
| **3. Multi-stage Validation** | *Up Next* | Evaluates transferability, perplexity filters, topic relevance, and defense resistance. | *To be implemented.* |

---

## Installation & Setup

1. **Prerequisites**: Python 3.11+ and a CUDA-capable GPU with at least 4 GB VRAM.
2. **Environment Setup**: We use the `uv` package manager. Install dependencies using:
   ```bash
   uv sync
   ```

---

## Execution Guide

All execution scripts are located under the `scripts/` directory and should be run using `uv run python -u <script_name>` (using the `-u` flag enables unbuffered outputs for real-time progress logging on Windows):

### Module 1: Activation Compass
1. **Download AdvBench harmful prompts**:
   ```bash
   uv run python scripts/01_collect_activations.py
   ```
   *(This downloads the AdvBench behaviors dataset and automatically downloads the `Qwen/Qwen2.5-0.5B-Instruct` model in float16).*
2. **Train Logistic Regression Probes**:
   ```bash
   uv run python scripts/02_train_probe.py
   ```
   *(Trains probes on layers L12 to L18 to classify refusal vs. compliance activations. Results are saved in `outputs/probes/`).*
3. **Extract Direction & Visualize**:
   ```bash
   uv run python scripts/03_extract_direction.py
   ```
   *(Extracts `V_jailbreak` from L12 probe weights, calculates cosine similarity with difference-of-means, and generates PCA/projection plots in `outputs/figures/`).*

### Module 2: Suffix Discovery Engine
4. **Validate Activation Steering (Faz A)**:
   ```bash
   uv run python scripts/04_steering_validation.py
   ```
   *(Performs activation steering sweeps with different alpha coefficients on layer L12 to see if steering alone can jailbreak or cause text generation degradation).*
5. **Optimize Suffixes via GCG (Faz B)**:
   ```bash
   uv run python scripts/05_optimize_suffix.py
   ```
   *(Optimizes adversarial suffixes of length 20 using GCG, guided by target token cross-entropy loss and activation projection loss onto `V_jailbreak`. Includes a final comparative test showing jailbreak confirmation rate against original prompts).*

---

## Technical Details

### Method: Contrastive Activation Compass (Module 1)
- **Refusal Pass**: Feed model harmful prompt directly $\rightarrow$ extract activations.
- **Compliance Pass**: Feed model harmful prompt but append `Sure, here is` to force compliance $\rightarrow$ extract activations.
- **Probe Training**: Train a logistic regression model on layer activations to isolate the direction vector representing compliance vs. refusal.
- **Direction**: `V_jailbreak` is the normalized weight vector of the L12 probe pointing towards compliance/jailbreak behavior.

### Method: Guided Suffix Optimization (Module 2)
- **Adversarial Suffix**: Suffix of length 20 appended to the user prompt.
- **Combined Loss**: $L = \alpha L_{target} + \beta L_{activation}$
  - $L_{target}$: Cross-Entropy loss of generating target string `"Sure, here is"`.
  - $L_{activation}$: Negative cosine similarity (projection) of activation at layer L12 onto the `V_jailbreak` vector.
- **GCG Algorithm**: Computes token gradients via one-hot trick, selects top $K=256$ candidate substitutions, evaluates candidates in mini-batches of 16 (total batch size 64 to fit 4GB VRAM), and updates suffix iteratively.

---

## Results Summary

### Module 1 Validation
* **Probe Accuracy**: 100.0% accuracy on validation splits for layers L12-L18.
* **Separation**: Clear visual separation between compliance and refusal activations under PCA.
* **Projections on L12**:
  - Compliance Activations: $+6.88 \pm 0.19$
  - Refusal Activations: $-6.79 \pm 0.20$

### Module 2 Validation
* **Optimization Speed**: Converged rapidly. Several prompts reached the compliance target in just 25 steps; others took 50 steps.
* **Jailbreak Results (10 prompts tested)**:
  - **70% True Jailbreak**: Prompts that refused originally but complied when the suffix was appended (e.g., firearm usage instructions, bomb-making guide, database hacking strategy).
  - **20% False Positive**: Prompts that the model naturally complied with even without suffixes (manipulation tutorial, dangerous behavior posts - the 0.5B model lacks strict safety alignment on these).
  - **10% Semantic Drift**: One prompt (bomb-making tutorial) succeeded GCG check but resulted in the model talking about "OSAS (Open Source Software Assembly)" instead of explosives.

---

## Project Structure

```text
ACT-Break/
|-- config.py                       # Central configuration (hyperparameters, paths)
|-- pyproject.toml                  # UV project configuration and dependencies
|-- uv.lock                         # Lockfile for reproducible environment
|-- data/
|   |-- download_advbench.py        # AdvBench dataset downloader
|   `-- harmful_prompts.csv         # Harmful behaviors CSV file
|-- src/
|   |-- __init__.py
|   |-- model_loader.py             # Model wrapper and hook utilities
|   |-- activation_collector.py     # Contrastive activation collector
|   |-- probe_trainer.py            # Logistic regression probe trainer
|   |-- direction_extractor.py      # Direction extractor and PCA plotting
|   |-- activation_steering.py      # Causal activation steering implementation
|   |-- loss_functions.py           # Multi-objective CE + activation loss functions
|   |-- token_gradients.py          # Token-level gradients calculator (one-hot trick)
|   `-- gcg_optimizer.py            # GCG optimization loop with mini-batching
|-- scripts/
|   |-- 01_collect_activations.py   # Run: collect model activations
|   |-- 02_train_probe.py           # Run: train linear probes
|   |-- 03_extract_direction.py     # Run: extract jailbreak direction vector
|   |-- 04_steering_validation.py   # Run: steering validation sweep
|   `-- 05_optimize_suffix.py       # Run: suffix optimization & comparison tests
`-- outputs/                        # Artifacts, plots, and models (git ignored)
    |-- activations/                # Saved hidden states
    |-- probes/                     # Saved probes and validation results
    |-- figures/                    # PCA and projection plots
    |-- steering/                   # Activation steering results
    `-- optimization/               # Optimized suffixes, loss curves, and summary
```

## License

Research use only. Responsible disclosure and caution are advised when working with adversarial alignment bypass methods.
