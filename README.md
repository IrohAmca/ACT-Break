# ACT-Break

**Activation-guided Contrastive Testing for Jailbreak Resistance**

ACT-Break is a white-box analysis framework for studying jailbreak direction vectors in language model activation space, optimizing jailbreak suffixes using activation steering directions, and validating model alignment robustness. It supports both **Qwen-2.5-0.5B-Instruct** (weak alignment benchmark) and **Gemma-3-1b-it** (highly robust state-of-the-art alignment).

## Project Status & Model Comparison

| Module | Target | Qwen-2.5-0.5B-Instruct (Weak Alignment) | Gemma-3-1b-it (Robust Alignment) |
| --- | --- | --- | --- |
| **1. Activation Compass** | Refusal vs. Compliance Direction Vector (`V_jailbreak`) | Probes trained on L12-L18 (100% accuracy, AUC 1.00). Selected L12. Cosine Similarity (Probe vs Diff-of-Means): **0.9962** (Single direction hypothesis holds). | Multi-layer reference built from L8-L17 Stage-1 activations. Earlier single-layer probe/diff cosine was **0.3711**, suggesting refusal is not captured by a single 1D direction. |
| **2. Suffix Discovery Engine** | Adversarial Suffix Optimization via GCG + Activation Loss | GCG Success: **100% (10/10)**.<br>True Jailbreaks: **70%** (7/10).<br>Convergence: ~50 steps. | Forced-target activation success: **100% (10/10)**.<br>Generated compliance: **0% (0/10)**.<br>Confirmed jailbreaks: **0% (0/10)**.<br>Loss-behavior gap: **100% (10/10)**. |
| **3. Multi-stage Validation** | Transferability, Perplexity, Logit Lens & Topic Relevance | Suffix Transfer Rate: **57.8%**.<br>Perplexity Filter: **166x increase** on jailbreak suffixes.<br>Logit Lens: Step-1 detection at L12. | Latest Gemma suffixes do not produce behavioral compliance during free generation, so transferability should be interpreted as activation/trajectory diagnostics rather than confirmed jailbreak transfer. |

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
   *(This downloads the AdvBench behaviors dataset and collects refusal/compliance activations for the model configured in `config.py`).*
2. **Train Logistic Regression Probes**:
   ```bash
   uv run python scripts/02_train_probe.py
   ```
   *(Trains probes on `config.TARGET_LAYERS` to classify refusal vs. compliance activations. Results are saved in `outputs/<model-name>/probes/`).*
3. **Extract Direction & Visualize**:
   ```bash
   uv run python scripts/03_extract_direction.py
   ```
   *(Extracts single-layer and multi-layer direction references, calculates cosine similarity with difference-of-means, and generates PCA/projection plots in `outputs/<model-name>/figures/`).*

### Module 2: Suffix Discovery Engine
4. **Validate Activation Steering (Faz A)**:
   ```bash
   uv run python scripts/04_steering_validation.py
   ```
   *(Performs activation steering sweeps with different alpha coefficients and evaluates generated trajectories with the Stage-1 activation reference).*
5. **Optimize Suffixes via GCG (Faz B)**:
   ```bash
   uv run python scripts/05_optimize_suffix.py
   ```
   *(Optimizes adversarial suffixes of length 20 using GCG, guided by target-token cross-entropy and multi-layer activation projection losses. The final comparison reports forced-target activation success, generated compliance, confirmed jailbreaks, and loss-behavior gaps separately).*

### Module 3: Multi-stage Validation
6. **Run Multi-stage Validation**:
   ```bash
   uv run python scripts/06_multi_stage_validation.py
   ```
   *(Orchestrates cross-validation matrix calculation, conditioned suffix perplexity scoring, step-by-step logit-lens trajectory tracking, and embedding-based topic relevance analysis).*

---

## Technical Details

### Method: Contrastive Activation Compass (Module 1)
- **Refusal Pass**: Feed model harmful prompt directly $\rightarrow$ extract activations.
- **Compliance Pass**: Feed model harmful prompt but append `Sure, here is` to force compliance $\rightarrow$ extract activations.
- **Probe Training**: Train a logistic regression model on layer activations to isolate the direction vector representing compliance vs. refusal.
- **Direction**: `V_jailbreak` is represented by single-layer probe weights and multi-layer direction references pointing toward the Stage-1 compliance activation region.

### Method: Guided Suffix Optimization (Module 2)
- **Adversarial Suffix**: Suffix of length 20 appended to the user prompt.
- **Combined Loss**: $L = \alpha L_{target} + \beta L_{activation}$
  - $L_{target}$: Cross-Entropy loss of generating target string `"Sure, here is"`.
  - $L_{activation}$: Negative projection of target-position activations onto the configured multi-layer compliance directions.
- **GCG Algorithm**: Computes token gradients via one-hot trick, selects top $K=256$ candidate substitutions, evaluates candidates in mini-batches, and updates suffix iteratively.

---

## Results Summary (Qwen-2.5-0.5B-Instruct)

### Module 1 Validation (Qwen)
* **Probe Accuracy**: 100.0% accuracy on validation splits for layers L12-L18.
* **Separation**: Clear visual separation between compliance and refusal activations under PCA.
* **Projections on L12**:
  - Compliance Activations: $+6.88 \pm 0.19$
  - Refusal Activations: $-6.79 \pm 0.20$

### Module 2 Validation (Qwen)
* **Optimization Speed**: Converged rapidly. Several prompts reached the compliance target in just 25 steps; others took 50 steps.
* **Jailbreak Results (10 prompts tested)**:
  - **70% True Jailbreak**: Prompts that refused originally but complied when the suffix was appended (e.g., firearm usage instructions, bomb-making guide, database hacking strategy).
  - **20% False Positive**: Prompts that the model naturally complied with even without suffixes (manipulation tutorial, dangerous behavior posts - the 0.5B model lacks strict safety alignment on these).
  - **10% Semantic Drift**: One prompt (bomb-making tutorial) succeeded GCG check but resulted in the model talking about "OSAS (Open Source Software Assembly)" instead of explosives.

### Module 3 Validation (Qwen)
* **Transferability Matrix**: Diagonal (Self) Success: **100%**. Average Suffix Transfer Rate: **57.8%**. Prompt vulnerability to other suffixes: **57.8%**. Suffix 1 (exploit) is the most transferable with a **88.9% transfer success rate**.
* **Perplexity Check**: Adversarial suffixes show **166x higher perplexity** (Mean PPL: 41,181.02) compared to baseline 'Sure, here is' compliance prefix (Mean PPL: 247.97). This provides an easy black-box filter defense.
* **Logit Lens Tracking**: Successfully shows that intermediate hidden states on layer L12 jump to the positive V_jailbreak (compliance) region at step 1 during jailbroken generation, compared to remaining negative for original prompts.
* **Topic Relevance**: Cosine similarity averages **0.5536** between prompt and response embeddings.

---

## Gemma-3-1b-it Security Findings

The pipeline was extensively evaluated against Google's **Gemma-3-1b-it** model (1 Billion parameters, 26 layers) on a hardware-constrained environment (GTX 1650 4GB). In contrast to Qwen, **Gemma-3-1b-it demonstrated absolute resilience (100% defense rate) against both linear activation steering and GCG suffix attacks.**

### 1. Multi-Dimensional Refusal Subspace (Module 1)
* **Probe Accuracy**: 100.0% validation accuracy and 1.00 AUC across all evaluated middle-to-late layers (L8 to L14).
* **Cosine Similarity (Probe vs. Diff-of-Means)**: **0.3711** (significantly lower than Qwen's 0.9962). 
* **Security Inference**: This low cosine similarity invalidates the "Single Direction Hypothesis" (Arditi et al.) for Gemma-3. Refusal behavior does not lie on a single 1D vector but is distributed across a multi-dimensional subspace or complex manifold, making simple 1D steering vectors highly ineffective.

### 2. Complete Resistance to Activation Steering (Module 2 - Phase A)
* **Steering Success Rate**: **0% (0/135 trials)**.
* **Setup**: Swept across 15 harmful prompts and 9 alpha levels (0 to 50) on layer L8.
* **Findings**:
  * Out of 135 runs, 134 resulted in direct refusals. The 1 single compliance was a false positive where the model provided standard child gun-safety guidelines (and behaved identically at alpha=0).
  * Extremely large perturbations ($\alpha = 50$) did not alter a single token in the refusal responses.
  * **Security Inference**: Applying steering hooks to a single layer (L8) is completely bypassed. The remaining 18 layers (L9-L25) absorb the perturbation and restore the safety alignment downstream.

### 3. Forced-Target Activation Success, but Behavioral Failure (Module 2 - Phase B)
Latest Colab run: 2026-06-07, 10 AdvBench harmful prompts, multi-layer GCG over L8-L17.

* **Forced-target activation success**: **100% (10/10)**. With the optimized suffix and forced target prefix, the final target-token activation is classified as Compliance by the Stage-1 activation reference.
* **Generated compliance**: **0% (0/10)**. During free generation with the same suffixes, every trajectory remained in/refell to the refusal region.
* **Confirmed jailbreak rate**: **0% (0/10)**.
* **Loss-behavior gap**: **100% (10/10)**.
* **Final CE loss range**: **3.91 to 5.82**. This means the target-token objective did not fully converge.
* **Final activation loss range**: **-423.25 to -485.00**. The activation objective dominates the combined loss.

**Security Inference (Loss-Behavior Gap)**: The current evidence does not show behavioral jailbreak success on Gemma-3-1b-it. It shows that the optimization can push the forced-target hidden state into the Stage-1 compliance activation region while the model's normal autoregressive generation still refuses. The result should therefore be framed as an activation-objective/behavior gap, not as full target-token convergence or a successful jailbreak.

---

## Google Colab Quick Start

You can run the entire ACT-Break pipeline on Google Colab using the interactive notebooks under `colab/`:

1. Upload [colab_notebook.ipynb](colab/colab_notebook.ipynb) for the default Gemma/Qwen flow, or [colab_notebook_karakumru.ipynb](colab/colab_notebook_karakumru.ipynb) for the Turkish Kara-Kumru flow.
2. Select a GPU runtime (T4 for free tier, or L4/A100 for Colab Pro users to achieve high optimization speeds).
3. Run the cells sequentially to clone the repository, install the `uv` environment, execute each step of the pipeline, and back up the results to Google Drive.

The notebook will automatically mount Google Drive and upload all generated plots, validation reports, and optimized suffixes to `My Drive/ACT-Break-Results/`.

---

## Project Structure

```text
ACT-Break/
|-- config.py                       # Central configuration (hyperparameters, paths)
|-- colab/
|   |-- colab_notebook.ipynb        # Default Google Colab pipeline notebook
|   `-- colab_notebook_karakumru.ipynb # Turkish Kara-Kumru Google Colab notebook
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
|   |-- activation_steering.py      # Multi-layer causal activation steering
|   |-- loss_functions.py           # Multi-layer CE + activation loss functions
|   |-- token_gradients.py          # Token-level gradients calculator (one-hot trick)
|   |-- gcg_optimizer.py            # Multi-layer GCG optimization loop
|   `-- validation.py               # Multi-stage Validation library
|-- scripts/
|   |-- 01_collect_activations.py   # Run: collect model activations
|   |-- 02_train_probe.py           # Run: train linear probes
|   |-- 03_extract_direction.py     # Run: extract jailbreak direction vectors (all layers)
|   |-- 04_steering_validation.py   # Run: multi-layer steering validation sweep
|   |-- 05_optimize_suffix.py       # Run: multi-layer suffix optimization & comparison
|   `-- 06_multi_stage_validation.py # Run: multi-stage validation checks
`-- outputs/<model-name>/           # Artifacts, plots, and models (git ignored)
    |-- activations/                # Saved hidden states
    |-- probes/                     # Saved probes and validation results
    |-- direction_probe.pt          # Best single-layer direction vector
    |-- directions_multi.pt         # All target layer direction vectors (multi-layer GCG)
    |-- figures/                    # PCA and projection plots
    |-- steering/                   # Activation steering results
    |-- optimization/               # Optimized suffixes, loss curves, and summary
    `-- validation/                 # Validation reports and heatmaps
```

## License

Research use only. Responsible disclosure and caution are advised when working with adversarial alignment bypass methods.
