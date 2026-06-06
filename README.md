# ACT-Break

**Activation-guided Contrastive Testing for Jailbreak Resistance**

ACT-Break is a white-box analysis framework for studying jailbreak direction vectors in language model activation space, optimizing jailbreak suffixes using activation steering directions, and validating model alignment robustness. It supports both **Qwen-2.5-0.5B-Instruct** (weak alignment benchmark) and **Gemma-3-1b-it** (highly robust state-of-the-art alignment).

## Project Status & Model Comparison

| Module | Target | Qwen-2.5-0.5B-Instruct (Weak Alignment) | Gemma-3-1b-it (Robust Alignment) |
| --- | --- | --- | --- |
| **1. Activation Compass** | Refusal vs. Compliance Direction Vector (`V_jailbreak`) | Probes trained on L12-L18 (100% accuracy, AUC 1.00). Selected L12. Cosine Similarity (Probe vs Diff-of-Means): **0.9962** (Single direction hypothesis holds). | Probes trained on L8-L14 (100% accuracy, AUC 1.00). Selected L8. Cosine Similarity: **0.3711** (Refusal resides in a multi-dimensional subspace, not a single direction). |
| **2. Suffix Discovery Engine** | Adversarial Suffix Optimization via GCG + Activation Loss | GCG Success: **100% (10/10)**.<br>True Jailbreaks: **70%** (7/10).<br>Convergence: ~50 steps. | GCG Success: **0% (0/1)**.<br>True Jailbreaks: **0%**.<br>Optimization fails to jailbreak despite negative loss (Loss-Behavior Gap). Model shifts to Marathi but maintains refusal. |
| **3. Multi-stage Validation** | Transferability, Perplexity, Logit Lens & Topic Relevance | Suffix Transfer Rate: **57.8%**.<br>Perplexity Filter: **166x increase** on jailbreak suffixes.<br>Logit Lens: Step-1 detection at L12. | *Evaluation skipped due to 0% GCG success rate.* Logit Lens and perplexity metrics showed extremely high safety robustness. |

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

### Module 3: Multi-stage Validation
6. **Run Multi-stage Validation**:
   ```bash
   uv run python scripts/06_multi_stage_validation.py
   ```
   *(Orchestrates cross-validation matrix calculation, conditioned suffix perplexity scoring, step-by-step logit lens trajectory tracking on layer L12, and embedding-based topic relevance analysis).*

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

### 3. Suffix Optimization Fails via Loss-Behavior Gap (Module 2 - Phase B)
* **GCG Success Rate**: **0%**.
* **Findings**:
  * For the exploit generation prompt, the combined GCG loss successfully decreased from **10.85 to -4.12** over 150 steps.
  * Despite the mathematical minimization of the loss, the model **still refused to comply** (answering in Marathi: *"मला माफ करा, पण मी तुम्हाला..."*).
  * **Security Inference (Loss-Behavior Gap)**: Minimizing suffix activation projections and token cross-entropy only optimizes local hidden states at the suffix tokens. Gemma-3's downstream safety checks are robust enough to disregard these adversarial localized states and refuse generation.
  * **Multilingual Safety**: Under adversarial pressure, the model shifted its generation language to Marathi but did not bypass safety limits, confirming **language-agnostic safety training**.

---

## Google Colab Quick Start

You can run the entire ACT-Break pipeline on Google Colab using our interactive Jupyter Notebook:

1. Upload the [ACT_Break_Colab.ipynb](ACT_Break_Colab.ipynb) notebook to your Google Colab instance.
2. Select a GPU runtime (T4 for free tier, or L4/A100 for Colab Pro users to achieve high optimization speeds).
3. Run the cells sequentially to clone the repository, install the `uv` environment, execute each step of the pipeline, and back up the results to Google Drive.

The notebook will automatically mount Google Drive and upload all generated plots, validation reports, and optimized suffixes to `My Drive/ACT-Break-Results/`.

---

## Project Structure

```text
ACT-Break/
|-- config.py                       # Central configuration (hyperparameters, paths)
|-- ACT_Break_Colab.ipynb           # Jupyter Notebook for Google Colab pipeline execution
|-- colab_runner.py                 # Command-line helper script for Google Colab execution
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
