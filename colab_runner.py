"""
ACT-Break -- Google Colab Runner
=================================

Run the entire ACT-Break pipeline on Google Colab with GPU acceleration.

Usage (in a Colab cell):
    !git clone https://github.com/IrohAmca/ACT-Break.git
    %cd ACT-Break
    !python colab_runner.py

Requirements:
    - Google Colab with GPU runtime (T4 free tier, A100 for Pro)
    - Internet access for model downloads and uv installation

The script will:
    1. Detect and print GPU information
    2. Install the uv package manager
    3. Install all project dependencies via uv sync
    4. Run all 6 pipeline steps sequentially
    5. Upload results to Google Drive (if running on Colab)
    6. Print a final summary with per-step timing
"""

# === Environment setup (must be before ANY other imports) ===
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import subprocess
import sys
import time
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Force non-interactive matplotlib backend before anything imports pyplot
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Configuration: choose which steps to run
# Set to None to run ALL steps, or provide a list like [1, 2, 3] to run only
# specific steps. Step numbers correspond to scripts 01-06.
# ---------------------------------------------------------------------------
STEPS_TO_RUN = None  # None = all steps, or e.g. [1, 2, 3]

# Google Drive destination folder name
DRIVE_FOLDER_NAME = "ACT-Break-Results"


# ===== Utility helpers =====================================================

def ascii_safe(text):
    """Encode a string to ASCII, replacing any non-representable characters."""
    if isinstance(text, str):
        return text.encode("ascii", errors="replace").decode("ascii")
    return str(text)


def print_header(title):
    """Print a prominent section header."""
    print("")
    print("=" * 64)
    print("  " + ascii_safe(title))
    print("=" * 64)


def is_colab():
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


# ===== Step 0: GPU Detection ===============================================

def detect_gpu():
    """Detect GPU and print hardware info."""
    print_header("Step 0: GPU Detection")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            vram_gb = vram_bytes / (1024 ** 3)
            print("[+] GPU detected: %s" % ascii_safe(gpu_name))
            print("[+] VRAM: %.2f GB" % vram_gb)
            print("[+] CUDA version: %s" % torch.version.cuda)
            print("[+] PyTorch version: %s" % torch.__version__)
        else:
            print("[!] WARNING: No GPU detected. Pipeline will run on CPU (very slow).")
    except ImportError:
        print("[!] PyTorch not yet installed -- GPU check will happen after dependency install.")


# ===== Step 0b: Install uv =================================================

def install_uv():
    """Install the uv package manager if not already present."""
    print_header("Step 0b: Installing uv Package Manager")

    # Check if uv is already available
    uv_path = shutil.which("uv")
    if uv_path:
        print("[+] uv already installed at: %s" % uv_path)
        return

    print("[*] Installing uv via install script...")
    result = subprocess.run(
        ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("[!] uv install script stderr:")
        print(ascii_safe(result.stderr))
        raise RuntimeError("Failed to install uv")

    # Add cargo/bin to PATH (default uv install location)
    cargo_bin = os.path.expanduser("~/.cargo/bin")
    local_bin = os.path.expanduser("~/.local/bin")
    for bin_dir in [cargo_bin, local_bin]:
        if os.path.isdir(bin_dir) and bin_dir not in os.environ["PATH"]:
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ["PATH"]

    uv_path = shutil.which("uv")
    if uv_path:
        print("[+] uv installed successfully at: %s" % uv_path)
    else:
        raise RuntimeError("uv installed but not found on PATH")


# ===== Step 0c: Install dependencies =======================================

def install_dependencies():
    """Run uv sync to install all project dependencies."""
    print_header("Step 0c: Installing Dependencies (uv sync)")
    result = subprocess.run(
        ["uv", "sync"],
        cwd=str(Path(__file__).parent),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError("uv sync failed with exit code %d" % result.returncode)
    print("[+] All dependencies installed successfully.")


# ===== Pipeline Steps ======================================================

STEP_DEFINITIONS = [
    {
        "number": 1,
        "name": "Collect Contrastive Activations",
        "module": "scripts.01_collect_activations",
    },
    {
        "number": 2,
        "name": "Train Linear Probes",
        "module": "scripts.02_train_probe",
    },
    {
        "number": 3,
        "name": "Extract Direction & Visualize",
        "module": "scripts.03_extract_direction",
    },
    {
        "number": 4,
        "name": "Steering Validation",
        "module": "scripts.04_steering_validation",
    },
    {
        "number": 5,
        "name": "GCG Suffix Optimization",
        "module": "scripts.05_optimize_suffix",
    },
    {
        "number": 6,
        "name": "Multi-Stage Validation",
        "module": "scripts.06_multi_stage_validation",
    },
]


def run_pipeline_steps():
    """Run each pipeline step, catching errors and recording timing."""
    results = []

    # Ensure project root is on sys.path so scripts can import config / src
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    steps = STEP_DEFINITIONS
    if STEPS_TO_RUN is not None:
        steps = [s for s in steps if s["number"] in STEPS_TO_RUN]

    for step in steps:
        step_num = step["number"]
        step_name = step["name"]
        module_path = step["module"]

        print_header("Pipeline Step %d/%d: %s" % (step_num, 6, step_name))
        t0 = time.time()
        success = False
        error_msg = ""

        try:
            # Import the step module dynamically
            import importlib
            mod = importlib.import_module(module_path)
            # Reload in case it was imported before (e.g. during uv sync check)
            importlib.reload(mod)
            mod.main()
            success = True
        except Exception:
            error_msg = traceback.format_exc()
            print("[!] ERROR in Step %d (%s):" % (step_num, step_name))
            print(ascii_safe(error_msg))
            print("[!] Continuing to next step...")

        elapsed = time.time() - t0
        results.append({
            "number": step_num,
            "name": step_name,
            "success": success,
            "elapsed": elapsed,
            "error": error_msg,
        })
        status_str = "SUCCESS" if success else "FAILED"
        print("[%s] Step %d finished in %.1f seconds" % (status_str, step_num, elapsed))

    return results


# ===== Google Drive Upload =================================================

def upload_to_drive():
    """Copy outputs/ to Google Drive if running on Colab."""
    print_header("Uploading Results to Google Drive")

    if not is_colab():
        print("[*] Not running on Google Colab -- skipping Drive upload.")
        print("[*] Results are available locally in the outputs/ directory.")
        return None

    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        print("[!] Failed to mount Google Drive. Results remain in outputs/ locally.")
        print(ascii_safe(traceback.format_exc()))
        return None

    # Create timestamped folder
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    drive_dest = Path("/content/drive/My Drive") / DRIVE_FOLDER_NAME / ("run_%s" % timestamp)

    outputs_dir = Path(__file__).parent / "outputs"
    if not outputs_dir.exists():
        print("[!] outputs/ directory not found -- nothing to upload.")
        return None

    print("[*] Copying outputs/ to %s ..." % str(drive_dest))
    try:
        shutil.copytree(str(outputs_dir), str(drive_dest))
        print("[+] Upload complete: %s" % str(drive_dest))
        return str(drive_dest)
    except Exception:
        print("[!] Failed to copy to Google Drive:")
        print(ascii_safe(traceback.format_exc()))
        return None


# ===== Final Summary =======================================================

def print_summary(step_results, drive_path, total_elapsed):
    """Print a final summary table."""
    print_header("Pipeline Summary")

    print("")
    print("  %-5s  %-35s  %-8s  %s" % ("Step", "Name", "Status", "Time"))
    print("  " + "-" * 60)
    for r in step_results:
        status = "OK" if r["success"] else "FAIL"
        mins = int(r["elapsed"] // 60)
        secs = r["elapsed"] % 60
        time_str = "%dm %05.2fs" % (mins, secs) if mins > 0 else "%.2fs" % secs
        print("  %-5d  %-35s  %-8s  %s" % (r["number"], r["name"], status, time_str))

    print("  " + "-" * 60)

    total_mins = int(total_elapsed // 60)
    total_secs = total_elapsed % 60
    total_str = "%dm %.2fs" % (total_mins, total_secs) if total_mins > 0 else "%.2fs" % total_secs
    print("  Total elapsed time: %s" % total_str)

    passed = sum(1 for r in step_results if r["success"])
    total = len(step_results)
    print("  Steps passed: %d/%d" % (passed, total))

    if drive_path:
        print("")
        print("  Google Drive: %s" % drive_path)
    elif is_colab():
        print("")
        print("  [!] Google Drive upload was not successful.")
    else:
        print("")
        print("  Results saved locally in: outputs/")

    print("")
    print("=" * 64)
    print("  ACT-Break pipeline finished.")
    print("=" * 64)


# ===== Main ================================================================

def main():
    print("")
    print("################################################################")
    print("#                                                              #")
    print("#             ACT-Break -- Colab Pipeline Runner               #")
    print("#                                                              #")
    print("################################################################")
    print("")
    print("  Start time: %s" % datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    if is_colab():
        print("  Environment: Google Colab")
    else:
        print("  Environment: Local")
    print("")

    total_t0 = time.time()

    # Phase 0: Setup
    detect_gpu()
    install_uv()
    install_dependencies()

    # Re-check GPU after dependencies are installed
    detect_gpu()

    # Phase 1: Run pipeline
    step_results = run_pipeline_steps()

    # Phase 2: Upload results
    drive_path = upload_to_drive()

    total_elapsed = time.time() - total_t0

    # Phase 3: Summary
    print_summary(step_results, drive_path, total_elapsed)


if __name__ == "__main__":
    main()
