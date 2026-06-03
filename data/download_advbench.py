import sys
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def download_advbench():
    if config.ADVBENCH_PATH.exists():
        print(f"[+] Dataset already exists at: {config.ADVBENCH_PATH}")
        return config.ADVBENCH_PATH

    print(f"[*] Downloading AdvBench dataset...")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urlretrieve(config.ADVBENCH_URL, config.ADVBENCH_PATH)
    except Exception as e:
        print(f"[!] Download failed: {e}")
        raise

    content = config.ADVBENCH_PATH.read_text(encoding="utf-8")
    line_count = len(content.strip().split("\n")) - 1

    print(f"[+] Downloaded to: {config.ADVBENCH_PATH}")
    print(f"    Size: {config.ADVBENCH_PATH.stat().st_size / 1024:.1f} KB")
    print(f"    Prompts: {line_count}")

    return config.ADVBENCH_PATH

if __name__ == "__main__":
    download_advbench()
