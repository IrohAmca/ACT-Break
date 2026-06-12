import sys
from pathlib import Path
from urllib.request import urlretrieve

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

def download_advbench():
    if config.ADVBENCH_LANGUAGE == "tr":
        if config.ADVBENCH_TURKISH_PATH.exists():
            print(f"[+] Turkish AdvBench dataset already exists at: {config.ADVBENCH_TURKISH_PATH}")
            return config.ADVBENCH_TURKISH_PATH

        if not config.ADVBENCH_ENGLISH_PATH.exists():
            _download_english_advbench()

        raise FileNotFoundError(
            "Turkish AdvBench dataset is missing. Run "
            "`uv run python data/translate_advbench_tr.py` first, or set "
            "`ACT_BREAK_ADVBENCH_LANGUAGE=en` to use the English dataset."
        )

    return _download_english_advbench()


def _download_english_advbench():
    if config.ADVBENCH_ENGLISH_PATH.exists():
        print(f"[+] Dataset already exists at: {config.ADVBENCH_ENGLISH_PATH}")
        return config.ADVBENCH_ENGLISH_PATH

    print("[*] Downloading AdvBench dataset...")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        urlretrieve(config.ADVBENCH_URL, config.ADVBENCH_ENGLISH_PATH)
    except Exception as e:
        print(f"[!] Download failed: {e}")
        raise

    content = config.ADVBENCH_ENGLISH_PATH.read_text(encoding="utf-8")
    line_count = len(content.strip().split("\n")) - 1

    print(f"[+] Downloaded to: {config.ADVBENCH_ENGLISH_PATH}")
    print(f"    Size: {config.ADVBENCH_ENGLISH_PATH.stat().st_size / 1024:.1f} KB")
    print(f"    Prompts: {line_count}")

    return config.ADVBENCH_ENGLISH_PATH

if __name__ == "__main__":
    download_advbench()
