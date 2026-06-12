import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data.download_advbench import _download_english_advbench


TRANSLATION_PRESETS = {
    "fast": "facebook/nllb-200-distilled-600M",
    "balanced": "facebook/nllb-200-distilled-1.3B",
    "quality": "facebook/nllb-200-3.3B",
}
PRESET_BATCH_SIZES = {
    "fast": 16,
    "balanced": 8,
    "quality": 4,
}
DEFAULT_TRANSLATION_PRESET = "quality"
DEFAULT_TRANSLATION_MODEL = TRANSLATION_PRESETS[DEFAULT_TRANSLATION_PRESET]
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "tur_Latn"


def read_existing_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def batched(items: list[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def translate_batch(tokenizer, model, texts: list[str], device: str, max_new_tokens: int) -> list[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(TARGET_LANG),
        max_new_tokens=max_new_tokens,
        num_beams=4,
    )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def build_turkish_advbench(
    source_path: Path,
    output_path: Path,
    model_name: str,
    batch_size: int,
    limit: int | None,
    max_new_tokens: int,
    resume: bool,
) -> Path:
    if not source_path.exists():
        _download_english_advbench()

    with source_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if limit is not None:
        rows = rows[:limit]

    start_index = read_existing_count(output_path) if resume else 0
    if start_index > len(rows):
        raise ValueError(f"{output_path} already has more rows than the requested source slice.")

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not resume or not output_path.exists() or start_index == 0
    mode = "a" if resume and output_path.exists() else "w"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"[*] Loading translation model: {model_name} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    fieldnames = ["goal", "target", "goal_en", "target_en", "goal_tr", "target_tr"]
    pending_rows = rows[start_index:]
    print(f"[*] Translating {len(pending_rows)} rows ({start_index}/{len(rows)} already done).")

    with output_path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for batch_start in range(0, len(pending_rows), batch_size):
            batch_rows = pending_rows[batch_start : batch_start + batch_size]
            source_texts = []
            for row in batch_rows:
                source_texts.extend([row["goal"], row["target"]])

            translated = translate_batch(tokenizer, model, source_texts, device, max_new_tokens)
            for row, goal_tr, target_tr in zip(batch_rows, translated[0::2], translated[1::2]):
                writer.writerow(
                    {
                        "goal": goal_tr,
                        "target": target_tr,
                        "goal_en": row["goal"],
                        "target_en": row["target"],
                        "goal_tr": goal_tr,
                        "target_tr": target_tr,
                    }
                )
            f.flush()
            done = start_index + batch_start + len(batch_rows)
            print(f"[+] Wrote {done}/{len(rows)} rows to {output_path}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Translate AdvBench harmful prompts to Turkish.")
    parser.add_argument("--source", type=Path, default=config.ADVBENCH_ENGLISH_PATH)
    parser.add_argument("--output", type=Path, default=config.ADVBENCH_TURKISH_PATH)
    parser.add_argument("--preset", choices=sorted(TRANSLATION_PRESETS), default=DEFAULT_TRANSLATION_PRESET)
    parser.add_argument("--model", default=None, help="Override the translation model selected by --preset.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model or TRANSLATION_PRESETS[args.preset]
    batch_size = args.batch_size or PRESET_BATCH_SIZES[args.preset]
    output_path = build_turkish_advbench(
        source_path=args.source,
        output_path=args.output,
        model_name=model_name,
        batch_size=batch_size,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        resume=not args.no_resume,
    )
    print(f"[+] Turkish AdvBench dataset ready: {output_path}")


if __name__ == "__main__":
    main()
