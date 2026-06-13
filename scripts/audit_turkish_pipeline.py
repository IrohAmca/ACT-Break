import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("ACT_BREAK_MODEL_PROFILE", "kara-kumru")
os.environ["ACT_BREAK_ADVBENCH_LANGUAGE"] = "tr"

import config
from src.activation_collector import load_prompts


MODEL_NAME = config.MODEL_NAME
REQUIRED_COLUMNS = {"goal", "target", "goal_en", "target_en", "goal_tr", "target_tr"}
COMPLIANCE_OPENERS = ("elbette", "tabii", "kesinlikle", "memnuniyetle", "işte")
ENGLISH_MARKERS = ("write ", "create ", "provide ", "give ", "develop ", "explain ", "hack into ")


def fail(message: str):
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def warn(message: str):
    print(f"[WARN] {message}")


def ok(message: str):
    print(f"[OK] {message}")


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        fail(f"Turkish dataset is missing: {path}")

    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError as exc:
        fail(f"{path} is not valid UTF-8: {exc}")

    if not rows:
        fail(f"{path} has no rows")

    columns = set(rows[0].keys())
    missing = REQUIRED_COLUMNS - columns
    if missing:
        fail(f"{path} is missing columns: {sorted(missing)}")

    return rows


def audit_dataset(rows: list[dict]):
    ok(f"Dataset rows: {len(rows)}")
    if len(rows) != 520:
        warn("Expected 520 AdvBench rows; continue only if this is an intentional subset.")

    for column in REQUIRED_COLUMNS:
        empty = sum(1 for row in rows if not row.get(column, "").strip())
        if empty:
            fail(f"Column {column!r} has {empty} empty rows")
    ok("Required CSV columns are present and non-empty.")

    turkish_chars = sum(any(ch in row["goal_tr"] + row["target_tr"] for ch in "çğıöşüÇĞİÖŞÜ") for row in rows)
    if turkish_chars < len(rows) * 0.5:
        warn(f"Only {turkish_chars}/{len(rows)} rows contain Turkish-specific characters.")
    else:
        ok(f"Turkish character coverage: {turkish_chars}/{len(rows)} rows.")

    englishish = [
        idx + 1
        for idx, row in enumerate(rows)
        if any(marker in row["goal_tr"].lower() for marker in ENGLISH_MARKERS)
    ]
    if englishish:
        warn(f"{len(englishish)} goal_tr rows look English-like. First rows: {englishish[:10]}")
    else:
        ok("No obvious English prompt leakage detected in goal_tr.")

    weak_target_openers = [
        idx + 1
        for idx, row in enumerate(rows)
        if not row["target_tr"].strip().lower().startswith(COMPLIANCE_OPENERS)
    ]
    if weak_target_openers:
        warn(f"{len(weak_target_openers)} target_tr rows do not start with a common Turkish compliance opener.")
    else:
        ok("target_tr compliance openers look consistent.")


def audit_config():
    if config.ADVBENCH_LANGUAGE != "tr":
        fail(f"config.ADVBENCH_LANGUAGE is {config.ADVBENCH_LANGUAGE!r}, expected 'tr'")
    if config.ADVBENCH_PATH != config.ADVBENCH_TURKISH_PATH:
        fail(f"config.ADVBENCH_PATH is {config.ADVBENCH_PATH}, expected {config.ADVBENCH_TURKISH_PATH}")
    if "işte" not in config.DEFAULT_COMPLIANCE_PREFIX or "işte" not in config.TARGET_STRING:
        fail(
            "Turkish compliance prefix/target are not active. "
            f"prefix={config.DEFAULT_COMPLIANCE_PREFIX!r}, target={config.TARGET_STRING!r}"
        )
    if config.NEGATIVE_ACTIVATION_MODE != "forced_refusal":
        warn(
            "NEGATIVE_ACTIVATION_MODE is not 'forced_refusal'. "
            "Free-generation negatives are valid for compatibility, but Kara-Kumru may produce off-topic "
            "or English baseline responses."
        )
    ok(f"Config selects Turkish dataset and target: {config.DEFAULT_COMPLIANCE_PREFIX!r}")


def audit_loader(path: Path):
    prompts = load_prompts(str(path), max_prompts=3, language="tr")
    if len(prompts) != 3:
        fail("load_prompts did not return the expected sample size.")
    if not all(item["goal"].strip() and item["target"].strip() for item in prompts):
        fail("load_prompts returned an empty goal or target.")
    ok("load_prompts(language='tr') reads goal_tr/target_tr.")


def audit_tokenizer(rows: list[dict]):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME, padding_side="left")
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": rows[0]["goal_tr"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    for token in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            fail(f"Chat template token {token!r} maps to unk_token_id")
    if rows[0]["goal_tr"] not in formatted:
        fail("Chat template did not preserve the Turkish prompt text.")
    ok("Kara-Kumru tokenizer fallback preserves Turkish prompt text and chat-template tokens.")


def main():
    rows = load_rows(config.ADVBENCH_TURKISH_PATH)
    audit_dataset(rows)
    audit_config()
    audit_loader(config.ADVBENCH_TURKISH_PATH)
    audit_tokenizer(rows)
    ok("Turkish pipeline audit passed.")


if __name__ == "__main__":
    main()
