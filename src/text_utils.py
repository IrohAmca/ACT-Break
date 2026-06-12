import sys


def safe_console_text(value, max_chars: int | None = None) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if max_chars is not None:
        text = text[:max_chars]

    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding, errors="replace")


def safe_print(value) -> None:
    print(safe_console_text(value))
