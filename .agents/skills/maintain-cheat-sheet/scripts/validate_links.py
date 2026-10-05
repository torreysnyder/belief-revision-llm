"""Validate repository-relative Markdown links in cheat_sheet.md."""

import re
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SKILL_DIR.parents[2]
CHEAT_SHEET = REPOSITORY_ROOT / "cheat_sheet.md"
LINK_PATTERN = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def main() -> None:
    """Exit with an error when a local cheat-sheet link is missing."""

    missing: list[str] = []
    for target in LINK_PATTERN.findall(CHEAT_SHEET.read_text()):
        if "://" in target or target.startswith("#"):
            continue

        path_text = target.split("#", 1)[0]
        if path_text and not (CHEAT_SHEET.parent / path_text).exists():
            missing.append(target)

    if missing:
        raise SystemExit("Missing cheat-sheet link targets: " + ", ".join(missing))

    print("All local cheat-sheet links resolve.")


if __name__ == "__main__":
    main()
