from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/log_update.py \"your update note\"")
        return 1

    update_file = Path(__file__).resolve().parents[1] / "PROJECT_UPDATES.md"
    message = " ".join(sys.argv[1:]).strip()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with update_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\n- {timestamp}: {message}\n")
    print(f"Appended update to {update_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
