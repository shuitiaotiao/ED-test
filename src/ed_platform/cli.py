from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import (
    compare_cases,
    discover_case_dirs,
    summarize_for_terminal,
    write_validation_report,
    write_validation_table_csv,
    write_validation_table_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ED/QMC validation from the command line.")
    parser.add_argument("root", type=Path, help="Case directory, parent directory, or extracted upload bundle.")
    parser.add_argument("--bc-y", choices=("PBC", "APBC"), default="PBC")
    parser.add_argument("--max-basis-states", type=int, default=250_000)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated JSON/CSV/Markdown reports.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = args.root.resolve()
    case_dirs = discover_case_dirs(root)
    if not case_dirs:
        parser.error(f"No ED case directories were found under {root}")

    results = compare_cases(case_dirs, bc_y=args.bc_y, max_basis_states=args.max_basis_states)
    for result in results:
        print(summarize_for_terminal(result))
        print("-" * 80)

    output_dir = (args.output_dir or root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = output_dir / "ed_validation_report.json"
    report_csv = output_dir / "ed_validation_comparison.csv"
    report_md = output_dir / "ed_validation_comparison.md"
    write_validation_report(results, report_json)
    write_validation_table_csv(results, report_csv)
    write_validation_table_markdown(results, report_md)
    print(f"Wrote reports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
