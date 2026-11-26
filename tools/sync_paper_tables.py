"""Synchronize the two empirical manuscript tables, saving a content-addressed backup."""

import argparse
import hashlib
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tables", type=Path, default=ROOT / "results/paper_tables")
    p.add_argument("--manuscript", type=Path, default=ROOT / "paper/main.tex")
    args = p.parse_args()
    original = args.manuscript.read_text()
    text = original
    for label, filename in [
        ("tab:exp1-matched", "exp1_comparison_table.tex"),
        ("tab:trc-models", "exp2_comparison_table.tex"),
    ]:
        start = text.index(r"\label{" + label + "}")
        stop = text.index(r"\end{table}", start)
        table = (args.tables / filename).read_text().strip()
        section, count = re.subn(
            r"\\begin\{tabular\}.*?\\end\{tabular\}",
            lambda _: table,
            text[start:stop],
            count=1,
            flags=re.S,
        )
        if count != 1:
            raise ValueError(f"Expected one table for {label}")
        text = text[:start] + section + text[stop:]
    backup = (
        ROOT
        / "artifacts/manuscript_backups"
        / (hashlib.sha256(original.encode()).hexdigest()[:16] + ".tex")
    )
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(args.manuscript, backup)
    args.manuscript.write_text(text)
    print(f"Synchronized {args.manuscript}; previous source: {backup}")


if __name__ == "__main__":
    main()
