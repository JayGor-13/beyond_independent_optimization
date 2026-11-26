"""Check source syntax, frozen evidence, notebook hygiene and manuscript dependencies."""

import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    from build_release import release_files

    errors = []
    scripts = notebooks = frozen = 0
    for path in release_files():
        relative = path.relative_to(ROOT)
        if path.stat().st_size >= 95 * 1024**2:
            errors.append(f"File exceeds source-release size limit: {relative}")
        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(), filename=str(relative))
                scripts += 1
            except SyntaxError as exc:
                errors.append(f"{relative}: {exc}")
        if path.suffix == ".sh":
            result = subprocess.run(
                ["bash", "-n", str(path)], capture_output=True, text=True
            )
            if result.returncode:
                errors.append(f"Shell syntax: {relative}: {result.stderr}")
        if path.suffix == ".ipynb":
            data = json.loads(path.read_text())
            notebooks += 1
            if data.get("nbformat") != 4:
                errors.append(f"Invalid notebook: {relative}")
            if relative.parts[0] == "experiments" and any(
                c.get("outputs") or c.get("execution_count") is not None
                for c in data["cells"]
                if c["cell_type"] == "code"
            ):
                errors.append(
                    f"Runnable notebook contains execution outputs: {relative}"
                )
        if path.suffix in {".py", ".sh", ".ipynb", ".md", ".json", ".yml", ".txt"}:
            text = path.read_text()
            if re.search(
                r"(?:hf_[A-Za-z0-9]{25,}|ghp_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)",
                text,
            ):
                errors.append(f"Credential-like content: {relative}")
            if (
                path.suffix in {".py", ".sh"}
                and "/Users/" in text
                and path.name != "check_repository.py"
            ):
                errors.append(f"Machine-specific path: {relative}")
    for entry in json.loads((ROOT / "docs/source_manifest.json").read_text())["files"]:
        path = ROOT / entry["destination"]
        if not path.exists():
            errors.append(f"Missing relocated source: {entry['destination']}")
        elif entry["destination"].startswith(
            ("results/published/", "results/evidence/")
        ):
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["source_sha256"]:
                errors.append(f"Changed published evidence: {entry['destination']}")
            frozen += 1
    paper = ROOT / "paper"
    tex = (paper / "main.tex").read_text()
    for match in re.findall(
        r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}|\\input\{([^}]+)\}", tex
    ):
        name = next(v for v in match if v)
        if not (paper / name).exists():
            errors.append(f"Missing paper asset: {name}")
    for filename in ["references.bib", "tmlr.sty", "tmlr.bst"]:
        if not (paper / filename).exists():
            errors.append(f"Missing LaTeX dependency: {filename}")
    for path in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")]:
        for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", path.read_text()):
            target = target.strip("<>").split("#", 1)[0]
            if (
                target
                and not re.match(r"(?:https?://|mailto:)", target)
                and not (path.parent / target).exists()
            ):
                errors.append(f"Broken link: {path.relative_to(ROOT)} -> {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Repository checks passed: {scripts} Python files, {notebooks} notebooks, {frozen} unchanged evidence files; paper assets and documentation links resolve."
    )


if __name__ == "__main__":
    main()
