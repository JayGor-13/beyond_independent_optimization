"""Build a portable source release without runtime environments, datasets or weights."""

import hashlib
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {
    ".github",
    "docs",
    "environments",
    "experiments",
    "experimental",
    "paper",
    "results",
    "tools",
}
ROOT_FILES = {
    "README.md",
    "Makefile",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    "requirements-analysis.txt",
}


def release_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink():
            continue
        if (
            relative.parts[0] not in DIRECTORIES
            and relative.as_posix() not in ROOT_FILES
        ):
            continue
        if any(
            v in {"__pycache__", ".pytest_cache", ".git", "model_shards", "outputs"}
            or v.startswith((".venv", "outputs_"))
            for v in relative.parts
        ):
            continue
        if (
            path.suffix
            in {".pyc", ".safetensors", ".pt", ".pth", ".bin", ".mp4", ".zip"}
            or path.name == ".DS_Store"
        ):
            continue
        yield path


def main():
    subprocess.run(
        [sys.executable, str(ROOT / "tools/check_repository.py")], check=True
    )
    target = ROOT / "dist/tmlr-survey-code.zip"
    target.parent.mkdir(exist_ok=True)
    checksums = []
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in release_files():
            relative = path.relative_to(ROOT).as_posix()
            archive.write(path, f"tmlr-survey-code/{relative}")
            checksums.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}"
            )
        archive.writestr("tmlr-survey-code/SHA256SUMS", "\n".join(checksums) + "\n")
    print(
        f"Release: {target} ({target.stat().st_size / 1024**2:.2f} MiB, {len(checksums)} files)"
    )


if __name__ == "__main__":
    main()
