"""Download the processed files that are distributed as release assets.

Two files are too large for the GitHub web uploader. They are attached to the
latest release and placed in data/processed/ by this script. Existing files
with the expected checksum are left untouched.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "processed"
BASE = "https://github.com/TTFriday/arabidopsis-dormancy-kernels/releases/latest/download"

ASSETS = {
    "graph_embeddings_83.npy": "c2132c4803ac9f411f957ca6b4fe8565653d766aea8e23d0c0893c19cb245f56",
    "snp_annotation.csv": "09fa3404feaa640a1ee03724e0a233f5aa314f071ebe445832748606f076acb5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSETS.items():
        out = DEST / name
        if out.exists() and sha256(out) == expected:
            print(f"{name}: present")
            continue
        url = f"{BASE}/{name}"
        tmp = out.with_suffix(out.suffix + ".part")
        print(f"{name}: downloading")
        try:
            with urllib.request.urlopen(url, timeout=300) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"Could not download {url}\n"
                "Check the latest release, or download the asset manually and place it in "
                f"{DEST.relative_to(ROOT)}."
            ) from exc
        actual = sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise SystemExit(f"Checksum mismatch for {name}: {actual}")
        tmp.replace(out)
        print(f"{name}: written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
