"""Package source and locked dependencies, never runs or virtual environments."""

import argparse
import hashlib
from pathlib import Path
import tarfile

from scripts.experiments.run_encoder_matrix import REPO, source_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/acd_colab_source.tar.gz"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Archive exists: {args.output}")
    paths = source_files() + [REPO / "README.md"]
    paths += list((REPO / "docs").glob("*.md"))
    paths += list((REPO / "notebooks").glob("*.ipynb"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz") as archive:
        for path in sorted(set(paths)):
            archive.add(path, arcname=Path("acd_benchmark") / path.relative_to(REPO))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"archive={args.output.resolve()}\nsha256={digest}")


if __name__ == "__main__":
    main()
