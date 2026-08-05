import sys
from pathlib import Path


def repository_root() -> Path:
    """Return the code repository root for every scripts subpackage."""
    return Path(__file__).resolve().parents[1]


def ensure_src_on_path() -> None:
    """Make the local ``src`` package importable without installation."""
    src_dir = repository_root() / "src"
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
