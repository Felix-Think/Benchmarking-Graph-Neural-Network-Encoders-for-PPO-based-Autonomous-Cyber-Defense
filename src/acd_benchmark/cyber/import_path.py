import sys
from pathlib import Path


def ensure_cyborg_on_path() -> None:
    """Prefer the vendored CybORG package path when running from the repo root."""
    repo_root = Path(__file__).resolve().parents[3]
    vendor_root = repo_root / "CybORG"
    vendor_root_str = str(vendor_root)
    if vendor_root_str not in sys.path:
        sys.path.insert(0, vendor_root_str)
