#!/usr/bin/env python3
"""Read-only runtime identity check for GELLO / Diffusion Policy terminals."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional


DEFAULT_MODULES = (
    "gello",
    "diffusion_policy",
    "zmq",
    "yaml",
    "torch",
    "pyrealsense2",
    "rtde_control",
    "onRobot",
    "onRobot.gripper",
)


def _module_origin(name: str) -> Optional[str]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, AttributeError, ValueError):
        return None
    if spec is None:
        return None
    if spec.origin is not None:
        return str(spec.origin)
    locations = spec.submodule_search_locations
    return None if not locations else ":".join(str(item) for item in locations)


def _git_identity(path: Path) -> dict:
    def run(*args: str) -> Optional[str]:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "path": str(path),
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "porcelain": run("status", "--short"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", action="append", default=[])
    parser.add_argument(
        "--python-path",
        action="append",
        default=[],
        help="directory to prepend to this interpreter's module search path",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--require-module", action="append", default=[])
    args = parser.parse_args()

    for item in reversed(args.python_path):
        resolved = str(Path(item).expanduser().resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)

    module_names = tuple(dict.fromkeys((*DEFAULT_MODULES, *args.require_module)))
    modules = {name: _module_origin(name) for name in module_names}
    checkpoint = None
    if args.checkpoint:
        path = Path(args.checkpoint).expanduser().resolve()
        checkpoint = {
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
        }

    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "working_directory": str(Path.cwd()),
        "modules": modules,
        "repositories": [
            _git_identity(Path(item).expanduser().resolve()) for item in args.repo_root
        ],
        "checkpoint": checkpoint,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    missing = [name for name in args.require_module if modules.get(name) is None]
    if missing:
        print(f"ERROR: required modules missing: {missing}", file=sys.stderr)
        return 2
    if checkpoint is not None and not checkpoint["exists"]:
        print("ERROR: checkpoint does not exist", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
