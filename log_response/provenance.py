"""Provenance capture: everything needed to trust and reproduce a saved run.

A stored `D(freq, contrast)` surface is only worth keeping if you can tell what
produced it. In particular a run whose pretrained weights failed to download is
numerically indistinguishable from a real one (see ``weights_ok`` on the model
back-ends), so the weight state is recorded explicitly rather than assumed.

Every helper here reports *why* it could not collect something instead of
silently omitting it -- a missing field must never be mistakable for a clean one.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys


def _git(*args: str, repo: str) -> str | None:
    """Run a git command, or return None if git/the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # Trailing newline only: `git status --porcelain` encodes the index/worktree
    # state in two leading columns, so an unstaged-only change begins with a
    # space. Stripping leading whitespace would eat that column on the first
    # line alone and silently truncate the first character of its path.
    return out.stdout.rstrip("\n")


def git_provenance(repo: str | None = None) -> dict:
    """The commit a run was produced from, plus whether the tree was dirty.

    ``dirty`` matters as much as the commit: a run made from uncommitted edits
    is not reproducible from the commit alone, and is flagged so.
    """
    repo = repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    commit = _git("rev-parse", "HEAD", repo=repo)
    if commit is None:
        return {"available": False, "reason": "git not available or not a repository"}
    status = _git("status", "--porcelain", repo=repo)
    return {
        "available": True,
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
        "dirty": bool(status),
        "dirty_files": sorted(
            line[3:] for line in (status or "").splitlines() if line[3:]
        )
        or None,
    }


def package_versions() -> dict:
    """Versions of the packages that can change the numbers."""
    versions = {"python": platform.python_version()}
    for name in (
        "numpy",
        "torch",
        "torchvision",
        "timm",
        "huggingface_hub",
        "transformers",
        "open_clip",
    ):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def environment() -> dict:
    """Where the run happened. Wall time is comparable only within a machine."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "executable": sys.executable,
    }


def state_dict_digest(state) -> str:
    """A digest of *weights*, stable across machines and re-saves.

    ``torch.save`` is not byte-reproducible -- saving the same tensors twice in
    one process yields different files -- so a file sha256 cannot answer "did
    this run use the same checkpoint?". Hashing the tensors themselves can:
    name, dtype, shape and bytes, in sorted key order.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def file_fingerprint(path: str) -> dict:
    """Identify a weights file by content, not by a path that will not survive.

    A local ``--weights`` path is meaningless to anyone else and disappears with
    the machine. Two digests are recorded because they answer different
    questions: ``sha256`` identifies the *file* (and is not reproducible across
    re-saves), while ``weights_sha256`` identifies the *tensors* and is what
    pins a checkpoint -- a regenerated conversion matches on the second and not
    the first.
    """
    info: dict = {"path": os.path.abspath(path)}
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        info["sha256"] = digest.hexdigest()
        info["bytes"] = os.path.getsize(path)
    except OSError as exc:
        info["sha256"] = None
        info["error"] = f"could not hash: {exc}"
        return info

    # The weights digest needs the file to actually be a torch state_dict; CLIP
    # checkpoints and hf: directories are not, so say why rather than omitting.
    try:
        import torch

        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and state and all(
            hasattr(v, "detach") for v in state.values()
        ):
            info["weights_sha256"] = state_dict_digest(state)
        else:
            info["weights_sha256"] = None
            info["weights_note"] = "not a flat tensor state_dict; file hash only"
    except Exception as exc:  # not loadable as a state_dict
        info["weights_sha256"] = None
        info["weights_note"] = f"could not read as a state_dict: {exc}"
    return info
