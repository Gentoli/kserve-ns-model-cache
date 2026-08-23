#!/usr/bin/env python3
"""Populate a namespace-shared RWX model cache from Hugging Face, once per model.

Runs on the stock kserve/storage-initializer image (python3 + kserve_storage),
delivered via a ConfigMap -- no custom image build required.

Used as a pod init container. The InferenceService's storageUri is
``pvc://<claim>/<subPath>``: the KServe webhook mounts that PVC subPath read-only
at ``/mnt/models`` on kserve-container. This script's only job is to make sure the
model exists at that subPath on the PVC before the server starts:

    populate_hf.py <hfUri> <subPath>
        hfUri     source URI (e.g. hf://owner/model[:revision])
        subPath   path inside the PVC; == the pvc:// subPath

    populate_hf.py gguf <repo> <quant> <subPath> [options]
        repo      HF repo containing .gguf weights (e.g. unsloth/Qwen3-8B-GGUF)
        quant     ONE quantized file type to cache/serve (e.g. Q8_0, Q4_K_M)
        subPath   path inside the PVC; == the pvc:// subPath
        --file <filename>            exact repo-relative filename override
        --tokenizer-repo <repo>      tokenizer/config source, defaults to <repo>
        --revision <rev>             HF revision for all downloads

It downloads the model into ``$CACHE_ROOT/<subPath>`` exactly once per namespace:
a single global lockfile serializes downloads of all models across all pods, with
a lock-free fast path for already-cached models (double-checked under the lock).
The Hugging Face library lays the model out inside the destination; no symlinks
and no per-pod copy -- the server reads the PVC subPath directly via pvc://.

For GGUF, only the selected quant (all shards if the file is split) plus the small
tokenizer/config files are downloaded -- never the whole multi-quant repo. Files
are flattened into the subPath root so the vLLM GGUF plugin's ``dir:quant``
resolution (``--model=/mnt/models:<quant>`` -> ``*-<quant>.gguf``) finds them.
"""
import fcntl
import os
import re
import shutil
import sys
from pathlib import Path

from kserve_storage import Storage
from kserve_storage.logging import configure_logging, logger

CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", "/cache"))
GLOBAL_LOCK_NAME = ".populate.lock"
GGUF_TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "added_tokens.json",
)
_GGUF_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$")
_GGUF_FLAG_KEYS = {
    "--file": "file",
    "--tokenizer-repo": "tokenizer_repo",
    "--revision": "revision",
}


def _gguf_weight_patterns(quant: str) -> list[str]:
    """Mirror vllm-gguf-plugin's quant -> file match patterns."""
    patterns: list[str] = []
    for quant_case in (quant.upper(), quant.lower()):
        patterns.extend(
            [
                f"*-{quant_case}.gguf",
                f"*-{quant_case}-*.gguf",
                f"*/*-{quant_case}.gguf",
                f"*/*-{quant_case}-*.gguf",
            ]
        )
    return patterns


def _shard_filenames(filename: str) -> list[str]:
    """Expand ``*-00001-of-00002.gguf`` into every shard filename."""
    match = _GGUF_SHARD_RE.search(filename)
    if not match:
        return [filename]
    total = int(match.group(2))
    digits = len(match.group(1))
    prefix = filename[: match.start(1)]
    suffix = filename[match.end(2) :]
    return [
        f"{prefix}{index:0{digits}d}-of-{total:0{digits}d}{suffix}"
        for index in range(1, total + 1)
    ]


def _snapshot(repo_id: str, revision: str | None, allow_patterns: list[str], local_dir: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo_id,
        revision=revision or None,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir),
    )


def _flatten(src: Path, dest: Path, wanted) -> list[Path]:
    """Move wanted files from src (recursively) to dest root, skipping HF bookkeeping."""
    moved: list[Path] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        if not wanted(path):
            continue
        target = dest / path.name
        os.replace(str(path), str(target))
        moved.append(target)
    return moved


def _prune_empty_dirs(root: Path) -> None:
    """Remove directories left empty after flattening (deepest first)."""
    dirs = sorted(
        (p for p in root.rglob("*") if p.is_dir() and ".cache" not in p.parts),
        key=lambda p: len(p.parts),
        reverse=True,
    )
    for dirpath in dirs:
        try:
            dirpath.rmdir()
        except OSError:
            pass


def _no_gguf_message(repo: str, quant: str) -> str:
    hint = ""
    try:
        from huggingface_hub import HfApi

        candidates = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf")]
        near = [f for f in candidates if quant.upper() in f.upper()]
        hint = f" (files containing '{quant}': {near})" if near else f" (.gguf files in repo: {candidates[:20]})"
    except Exception as e:  # listing is best-effort; the download error is what matters
        hint = f" (listing failed: {e})"
    return f"No GGUF file matched quant '{quant}' in repo '{repo}'{hint}"


def _verify_tokenizer(dest: Path) -> None:
    if not (dest / "config.json").is_file():
        raise RuntimeError("missing required config.json in cached tokenizer files")
    if not (dest / "tokenizer.json").is_file() and not (dest / "tokenizer.model").is_file():
        raise RuntimeError(
            "missing required tokenizer.json (or tokenizer.model) in cached tokenizer files"
        )


def populate_hf(uri: str, subpath: str) -> Path:
    """Download an HF repo snapshot into the cache subPath (original mode)."""
    dest = CACHE_ROOT / subpath                 # e.g. /cache/hf/llama-3-8b
    safe = subpath.replace("/", "_")
    ready = CACHE_ROOT / f".{safe}.ready"

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: already cached -> no lock, no contention.
    if ready.exists():
        logger.info("already cached: %s", dest)
        return dest

    # Global lock across ALL models/pods in the namespace; serialize downloads.
    with open(CACHE_ROOT / GLOBAL_LOCK_NAME, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if ready.exists():                      # double-check
            return dest
        # Download into staging, then atomically rename into the subPath so
        # no consumer can observe a partial model. rename() is atomic within
        # a filesystem; staging and final both live on the RWX volume.
        tmp = CACHE_ROOT / f".{safe}.incomplete.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        try:
            Storage.download(uri, str(tmp))
            # Hugging Face writes per-file download bookkeeping into
            # <local_dir>/.cache/huggingface (its own, separate from
            # HF_HOME). Drop it so it doesn't leak into the served model
            # dir (/mnt/models/.cache/...). Model files are untouched.
            shutil.rmtree(tmp / ".cache", ignore_errors=True)
            shutil.rmtree(dest, ignore_errors=True)
            os.rename(tmp, dest)
            ready.touch()
            logger.info("cached %s -> %s", uri, dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return dest


def populate_gguf(
    repo: str,
    quant: str,
    subpath: str,
    file: str | None = None,
    tokenizer_repo: str | None = None,
    revision: str | None = None,
) -> Path:
    """Cache one GGUF quant + its tokenizer/config files into the cache subPath."""
    dest = CACHE_ROOT / subpath
    safe = subpath.replace("/", "_")
    ready = CACHE_ROOT / f".{safe}.ready"
    tokenizer_repo = tokenizer_repo or repo

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: already cached -> no lock, no contention.
    if ready.exists():
        logger.info("already cached: %s", dest)
        return dest

    with open(CACHE_ROOT / GLOBAL_LOCK_NAME, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if ready.exists():
            return dest

        tmp = CACHE_ROOT / f".{safe}.incomplete.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        weights_dir = tmp / "weights"
        tokenizer_dir = tmp / "tokenizer"
        try:
            weights_dir.mkdir(parents=True, exist_ok=True)
            tokenizer_dir.mkdir(parents=True, exist_ok=True)

            # Weights: only the selected quant. snapshot_download with
            # allow_patterns downloads just the matching file(s), never the
            # whole multi-quant repo; sharded files match via the "-*" pattern
            # (or exact shard names when --file is given).
            if file:
                weight_patterns = _shard_filenames(file)
            else:
                weight_patterns = _gguf_weight_patterns(quant)
            _snapshot(repo, revision, weight_patterns, weights_dir)
            moved_weights = _flatten(weights_dir, tmp, lambda p: p.suffix == ".gguf")
            if not moved_weights:
                raise RuntimeError(_no_gguf_message(repo, quant))

            # Tokenizer/config files (small JSONs) from the GGUF repo by default.
            tokenizer_patterns = list(GGUF_TOKENIZER_FILES) + ["tokenizer.model"]
            _snapshot(tokenizer_repo, revision, tokenizer_patterns, tokenizer_dir)
            _flatten(
                tokenizer_dir,
                tmp,
                lambda p: p.name in GGUF_TOKENIZER_FILES or p.name == "tokenizer.model",
            )
            _verify_tokenizer(tmp)

            shutil.rmtree(tmp / ".cache", ignore_errors=True)
            _prune_empty_dirs(tmp)
            shutil.rmtree(dest, ignore_errors=True)
            os.rename(tmp, dest)
            ready.touch()
            logger.info("cached gguf %s:%s -> %s", repo, quant, dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
    return dest


def _parse_gguf_args(argv: list[str]) -> tuple[str, str, str, dict]:
    positional: list[str] = []
    options: dict = {"file": None, "tokenizer_repo": None, "revision": None}
    i = 2  # skip program name and "gguf"
    while i < len(argv):
        arg = argv[i]
        if arg in _GGUF_FLAG_KEYS:
            if i + 1 >= len(argv):
                raise ValueError(f"{arg} requires a value")
            options[_GGUF_FLAG_KEYS[arg]] = argv[i + 1]
            i += 2
        else:
            positional.append(arg)
            i += 1
    if len(positional) != 3:
        raise ValueError("gguf mode requires: <repo> <quant> <subPath>")
    return positional[0], positional[1], positional[2], options


def main() -> int:
    # Same logging setup as the stock storage-initializer entrypoint: a
    # timestamped "storage.initializer" logger on stderr. The download progress
    # itself is Hugging Face's tqdm/hf_transfer bars (also on stderr); the chart
    # runs this with `python3 -u` so that stream is unbuffered and the progress
    # shows in the pod logs live instead of being flushed only at exit.
    configure_logging()
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "gguf":
            repo, quant, subpath, options = _parse_gguf_args(sys.argv)
            logger.info("populating gguf %s:%s -> %s", repo, quant, subpath)
            populate_gguf(repo, quant, subpath, **options)
        else:
            if len(sys.argv) != 3:
                raise ValueError("usage: populate_hf.py <hfUri> <subPath>")
            uri, subpath = sys.argv[1], sys.argv[2]
            logger.info("populating %s -> %s", uri, subpath)
            populate_hf(uri, subpath)
    except BaseException as e:
        logger.error("%s", e)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
