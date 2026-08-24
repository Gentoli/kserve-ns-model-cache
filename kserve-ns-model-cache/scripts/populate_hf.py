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
        --extra-file <filename>      additional repo-relative file(s) to cache
                                     (repeatable; e.g. a custom mmproj)
        --tokenizer-repo <repo>      tokenizer/config source, defaults to <repo>
        --revision <rev>             HF revision for all downloads

    The multimodal projector (``*mmproj*.gguf``) is auto-detected in the GGUF
    repo and cached beside the backbone (best precision: BF16 > F16 > F32) so
    the vLLM GGUF plugin can find it at /mnt/models without extra config.
    The tokenizer allowlist also carries multimodal processor configs
    (preprocessor_config.json / processor_config.json /
    video_preprocessor_config.json) and the GPT-2-style BPE fallback
    (vocab.json + merges.txt), all best-effort.

It downloads the model into ``$CACHE_ROOT/<subPath>`` exactly once per namespace:
a single global lockfile serializes downloads of all models across all pods, with
a lock-free fast path for already-cached models (double-checked under the lock).
The Hugging Face library lays the model out inside the destination; no symlinks
and no per-pod copy -- the server reads the PVC subPath directly via pvc://.

For GGUF, only the selected quant (all shards if the file is split) plus the small
tokenizer/config files are downloaded -- never the whole multi-quant repo. Files
are flattened into the subPath root so the vLLM GGUF plugin's ``dir:quant``
resolution (``--model=/mnt/models:<quant>`` -> ``*-<quant>.gguf``) finds them.

The ready marker stores the cache-format version and a fingerprint of the source
(uri, or repo/quant/tokenizerRepo/revision/files). Changing the source config or
bumping CACHE_FORMAT_VERSION re-downloads that model on the next pod start.
"""
import fnmatch
import fcntl
import json
import os
import re
import shutil
import sys
from pathlib import Path

from kserve_storage import Storage
from kserve_storage.logging import configure_logging, logger

CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", "/cache"))
GLOBAL_LOCK_NAME = ".populate.lock"
# Bump when the cached layout changes (e.g. new required files like mmproj):
# ready markers written by older versions are then treated as stale and the
# model is re-downloaded on the next pod start.
CACHE_FORMAT_VERSION = "2"
GGUF_TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "merges.txt",
)
_GGUF_SHARD_RE = re.compile(r"-(\d+)-of-(\d+)\.gguf$")
_GGUF_FLAG_KEYS = {
    "--file": "file",
    "--tokenizer-repo": "tokenizer_repo",
    "--revision": "revision",
}
MMPROJ_PRECISION_ORDER = ("BF16", "F16", "F32")


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


def _remove_hf_cache_dirs(root: Path) -> None:
    """Recursively delete huggingface_hub's <local_dir>/.cache bookkeeping."""
    for dirpath in sorted(
        (p for p in root.rglob(".cache") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        shutil.rmtree(dirpath, ignore_errors=True)


def _ensure_compile_cache_dir() -> None:
    """Create the chart-requested vLLM compile cache dir (when configured).

    The kserve-container mounts <subPath>/<cacheDir>/<name> via subPath; creating
    it here (the init runs first, with the full PVC mounted rw) guarantees the
    server's mount always resolves, including on the first pod.
    """
    value = os.environ.get("COMPILE_CACHE_DIR", "")
    if value:
        Path(value).mkdir(parents=True, exist_ok=True)


def _ready_path(subpath: str) -> Path:
    return CACHE_ROOT / f".{subpath.replace('/', '_')}.ready"


def _expected_ready(fingerprint: str) -> str:
    return f"{CACHE_FORMAT_VERSION}|{fingerprint}"


def _ready_matches(ready: Path, fingerprint: str) -> bool:
    if not ready.exists():
        return False
    try:
        return ready.read_text() == _expected_ready(fingerprint)
    except OSError:
        return False


def _hf_fingerprint(uri: str) -> str:
    return uri


def _gguf_identity(
    repo: str,
    tokenizer_repo: str,
    revision: str | None,
) -> str:
    """Cache identity for a subPath: repo-level, deliberately excluding quant.

    Multiple quants of the same model may share one subPath; the per-file
    manifest keeps each quant's file and etag. A different repo (or tokenizer
    source / revision) is a real conflict and must use another subPath.
    """
    return "|".join(["gguf", repo, tokenizer_repo or "", revision or ""])


def _check_ready_identity(ready: Path, identity: str) -> None:
    """Fail when a same-version marker was written for a different source."""
    if not ready.exists():
        return
    try:
        content = ready.read_text()
    except OSError:
        return
    if content.startswith(CACHE_FORMAT_VERSION + "|") and content != _expected_ready(
        identity
    ):
        raise RuntimeError(
            "cache is already used by a different model source; "
            "pick a different subPath"
        )


def _is_tokenizer_name(name: str) -> bool:
    return name in GGUF_TOKENIZER_FILES or name == "tokenizer.model"


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


def _pick_mmproj(files: list[str]) -> str | None:
    """Pick the best-precision ``*mmproj*.gguf`` from a repo file listing."""
    candidates = [f for f in files if fnmatch.fnmatch(f, "*mmproj*.gguf")]
    if not candidates:
        return None

    def precision_key(filename: str):
        upper = filename.upper()
        for index, precision in enumerate(MMPROJ_PRECISION_ORDER):
            if precision in upper:
                return (index, filename)
        return (len(MMPROJ_PRECISION_ORDER), filename)

    return min(candidates, key=precision_key)


def _file_hash(info) -> str:
    """Content hash for a huggingface_hub RepoFile, across API versions.

    Newer hubs expose ``.etag``; the standard fields are the LFS sha256
    (``.sha``) and the git blob id (``.blob_id``). Any of them changes when the
    file content changes, which is all the cache freshness check needs.
    """
    for attr in ("etag", "sha", "blob_id"):
        value = getattr(info, attr, None)
        if value:
            return str(value)
    raise RuntimeError(
        f"cannot determine a content hash for {getattr(info, 'path', info)}"
    )


def _find_mmproj(repo: str, revision: str | None) -> str | None:
    """Return the best-precision ``*mmproj*.gguf`` in the repo, or None."""
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo, revision=revision)
    except Exception as e:
        raise RuntimeError(
            f"failed to list repo '{repo}' while auto-detecting mm_proj: {e}"
        ) from e
    return _pick_mmproj(files)


def _expected_gguf_files(
    repo: str,
    quant: str,
    file: str | None,
    tokenizer_repo: str,
    revision: str | None,
    extra_files: list[str] | None,
) -> list[dict]:
    """Resolve the exact cached file set and their current HF etags."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(repo, revision=revision)
    weight_patterns = _shard_filenames(file) if file else _gguf_weight_patterns(quant)
    weight_names = [
        name
        for name in files
        if any(fnmatch.fnmatch(name, pattern) for pattern in weight_patterns)
    ]
    if not weight_names:
        raise RuntimeError(_no_gguf_message(repo, quant))

    extra_files = list(extra_files or [])
    explicit_mmproj = any(
        fnmatch.fnmatch(f, "*mmproj*.gguf") for f in extra_files
    )
    mmproj = None if explicit_mmproj else _pick_mmproj(files)
    if mmproj:
        weight_names.append(mmproj)

    repo_infos = {
        info.path: info
        for info in api.get_paths_info(
            repo, weight_names + extra_files, revision=revision
        )
    }
    missing_extras = [name for name in extra_files if name not in repo_infos]
    if missing_extras:
        raise RuntimeError(
            f"requested GGUF extra file(s) not found in repo '{repo}': {missing_extras}"
        )

    tokenizer_names = list(GGUF_TOKENIZER_FILES) + ["tokenizer.model"]
    tokenizer_infos = {
        info.path: info
        for info in api.get_paths_info(
            tokenizer_repo, tokenizer_names, revision=revision
        )
    }

    expected: list[dict] = []
    for name in weight_names:
        if name in repo_infos:
            expected.append(
                {"name": name, "repo": repo, "etag": _file_hash(repo_infos[name])}
            )
    for name in extra_files:
        expected.append(
            {"name": name, "repo": repo, "etag": _file_hash(repo_infos[name])}
        )
    for name, info in tokenizer_infos.items():
        expected.append(
            {"name": name, "repo": tokenizer_repo, "etag": _file_hash(info)}
        )
    return expected


def _load_manifest(manifest_path: Path) -> dict | None:
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format") != CACHE_FORMAT_VERSION:
        return None
    return data


def _write_manifest(manifest_path: Path, expected: list[dict], dest: Path) -> None:
    files = [e for e in expected if (dest / Path(e["name"]).name).is_file()]
    manifest_path.write_text(
        json.dumps({"format": CACHE_FORMAT_VERSION, "files": files}, indent=1)
    )


def _cache_is_fresh(
    dest: Path,
    ready: Path,
    manifest_path: Path,
    identity: str,
    expected: list[dict],
) -> bool:
    if not _ready_matches(ready, identity):
        return False
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return False
    current = {(f["repo"], f["name"]): f["etag"] for f in manifest.get("files", [])}
    # Only this model's files must be fresh; the manifest may legitimately hold
    # other quants' entries (shared subPath).
    for entry in expected:
        if current.get((entry["repo"], entry["name"])) != entry["etag"]:
            return False
    return all((dest / Path(e["name"]).name).is_file() for e in expected)


def _update_changed_files(
    dest: Path,
    expected: list[dict],
    manifest_path: Path,
    safe: str,
    revision: str | None,
) -> list[dict]:
    """Re-download only changed/missing files; return what was updated.

    The manifest is a union across every model sharing the subPath: entries for
    files that still exist locally are kept (with this run's fresher etags where
    applicable), so one quant's refresh never drops another quant's file.
    """
    manifest = _load_manifest(manifest_path) or {"files": []}
    current = {(f["repo"], f["name"]): f["etag"] for f in manifest.get("files", [])}
    changed = [
        e
        for e in expected
        if current.get((e["repo"], e["name"])) != e["etag"]
        or not (dest / Path(e["name"]).name).is_file()
    ]
    if changed:
        logger.info("updating %d changed/missing file(s) for %s", len(changed), dest)
        tmp = CACHE_ROOT / f".{safe}.update.{os.getpid()}"
        for entry in changed:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            _snapshot(entry["repo"], revision, [entry["name"]], tmp)
            moved = _flatten(tmp, dest, lambda p: p.name == Path(entry["name"]).name)
            _remove_hf_cache_dirs(tmp)
            shutil.rmtree(tmp, ignore_errors=True)
            if not moved:
                raise RuntimeError(
                    f"updated file '{entry['name']}' was not found in "
                    f"repo '{entry['repo']}'"
                )

    # Union manifest: keep entries whose local files still exist, overlay this
    # run's fresh etags.
    union = {(f["repo"], f["name"]): f for f in manifest.get("files", [])}
    for entry in expected:
        union[(entry["repo"], entry["name"])] = entry
    _write_manifest(manifest_path, list(union.values()), dest)
    return changed


def _verify_tokenizer(dest: Path) -> None:
    if not (dest / "config.json").is_file():
        raise RuntimeError("missing required config.json in cached tokenizer files")
    has_bpe = (dest / "vocab.json").is_file() and (dest / "merges.txt").is_file()
    if (
        not (dest / "tokenizer.json").is_file()
        and not (dest / "tokenizer.model").is_file()
        and not has_bpe
    ):
        raise RuntimeError(
            "missing required tokenizer (tokenizer.json, tokenizer.model, "
            "or vocab.json + merges.txt) in cached tokenizer files"
        )


def populate_hf(uri: str, subpath: str) -> Path:
    """Download an HF repo snapshot into the cache subPath (original mode)."""
    dest = CACHE_ROOT / subpath                 # e.g. /cache/hf/llama-3-8b
    safe = subpath.replace("/", "_")
    ready = _ready_path(subpath)
    fingerprint = _hf_fingerprint(uri)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    _ensure_compile_cache_dir()

    # Fast path: already cached -> no lock, no contention.
    if _ready_matches(ready, fingerprint):
        logger.info("already cached: %s", dest)
        return dest

    # Global lock across ALL models/pods in the namespace; serialize downloads.
    with open(CACHE_ROOT / GLOBAL_LOCK_NAME, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if _ready_matches(ready, fingerprint):  # double-check
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
            _remove_hf_cache_dirs(tmp)
            shutil.rmtree(dest, ignore_errors=True)
            os.rename(tmp, dest)
            ready.write_text(_expected_ready(fingerprint))
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
    extra_files: list[str] | None = None,
) -> Path:
    """Cache one GGUF quant + supporting files, additively and etag-checked.

    The cache identity (repo + tokenizerRepo + revision) is stored in the ready
    marker, deliberately excluding the quant: several quants of the same model
    may share one subPath and are added/kept side by side. A sidecar manifest
    records every cached file's HF etag (union across sharing models). On each
    pod start the remote etags are compared (metadata-only, best-effort);
    changed or missing files are re-downloaded in place, unchanged files stay.
    """
    dest = CACHE_ROOT / subpath
    safe = subpath.replace("/", "_")
    ready = _ready_path(subpath)
    manifest_path = CACHE_ROOT / f".{safe}.manifest.json"
    tokenizer_repo = tokenizer_repo or repo
    identity = _gguf_identity(repo, tokenizer_repo, revision)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.mkdir(parents=True, exist_ok=True)
    _ensure_compile_cache_dir()
    _check_ready_identity(ready, identity)

    previously_cached = _ready_matches(ready, identity) and manifest_path.exists()
    if previously_cached:
        try:
            expected = _expected_gguf_files(
                repo, quant, file, tokenizer_repo, revision, extra_files
            )
        except Exception as e:
            # Metadata check failed (e.g. HF unreachable): keep serving the cache.
            logger.warning(
                "cache update check for %s failed (%s); keeping cached files", dest, e
            )
            return dest
        if _cache_is_fresh(dest, ready, manifest_path, identity, expected):
            logger.info("already cached: %s", dest)
            return dest
    else:
        expected = _expected_gguf_files(
            repo, quant, file, tokenizer_repo, revision, extra_files
        )

    # Global lock across ALL models/pods in the namespace; serialize downloads.
    with open(CACHE_ROOT / GLOBAL_LOCK_NAME, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _check_ready_identity(ready, identity)
        if _ready_matches(ready, identity) and manifest_path.exists():
            # Another pod populated/refreshed while we waited, or an in-place
            # update is needed for changed/missing files only.
            if _cache_is_fresh(dest, ready, manifest_path, identity, expected):
                logger.info("already cached: %s", dest)
                return dest
            changed = _update_changed_files(dest, expected, manifest_path, safe, revision)
            if changed and any(_is_tokenizer_name(e["name"]) for e in changed):
                _verify_tokenizer(dest)
            ready.write_text(_expected_ready(identity))
            logger.info("refreshed %s", dest)
            return dest

        # First population (or cache format changed): download every expected
        # file additively into the existing subPath (which may already hold
        # other quants of the same model).
        changed = _update_changed_files(dest, expected, manifest_path, safe, revision)
        if any(_is_tokenizer_name(e["name"]) for e in changed) or not manifest_path.exists():
            _verify_tokenizer(dest)
        ready.write_text(_expected_ready(identity))
        logger.info("cached gguf %s:%s -> %s", repo, quant, dest)
    return dest


def _parse_gguf_args(argv: list[str]) -> tuple[str, str, str, dict]:
    positional: list[str] = []
    options: dict = {
        "file": None,
        "tokenizer_repo": None,
        "revision": None,
        "extra_files": [],
    }
    i = 2  # skip program name and "gguf"
    while i < len(argv):
        arg = argv[i]
        if arg == "--extra-file":
            if i + 1 >= len(argv):
                raise ValueError(f"{arg} requires a value")
            options["extra_files"].append(argv[i + 1])
            i += 2
        elif arg in _GGUF_FLAG_KEYS:
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
