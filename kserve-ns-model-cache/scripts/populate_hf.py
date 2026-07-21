#!/usr/bin/env python3
"""Populate a namespace-shared RWX model cache from Hugging Face, once per model.

Runs on the stock kserve/storage-initializer image (python3 + kserve_storage),
delivered via a ConfigMap -- no custom image build required.

Used as a pod init container. The InferenceService's storageUri is
``pvc://<claim>/<subPath>``: the KServe webhook mounts that PVC subPath read-only
at ``/mnt/models`` on kserve-container. This script's only job is to make sure the
model exists at that subPath on the PVC before the server starts:

    sys.argv[1] = source URI   (e.g. hf://owner/model[:revision])
    sys.argv[2] = subPath      (path inside the PVC; == the pvc:// subPath)

It downloads the model into ``$CACHE_ROOT/<subPath>`` exactly once per namespace:
a single global lockfile serializes downloads of all models across all pods, with
a lock-free fast path for already-cached models (double-checked under the lock).
The Hugging Face library lays the model out inside the destination; no symlinks
and no per-pod copy -- the server reads the PVC subPath directly via pvc://.
"""
import fcntl
import os
import shutil
import sys
from pathlib import Path

from kserve_storage import Storage
from kserve_storage.logging import configure_logging, logger

CACHE_ROOT = Path(os.environ.get("CACHE_ROOT", "/cache"))
GLOBAL_LOCK_NAME = ".populate.lock"


def main() -> int:
    # Same logging setup as the stock storage-initializer entrypoint: a
    # timestamped "storage.initializer" logger on stderr. The download progress
    # itself is Hugging Face's tqdm/hf_transfer bars (also on stderr); the chart
    # runs this with `python3 -u` so that stream is unbuffered and the progress
    # shows in the pod logs live instead of being flushed only at exit.
    configure_logging()
    if len(sys.argv) != 3:
        logger.error(__doc__)
        return 2
    uri, subpath = sys.argv[1], sys.argv[2]
    logger.info("populating %s -> %s", uri, subpath)

    dest = CACHE_ROOT / subpath                 # e.g. /cache/hf/llama-3-8b
    safe = subpath.replace("/", "_")
    ready = CACHE_ROOT / f".{safe}.ready"

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: already cached -> no lock, no contention.
    if not ready.exists():
        # Global lock across ALL models/pods in the namespace; serialize downloads.
        with open(CACHE_ROOT / GLOBAL_LOCK_NAME, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            if not ready.exists():              # double-check
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
    else:
        logger.info("already cached: %s", dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
