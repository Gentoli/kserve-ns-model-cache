# kserve-ns-model-cache

Namespace-shared **ReadWriteMany** model cache for KServe. Each Hugging Face
model is fetched **once per namespace** (flock-guarded) into a shared PVC, and
every `kserve-container` then serves it **directly off the PVC** — no per-pod
download, **no symlinks, no per-pod copy**, and no cross-namespace hostPath
sharing. Works with **vLLM** and the **HuggingFace** runtime (both read the
standard `/mnt/models`), including **GGUF** weights served by a plugin-enabled
vLLM runtime.

The fetch logic is a **Python script delivered via ConfigMap**, run on the
**stock `kserve/storage-initializer` image** — **no custom image build** (no
BuildConfig/ImageStream). It runs as a **pod init container**; serving uses a
`pvc://` `storageUri` so the webhook mounts the populated PVC subPath read-only
at `/mnt/models`.

## How it works

```
ConfigMap (populate_hf.py) ── mounted into the init container

InferenceService (one per values.models[])
  predictor.initContainers:
    populate-hf  (stock image, command=python3 …/populate_hf.py)
       mounts PVC "model-cache" rw at /cache
       │ flock(global) → Storage.download(hf://…) → /cache/<subPath>   (atomic rename, once)
       ▼ exits 0  ── init containers complete before the server starts
  model.storageUri: pvc://model-cache/<subPath>
       │ webhook mounts PVC "model-cache" subPath=<subPath> read-only at /mnt/models
       ▼
  kserve-container reads /mnt/models = PVC:/<subPath>   (zero-copy, no symlink/copy)
```

- **Fetch once:** a single **global** lockfile (`$CACHE_ROOT/.populate.lock`,
  `flock(2)`) serializes downloads of *all* models across *all* pods; already-
  cached models take a lock-free fast path (double-checked under the lock). One
  HF pull per namespace per model.
- **No partial reads:** download lands in `*.incomplete.<pid>` then `rename(2)`
  (atomic, same filesystem) into the subPath; the server starts only after the
  init container completes.
- **Zero-copy serve, no symlink/copy:** the model bytes stay on the PVC; the
  server reads them through the `pvc://` subPath mount at `/mnt/models`. The HF
  library lays the files out — the script does no file-structure handling.
- **`subPath` is the shared key:** it is both the init container's download
  destination and the `pvc://` subPath. The chart renders both from the one
  `models[].subPath` value so they cannot drift.

## Install

```sh
# RWX-capable storage class + (optional) HF token for gated repos
kubectl create secret generic hf-secret --from-literal=HF_TOKEN=hf_xxx

helm install model-cache ./kserve-ns-model-cache -n team-a \
  --set cache.pvc.storageClassName=nfs-csi \
  --set populate.image.tag=<your storage-initializer tag> \
  --set hf.tokenSecret.name=hf-secret \
  -f my-models.yaml
```

Each `models[]` entry becomes one `InferenceService`, all sharing the cache PVC
and populate script.

## Flexible model/predictor pass-through

Beyond the required `name` / `hfUri` / `subPath`, each model accepts free-form
`predictor:` and `model:` blocks that are merged into the rendered
InferenceService. The chart only forces `predictor.volumes` /
`predictor.initContainers` and `model.storageUri` (derived from `hfUri`+`subPath`),
and defaults `model.modelFormat` to `huggingface` (`vLLM`/`1` + `runtime:
kserve-vllmserver` for `gguf:` entries). Everything else is yours:

```yaml
models:
  - name: mellum
    hfUri: hf://JetBrains/Mellum2-12B-A2.5B-Thinking
    subPath: hf/mellum
    predictor:                 # -> spec.predictor
      minReplicas: 0
    model:                     # -> spec.predictor.model
      modelFormat: { name: huggingface }
      args:
        - --max-model-len=131072
        - --reasoning-parser=qwen3
        - --enable-auto-tool-choice
        - --tool-call-parser=hermes
      resources:
        requests: { cpu: "4", memory: 4Gi }
        limits:   { cpu: "4", memory: 20Gi, nvidia.com/gpu.shared: "1" }
    initResources:             # -> resources for the populate init container
      requests: { cpu: "1", memory: 4Gi }
```

For vLLM, point `model.runtime` at your vLLM ServingRuntime/ClusterServingRuntime.

## GGUF models (vLLM + vllm-gguf-plugin)

GGUF repos usually contain many quantized weights; this chart caches **exactly
one** — the quant you serve — plus the small tokenizer/config files, never the
whole repo. Use a `gguf:` block instead of `hfUri`:

```yaml
models:
  - name: qwen3-8b-gguf
    gguf:
      repo: unsloth/Qwen3-8B-GGUF   # HF repo containing .gguf weights
      quant: Q8_0                    # ONE quant to cache/serve
      # file: Qwen3-8B-Q8_0.gguf     # optional exact filename override
      # tokenizerRepo: ""            # optional tokenizer source, defaults to repo
      # revision: ""                 # optional HF revision
    subPath: hf/qwen3-8b-gguf        # still the shared key (init dest + pvc:// subPath)
    predictor:
      minReplicas: 0
    model:
      args:
        - --max-model-len=32768
      resources:
        limits:
          nvidia.com/gpu: "1"
```

For a `gguf:` entry the chart:

- downloads only `*-<quant>.gguf` (all shards if split) and the tokenizer
  allowlist (`config.json`, `tokenizer.json`/`tokenizer.model`, …) into the
  subPath, flattening files to its root;
- auto-detects a multimodal projector (`*mmproj*.gguf`, best precision
  BF16 > F16 > F32) and caches it beside the backbone, so vision-capable GGUF
  models load without extra vLLM config; add `gguf.files: [<name>]` to pin any
  additional repo files instead;
- defaults `model.runtime` to `kserve-vllmserver` (a prebuilt image containing
  `vllm-gguf-plugin`; override with `models[].model.runtime`) and
  `model.modelFormat` to `vLLM`/`1`;
- appends `--model=/mnt/models:<quant> --tokenizer=/mnt/models` after your
  `model.args` so the plugin resolves `*-<quant>.gguf` in the mounted cache and
  serves the cached tokenizer — no Hugging Face access in `kserve-container`.

The vLLM GGUF plugin is auto-loaded from the image (`vllm.general_plugins`) and
auto-detects the `dir:quant` model reference; no `--quantization`/`--load-format`
flags are required.

**Tokenizer from a different repo:** this is the expected pattern (the plugin's
own docs serve `repo:quant` with `--tokenizer <base-repo>`). Set
`gguf.tokenizerRepo` to the official base repo, e.g.
`Qwen/Qwen3.8-27B`, and the populate init caches that repo's tokenizer/config
files into the same subPath — the served model stays fully offline. Weights and
mmproj still come from the GGUF repo.

**Auxiliary GGUF artifacts** such as `imatrix_*.gguf` are quantization-time
inputs (used by llama.cpp to build quants, never loaded by vLLM), so they are
not auto-downloaded. Add them to `gguf.files` if you want them on the cache.

**Cache invalidation:** the `.ready` marker stores a cache-format version plus a
fingerprint of the source (`repo`/`quant`/`file`/`tokenizerRepo`/`revision`/
`files`). Changing any of those, or upgrading to a populate script with a bumped
`CACHE_FORMAT_VERSION`, makes the next pod re-download that model's subPath
automatically. Updating the chart does not restart existing pods, so scale or
delete the ISVC pods to apply a new populate script.

On every pod start the init compares each cached file's **HF etag** (via
metadata-only API calls, `list_repo_files` + `get_paths_info` — never file
content) and re-downloads in place only the files that changed or went missing.
A fully fresh cache downloads zero bytes.

**Sharing a subPath:** `subPath` is the cache key — one directory, one `.ready`
marker, and one etag manifest per subPath. Models configured with the *same*
source (repo/quant/tokenizer/revision/files) share a single download and a
single refresh; the global lock serializes them. Multiple **quants of the same
model** (same repo, tokenizer source, and revision) may also share one subPath —
each quant's file is added and kept side by side, and the plugin's
`/mnt/models:<quant>` resolution picks the right one. A different repo or
tokenizer source on the same subPath is a conflict and fails at render time.

## vLLM torch.compile cache

vLLM's AOT compile (~5 minutes per pod) lands in `$VLLM_CACHE_ROOT`; the stock
runtime puts that under `/tmp`, which is ephemeral, so every pod recompiles.
Enable the shared cache to persist the artifacts on the same PVC, inside the
model's own subPath:

```yaml
vllm:
  compileCache:
    enabled: true
```

That renders, per model entry named `qwen3-8b-gguf` with
`subPath: hf/qwen3-8b-gguf`:

- PVC layout: `hf/qwen3-8b-gguf/cache/qwen3-8b-gguf/`
- `kserve-container` reuses the existing `kserve-pvc-source` volume with two
  mounts: the read-only model mount at `/mnt/models`
  (subPath `hf/qwen3-8b-gguf`) and the read-write compile-cache mount at
  `/mnt/cache` (subPath `hf/qwen3-8b-gguf/cache/qwen3-8b-gguf`) — no second
  volume for the claim (two volumes on one claim stall the pod on some
  backends). The chart renders both mounts itself because KServe's webhook
  skips adding `/mnt/models` once a `kserve-pvc-source` mount exists (it dedups
  volumeMounts by volume name).
- `VLLM_CACHE_ROOT=/mnt/cache` — a static per-container path; per-model
  separation comes from the mount subPath, not the env value

The populate init pre-creates `hf/qwen3-8b-gguf/cache/qwen3-8b-gguf/` on the PVC
so the server's subPath mount always resolves, including on the first pod.

The first pod compiles once; every later pod (scale-out, restart, new node)
reuses the artifacts. Adjust the paths via
`vllm.compileCache.mountPath` / `.cacheDir`.

**Why not an init container?** Compilation runs during model load, so a
compile-warmup init would load the weights a second time and needs its own GPU
allocation; the first pod would take *longer* (`2× weights + compile`). If even
the first serving pod must skip compilation, pre-generate once with a one-off
warmup Job (same vLLM image, same engine args, `VLLM_CACHE_ROOT` on the PVC)
before scaling — not a per-pod init.

**Caveats:** the cache hash includes vLLM/torch versions, engine args, and the
model, so image or config changes create new hash dirs (clean them up
periodically); nodes must have the same GPU architecture for the artifacts to
be reusable; the compile cache mount shares the model subPath with the
read-only `/mnt/models` mount, which requires your RWX backend to support both
mounts.

## Values

| Key | Default | Description |
|---|---|---|
| `populateScript.mountPath` | `/opt/kserve-populate` | Where the script ConfigMap is mounted. |
| `populate.image.repository` / `.tag` | `kserve/storage-initializer` / `latest` | **Must match** your deployed storage-initializer tag. |
| `populate.cacheRoot` | `/cache` | PVC mount path in the init container (PVC root). |
| `populate.hfHome` | `<cacheRoot>/.hf-home` | HF cache dir (`HF_HOME`) on the cache PVC (already mounted rw; no extra volume). Set to `/tmp` for an ephemeral per-pod cache. |
| `populate.resources` | `{}` | Default init-container resources (none by default; per model: `models[].initResources`). |
| `cache.pvc.name` / `.existingClaim` | `model-cache` / `""` | Cache PVC name, or bring your own. |
| `cache.pvc.storageClassName` | `""` | RWX provisioner (nfs-csi, cephfs, efs, …). |
| `cache.pvc.accessModes` | `[ReadWriteMany]` | PVC access modes; use `[ReadWriteOnce]` for a same-node (node-level) cache. |
| `cache.pvc.size` | `500Gi` | Cache capacity. |
| `hf.tokenSecret.name` / `.key` | `""` / `HF_TOKEN` | Existing secret holding the HF token (gated repos). |
| `models[].name` | — | ISVC name. |
| `models[].hfUri` | — | Source URI, e.g. `hf://owner/model[:revision]`. |
| `models[].gguf.repo` | — | HF repo with `.gguf` weights (GGUF mode; replaces `hfUri`). |
| `models[].gguf.quant` | — | One quantized file type to cache/serve, e.g. `Q8_0`, `Q4_K_M`. |
| `models[].gguf.file` | `""` | Exact repo-relative filename override for non-standard names. |
| `models[].gguf.tokenizerRepo` | `repo` | Where tokenizer/config files come from. |
| `models[].gguf.revision` | `""` | HF revision for all GGUF downloads. |
| `models[].gguf.files` | `[]` | Additional repo-relative files to cache (repeatable; mmproj is auto-detected otherwise). |
| `models[].subPath` | — | PVC path; the shared key (init dest + `pvc://` subPath). |
| `models[].predictor` | `{}` | Merged into `spec.predictor` (e.g. `minReplicas`, `affinity`). |
| `models[].model` | `{}` | Merged into `spec.predictor.model` (e.g. `args`, `resources`, `runtime`). |
| `models[].initResources` | `populate.resources` | Per-model populate init resources. |
| `models[].annotations` | `{}` | ISVC `metadata.annotations`. |

## Hugging Face token

KServe's credential builder injects `HF_TOKEN` from any ServiceAccount secret
whose data has an `HF_TOKEN` key — but only into the webhook-created
storage-initializer, which `pvc://` does **not** create. So this chart wires
`HF_TOKEN` into the `populate-hf` init container explicitly, using the same
`HF_TOKEN` key — one secret works both ways. Set `hf.tokenSecret.name` to enable.

## Notes & caveats

- **PVC write access:** the chart no longer sets `fsGroup`. The populate init
  runs as the stock image's uid (1000); make sure it can write the PVC (RWX
  permissions, a storage class that sets the gid, or run the image as root).
- **Read-only is per-mount:** KServe sets `readOnly` on the server's
  `volumeMount` (not the pod volume), so the init container mounts the same PVC
  read-write while `kserve-container` mounts it read-only.
- **Access modes:** defaults to `ReadWriteMany` (cache shared across nodes). For
  a node-level cache where every pod lands on one node, set
  `cache.pvc.accessModes: [ReadWriteOnce]`. `rename()` must be atomic on the
  backend; `flock` on NFS is emulated via POSIX locks (works on modern Linux).
- **Knative:** the script ConfigMap is mounted **read-only** on the init
  container — Knative rejects ConfigMap/Secret volume mounts that aren't readOnly.
- **HF cache (`HF_HOME`):** the init container sets `HF_HOME` (default
  `<cacheRoot>/.hf-home`, on the already-mounted cache PVC — no extra volume) plus
  KServe's standard HF transfer/xet env vars. Without an explicit `HF_HOME`,
  `$HOME` is often `/` (OpenShift random-UID SCC) and HuggingFace writes its cache
  to the unwritable `/.cache` (permission denied). The global flock serializes
  downloads per namespace, so the shared on-PVC cache is race-safe; recent
  `huggingface_hub` with `local_dir=` keeps only ref metadata there (no blobs). Set
  `populate.hfHome: /tmp` for an ephemeral per-pod cache instead.
- **Single PVC volume:** the cache PVC volume is deliberately named
  `kserve-pvc-source` (KServe's internal mount name). The webhook reuses it for
  the `pvc://` mount instead of adding a *second* PVC volume for the same claim —
  a duplicate PVC volume stalls the pod. Don't rename it.
- **No GC/eviction:** nothing reclaims the cache. Add a CronJob for stale
  `*.incomplete.*` dirs and old models; size the PVC accordingly.
