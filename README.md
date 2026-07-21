# Namespace-shared RWX model cache for KServe (Hugging Face, zero-copy)

Serve HF models directly off a **namespace-scoped RWX PVC**: the first pod
fetches the model once (`flock`-guarded), and every pod — in this namespace only
— then loads it straight from the PVC with **no per-pod copy**.

This deliberately avoids the built-in `LocalModelCache` machinery, which is
backed by a NodeGroup `PersistentVolumeSpec` (hostPath/local PV) shared across
namespaces.

---

## Does InferenceService actually support these container customizations? — Yes

Verified against the source:

| Customization | Supported via | Evidence |
|---|---|---|
| `spec.predictor.initContainers` | `PodSpec.InitContainers` (full `corev1.Container`) | `pkg/apis/serving/v1beta1/podspec.go` (`InitContainers`, `patchMergeKey:"name"`) |
| `spec.predictor.volumes` | `PodSpec.Volumes` | `pkg/apis/serving/v1beta1/podspec.go` |
| `spec.predictor.securityContext` (fsGroup) | `PodSpec.SecurityContext` | `pkg/apis/serving/v1beta1/podspec.go` |
| init container `image` / `args` / `env` / `volumeMounts` / `resources` | full `corev1.Container` | same |
| These fields reach the final pod | merged by strategic-merge patch (by name) | `MergePodSpec`, `pkg/controller/v1beta1/inferenceservice/utils/utils.go:267` (called from `MergeServingRuntimeAndInferenceServiceSpecs`) |
| Not rejected by validation | only a blocked-env-var check on predictor init containers | `pkg/apis/serving/v1beta1/inference_service_validation.go:307` |
| `pvc://` → PVC mounted read-only at `/mnt/models` on `kserve-container`, **no** storage-initializer injected | single-URI PVC branch | `pkg/webhook/admission/pod/storage_initializer_injector.go` + `AddModelMount` (`pkg/utils/storage.go`) |

Constraints to respect:
- Standard predictor framework containers (`model:`) may not set a `name`.
- Init-container env vars are checked against a blocklist; `HF_TOKEN` is fine.

### Why `pvc://` and not a `ClusterStorageContainer`
The `ClusterStorageContainer` overlay is merged onto the **init container only**.
For any non-PVC `storageUri` (e.g. `hf://`), the webhook force-mounts an
`emptyDir` at `/mnt/models` on `kserve-container`, and its mount dedup is by
**volume name**, so you cannot pre-empt that path with your own PVC mount.
`pvc://` is the one path where the webhook mounts a PVC onto `kserve-container`
at `/mnt/models` — that is the zero-copy mount. The `hf://` fetch therefore
moves into a populate init container instead.

---

## How it fits together

```
                       PVC "model-cache" (RWX, per-namespace)
                       └── hf/llama-3-8b/   ← model files
                       └── .hf_llama-3-8b.lock / .ready   (siblings, not served)

populate-hf init (rw at /cache)            kserve-container (ro at /mnt/models)
  flock → if !ready:                          webhook mounts pvc://model-cache/hf/llama-3-8b
    storage-initializer hf://… → TMP          → subPath hf/llama-3-8b → /mnt/models
    mv TMP → hf/llama-3-8b (atomic)           → loads model straight off the PVC
    touch .ready
```

- **Fetch once:** `flock` on a per-model lockfile serializes the first download
  across all replicas; later pods see `.ready` and skip. One HF pull per
  namespace per model.
- **No partial reads:** download lands in `*.incomplete.<pid>` then `rename(2)`
  into place (atomic, same filesystem). Each pod's `kserve-container` starts
  only after its **own** init finishes (init containers gate the pod), and that
  init either holds the lock while downloading or exits on `.ready` — so a
  serving container never observes a half-written model.
- **Namespace isolation:** PVCs are namespaced; nothing is shared across
  namespaces (the whole point vs. the hostPath node cache).
- **The two paths are coupled:** the init `args[1]` (`hf/llama-3-8b`) must equal
  the `pvc://` subPath. Templating both from one value (Helm/kustomize) avoids drift.

---

## Deploy

```sh
# 1. Build & push the populate image (match BASE_TAG to your storage-initializer tag)
docker build -t registry.example.com/populate-hf:v1 \
  --build-arg BASE_TAG=<deployed storage-initializer tag> \
  work2/populate-hf/
docker push registry.example.com/populate-hf:v1

# 2. HF token secret (gated repos; skip if not needed)
kubectl -n team-a create secret generic hf-secret --from-literal=token=hf_xxx

# 3. Apply
kubectl apply -k work2/
```

---

## Files

- `pvc.yaml` — the RWX `model-cache` claim.
- `populate-hf/Dockerfile` + `populate-hf/populate-hf.sh` — the fetch-once init image.
- `isvc.yaml` — the InferenceService (populate init + `pvc://` zero-copy serve).
- `kustomization.yaml` — applies PVC + ISVC.

---

## Caveats

- **RWX backend + permissions:** the provisioner must honor `fsGroup: 1000`
  (NFS/cephfs generally do) or the claim must be writable by uid 1000. `rename()`
  must be atomic on the backend (true for NFS/cephfs within one filesystem).
- **No GC / eviction:** nothing reclaims the cache. Add a CronJob to remove
  stale `*.incomplete.*` staging dirs and old models; size the PVC accordingly.
- **HF snapshot is self-contained:** `snapshot_download(local_dir=…)` writes real
  files (not symlinks into an external cache), so the `mv` carries everything
  (`_download_hf`, `python/storage/kserve_storage/kserve_storage.py`).
- **Runtime reads `/mnt/models`:** the default HF/vLLM ServingRuntimes do. If
  you use a custom runtime, point its model dir at `/mnt/models`.
