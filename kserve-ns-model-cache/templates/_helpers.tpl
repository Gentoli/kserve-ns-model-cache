{{/*
Expand the name of the chart.
*/}}
{{- define "kserve-ns-model-cache.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name (truncated to the 63-char DNS limit).
*/}}
{{- define "kserve-ns-model-cache.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart name and version as used by the chart label.
*/}}
{{- define "kserve-ns-model-cache.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "kserve-ns-model-cache.labels" -}}
helm.sh/chart: {{ include "kserve-ns-model-cache.chart" . }}
{{ include "kserve-ns-model-cache.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "kserve-ns-model-cache.selectorLabels" -}}
app.kubernetes.io/name: {{ include "kserve-ns-model-cache.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Name of the ConfigMap holding the populate script.
*/}}
{{- define "kserve-ns-model-cache.populateConfigMapName" -}}
{{- printf "%s-populate" (include "kserve-ns-model-cache.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Pod volume name for the cache PVC. MUST be "kserve-pvc-source" (KServe's
PvcSourceMountName): for a pvc:// storageUri the webhook adds a PVC volume under
that name and mounts it on kserve-container, but it SKIPS adding the volume if one
with this name already exists (AddModelMount dedups by name). Naming our podspec
volume the same makes the webhook reuse it, so the PVC appears ONCE in the pod
(two mounts: init rw at cacheRoot, server ro at /mnt/models) instead of twice --
a duplicate PVC volume for one claim stalls the pod.
*/}}
{{- define "kserve-ns-model-cache.cacheVolumeName" -}}
kserve-pvc-source
{{- end }}

{{/*
Pod volume name for the populate-script ConfigMap.
*/}}
{{- define "kserve-ns-model-cache.scriptVolumeName" -}}
populate-script
{{- end }}
