{{- define "oc-serve.namespace" -}}
{{- if .Values.global.namespaceOverride -}}
{{- .Values.global.namespaceOverride -}}
{{- else -}}
{{- .Release.Namespace -}}
{{- end -}}
{{- end -}}

{{- define "oc-serve.redisAddress" -}}
{{- $user := "" -}}
{{- if and .Values.orchestrator .Values.orchestrator.ray .Values.orchestrator.ray.gcsFaultToleranceOptions .Values.orchestrator.ray.gcsFaultToleranceOptions.redisAddress -}}
{{- $user = .Values.orchestrator.ray.gcsFaultToleranceOptions.redisAddress -}}
{{- end -}}
{{- if $user -}}
{{- $user -}}
{{- else if .Values.redis.enabled -}}
  {{- /* Bitnami redis chart default service name is: <fullname>-master */ -}}
  {{- $fullname := "" -}}
  {{- if .Values.redis.fullnameOverride -}}
    {{- $fullname = .Values.redis.fullnameOverride -}}
  {{- else -}}
    {{- /* common dependency name pattern: <release>-redis */ -}}
    {{- $fullname = printf "%s-%s" .Release.Name "redis" -}}
  {{- end -}}
  {{- printf "%s-master.%s:6379" $fullname (include "oc-serve.namespace" .) -}}
{{- end -}}
{{- end -}}
