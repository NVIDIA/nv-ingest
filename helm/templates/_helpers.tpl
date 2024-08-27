{{/*
Expand the name of the chart.
*/}}
{{- define "nv-ingest.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "nv-ingest.fullname" -}}
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
Create chart name and version as used by the chart label.
*/}}
{{- define "nv-ingest.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "nv-ingest.labels" -}}
helm.sh/chart: {{ include "nv-ingest.chart" . }}
{{ include "nv-ingest.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "nv-ingest.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nv-ingest.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "nv-ingest.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nv-ingest.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create secret to access docker registry
*/}}
{{- define "nv-ingest.imagePullSecret" }}
{{- printf "{\"auths\": {\"%s\": {\"auth\": \"%s\"}}}" .Values.imagePullSecret.registry (printf "%s:%s" .Values.imagePullSecret.username .Values.imagePullSecret.password | b64enc) | b64enc }}
{{- end }}

{{/*
Create secret to access docker registry
*/}}
{{- define "nv-ingest.ngcAPIKey" }}
{{- printf "%s" .Values.ngcSecret.password  }}
{{- end }}


{{/*
Create a deployment with the NGC Puller
*/}}
{{- define "nv-ingest.triton-statefulset" }}

{{ $tritonCommon := .Values.triton.common }}
{{ $root := .Values }}

{{- range $tName, $triton := .Values.triton.instances }}

{{- $pvcUsingTemplate := and $triton.persistence.enabled (not $triton.persistence.existingClaim) (ne $triton.persistence.accessMode "ReadWriteMany") | ternary true false }}
---

apiVersion: v1
kind: Service
metadata:
  name: "{{ $.Release.Name }}-{{ $triton.service.name }}"
  labels:
    {{- include "nv-ingest.labels" $ | nindent 4 }}
spec:
  type: ClusterIP
  ports:
    {{- if $tritonCommon.http_port }}
    - port: {{ $tritonCommon.http_port }}
      targetPort: http
      protocol: TCP
      name: http
    {{- end }}
    {{- if $tritonCommon.grpc_port }}
    - port: {{ $tritonCommon.grpc_port }}
      targetPort: grpc
      protocol: TCP
      name: grpc
    {{- end }}
    {{- if $tritonCommon.openai_port }}
    - port: {{ $tritonCommon.openai_port }}
      targetPort: http-openai
      name: http-openai
    {{- end }}
    {{- if $tritonCommon.nemo_port }}
    - port: {{ $tritonCommon.nemo_port }}
      targetPort: http-nemo
      name: http-nemo
    {{- end }}
    {{- if and $tritonCommon.metrics.enabled $tritonCommon.metrics.port }}
    - port: {{ $tritonCommon.metrics.port }}
      targetPort: metrics
      name: metrics
    {{- end }}
  selector:
    {{- include "nv-ingest.selectorLabels" $ | nindent 4 }}

---
{{- if and $triton.persistence.enabled (not $pvcUsingTemplate) (not $triton.persistence.existingClaim )}}
kind: PersistentVolumeClaim
apiVersion: v1
metadata:
  name: "{{ include "nv-ingest.fullname" $ }}-{{ $tName }}"
  labels:
    {{- include "nv-ingest.labels" $ | nindent 4 }}
  {{- with $triton.persistence.annotations  }}
  annotations:
    {{ toYaml . | indent 4 }}
  {{- end }}
spec:
  accessModes:
    - {{ $triton.persistence.accessMode | quote }}
  resources:
    requests:
      storage: {{ $triton.persistence.size | quote }}
{{- if $triton.persistence.storageClass }}
  storageClassName: "{{ $triton.persistence.storageClass }}"
{{- end }}
{{ end }}

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: "{{ $.Release.Name }}-{{ $tName }}"
  labels:
    {{- include "nv-ingest.labels" $ | nindent 4 }}
spec:
  podManagementPolicy: "Parallel"
  {{- if not $root.autoscaling.enabled }}
  replicas: {{ $triton.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "nv-ingest.selectorLabels" $ | nindent 6 }}
  serviceName: {{ include "nv-ingest.fullname" $ }}
  template:
    metadata:
      {{- with $root.podAnnotations }}
      annotations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      labels:
        {{- include "nv-ingest.selectorLabels" $ | nindent 8 }}
    spec:
      {{- with $root.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "nv-ingest.serviceAccountName" $ }}
      securityContext:
        {{- toYaml $root.podSecurityContext | nindent 8 }}
      initContainers:

        {{- range $name, $model := $triton.models }}
        - name: ngc-model-puller-{{ $name }}
          image: "{{ $root.image.repository }}:{{ $root.image.tag }}"
          command:
            - "/bin/bash"
            - "-c"
          args:
            - apt-get update && apt-get install --yes wget && wget "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.34.1/files/ngccli_linux.zip" -O "/workspace/ngccli_linux.zip" && unzip /workspace/ngccli_linux.zip && chmod u+x ngc-cli/ngc && export PATH="$PATH:/workspace/ngc-cli" &&  /scripts/ngc_pull.sh
          env:
            - name: NGC_CLI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: "ngc-api"
                  key: NGC_API_KEY
            - name: NGC_DECRYPT_KEY
              valueFrom:
                secretKeyRef:
                  name: "ngc-api"
                  key: NGC_DECRYPT_KEY
                  optional: true
            - name: STORE_MOUNT_PATH
              value: "/model-store"
            - name: NGC_CLI_ORG
              value: {{ $model.NGC_CLI_ORG | quote }}
            - name: NGC_CLI_TEAM
              value: {{ $model.NGC_CLI_TEAM | quote }}
            - name: NGC_CLI_VERSION
              value: {{ $model.NGC_CLI_VERSION | quote }}
            - name: NGC_MODEL_NAME
              value: {{ $model.NGC_MODEL_NAME | quote }}
            - name: NGC_MODEL_VERSION
              value: {{ $model.NGC_MODEL_VERSION | quote }}
            - name: MODEL_NAME
              value: {{ $model.MODEL_NAME | quote }}
            - name: TARFILE
              value: {{ $model.TARFILE | quote }}
            - name: NGC_EXE
              value: "ngc"
          volumeMounts:
            - mountPath: /model-store
              name: model-store-{{ $tName }}
              subPath: {{ $tName }}
            - name: scripts-volume
              mountPath: /scripts
        {{- end }}
        - name: clean-up
          image: ubuntu:latest
          command:
            - /bin/sh
            - -c
            - >
              rm -rf /model-store/nemo-retriever-* &&
              rm -rf /model-store/nemo-retrieval-*
          volumeMounts:
            - mountPath: /model-store
              subPath: {{ $tName }}
              name: model-store-{{ $tName }}

      containers:
        - name: {{ $tName }}
          securityContext:
            {{- toYaml $root.containerSecurityContext | nindent 12 }}
          image: "{{ $triton.image.repository }}:{{ $triton.image.tag }}"
          imagePullPolicy: {{ $root.image.pullPolicy }}
          command:
            - /bin/sh
            - -c
          args:
            - "tritonserver --model-repository=/model-store/ --log-verbose=1 || sleep 120"
          ports:
            - containerPort: 8000
              name: http
            - containerPort: {{ $tritonCommon.health_port }}
              name: health
            - containerPort: {{ $tritonCommon.grpc_port }}
              name: grpc
            {{- if $tritonCommon.metrics.enabled }}
            - containerPort: {{ $tritonCommon.metrics.port }}
              name: metrics
            {{- end }}
            - containerPort: {{ $tritonCommon.openai_port }}
              name: http-openai
            - containerPort: {{ $tritonCommon.nemo_port }}
              name: http-nemo
          {{- if $tritonCommon.livenessProbe.enabled }}
          {{- with $tritonCommon.livenessProbe }}
          livenessProbe:
            httpGet:
              path: {{ .path }}
              port: {{ .port }}
            initialDelaySeconds: {{ .initialDelaySeconds }}
            periodSeconds: {{ .periodSeconds }}
            timeoutSeconds: {{ .timeoutSeconds }}
            successThreshold: {{ .successThreshold }}
            failureThreshold: {{ .failureThreshold }}
          {{- end }}
          {{- end }}
          {{- if $tritonCommon.readinessProbe.enabled }}
          {{- with $tritonCommon.readinessProbe }}
          readinessProbe:
            httpGet:
              path: {{ .path }}
              port: {{ .port }}
            initialDelaySeconds: {{ .initialDelaySeconds }}
            periodSeconds: {{ .periodSeconds }}
            timeoutSeconds: {{ .timeoutSeconds }}
            successThreshold: {{ .successThreshold }}
            failureThreshold: {{ .failureThreshold }}
          {{- end }}
          {{- end }}
          {{- if $tritonCommon.startupProbe.enabled }}
          {{- with $tritonCommon.startupProbe }}
          startupProbe:
            httpGet:
              path: {{ .path }}
              port: {{ .port }}
            initialDelaySeconds: {{ .initialDelaySeconds }}
            periodSeconds: {{ .periodSeconds }}
            timeoutSeconds: {{ .timeoutSeconds }}
            successThreshold: {{ .successThreshold }}
            failureThreshold: {{ .failureThreshold }}
          {{- end }}
          {{- end }}
          resources:
            {{- toYaml $triton.resources | nindent 12 }}
          volumeMounts:
            - name: model-store-{{ $tName }}
              mountPath: /model-store
              subPath: {{ $tName }}
            - mountPath: /dev/shm
              name: dshm
          {{- if $root.extraVolumeMounts }}
          {{- range $k, $v := $root.extraVolumeMounts }}
            - name: {{ $k }}
              {{- toYaml $v | nindent 14 }}
          {{- end }}
          {{- end }}
      terminationGracePeriodSeconds: 60
      {{- with $root.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with $root.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with $root.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: scripts-volume
          configMap:
            name: {{ $.Release.Name }}-scripts-configmap
            defaultMode: 0555
        {{- if not $pvcUsingTemplate }}
        - name: model-store-{{ $tName }}
          {{- if $triton.persistence.enabled }}
          persistentVolumeClaim:
            {{- if $triton.persistence.existingClaim }}
            claimName: {{ $triton.persistence.existingClaim }}
            {{- else }}
            claimName:  "{{ include "nv-ingest.fullname" $ }}-{{ $tName }}"
            {{- end }}
          {{- else if $triton.hostPath.enabled }}
          hostPath:
            path: {{ $triton.hostPath.path }}
            type: DirectoryOrCreate
          {{- else }}
          emptyDir: {}
          {{- end }}
        {{- end }}
      {{- if $root.extraVolumes }}
      {{- range $k, $v := $root.extraVolumes }}
        - name: {{ $k }}
          {{- toYaml $v | nindent 10 }}
      {{- end }}
      {{- end }}
  {{- if $pvcUsingTemplate }}
  {{- with $triton.stsPersistentVolumeClaimRetentionPolicy }}
  persistentVolumeClaimRetentionPolicy:
    whenDeleted: {{ .whenDeleted }}
    whenScaled: {{ .whenScaled }}
  {{- end }}
  volumeClaimTemplates:
  - metadata:
      name: model-store-{{ $tName }}
      labels:
        {{- include "nv-ingest.labels" $ | nindent 8 }}
      {{- with $triton.persistence.annotations  }}
      annotations:
        {{ toYaml . | indent 4 }}
      {{- end }}
    spec:
      accessModes:
        - {{ $triton.persistence.accessMode | quote }}
      resources:
        requests:
          storage: {{ $triton.persistence.size | quote }}
    {{- if $triton.persistence.storageClass }}
      storageClassName: "{{ $triton.persistence.storageClass }}"
    {{- end }}
  {{- end }}

{{- end }}
{{- end }}
