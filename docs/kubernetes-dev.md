<!--
SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Developing with Kubernetes

Developing directly on Kubernetes gives us more confidence that things will work as expected in end user deployments.

This page describes how to use Kubernetes generally, and how to deploy nv-ingest on a local Kubernetes clusters.

> **NOTE:** _Unless otherwise noted, all commands below should be run from the root of this repo._

## Kubernetes cluster

To start you need a Kubernetes cluster.
To get started we recommend using `kind` which creates a single Docker container with a Kubernetes cluster inside it.

Also because this the `kind` cluster needs access to the GPUs on your system you need to install `kind-with-gpus`. The easiest way to do this is following the instructions laid out in this Github repo https://github.com/klueska/kind-with-gpus-examples/tree/master

Benefits of this:

- allows many developers on the same system to have isolated Kubernetes clusters
- enables easy creation/deletion of clusters

Run the following **from the root of the repo** to create a configuration file for your cluster.

```yaml
mkdir -p ./.tmp

cat <<EOF > ./.tmp/kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: nv-ingest-${USER}
nodes:
  - role: control-plane
    image: kindest/node:v1.29.2
  {{- range \$gpu := until numGPUs }}
  - role: worker
    extraMounts:
      # We inject all NVIDIA GPUs using the nvidia-container-runtime.
      # This requires 'accept-nvidia-visible-devices-as-volume-mounts = true' be set
      # in '/etc/nvidia-container-runtime/config.toml'
      - hostPath: /dev/null
        containerPath: /var/run/nvidia-container-devices/{{ \$gpu }}
  {{- end }}
EOF
```

Then use the `nvkind` CLI to create your cluster.

```shell
nvkind cluster create \
    --config-template ./.tmp/kind-config.yaml
```

You should see output like this:

```shell
Creating cluster "jdyer" ...
 ✓ Ensuring node image (kindest/node:v1.27.11) 🖼
 ✓ Preparing nodes 📦
 ✓ Writing configuration 📜
 ✓ Starting control-plane 🕹️
 ✓ Installing CNI 🔌
 ✓ Installing StorageClass 💾
Set kubectl context to "kind-jdyer"
You can now use your cluster with:

kubectl cluster-info --context kind-jdyer

Have a nice day! 👋
```

You can list clusters on the system with `kind get clusters`.

```shell
kind get clusters
# jdyer
```

You can also just use `docker ps` to see the kind container.

```shell
docker ps | grep kind
# aaf5216a3cc8   kindest/node:v1.27.11  "/usr/local/bin/entr…"   44 seconds ago   Up 42 seconds 127.0.0.1:45099->6443/tcp jdyer-control-plane
```

`kind create cluster` will do the following:

- add a context for this cluster to `${HOME}/.kube/config`, the default config file used by tools like `kubectl`
- change the default context to that one

You should be able to use `kubectl` immediately, and it should be pointed at that cluster you just created.

For example, to ensure the cluster was set up successfully, try listing nodes.

```shell
kubectl get nodes
```

If that worked, you should see a single node, like this:

```text
NAME                  STATUS   ROLES           AGE   VERSION
jdyer-control-plane   Ready    control-plane   63s   v1.27.11
```

Note: All of the containers created inside your Kubernetes cluster will not show up when you run `docker ps` as they are nested within a separate containerd namespace.

See "debugging tools" in the "Troubleshooting" section below.

## Skaffold

Now that you have a Kubernetes cluster you can use `skaffold` to build and deploy your development environment.

Skaffold does a few things for you in a single command:

- Build containers from the current directory (via `docker build`).
- Install the retriever-ingest helm charts (via `helm install`).
- Apply additional Kubernetes manifests (via `kustomize`).
- Hot reloading - skaffold watches your local directory for changes and syncs them into the Kubernetes container.
  - _for details on this, see "Hot reloading" below ([link](#hot-reloading))_
- Port forwards the -ingest service to the host.

### Directory structure

- `skaffold/sensitive/` contains any secrets or manifests you want deployed to your cluster, but not checked into git, as your local cluster is unlikely to have ESO installed. If it does, feel free to use `kind: ExternalSecret` instead.
- `skaffold/components` contains any k8s manifests you want deployed in any skaffold file. The paths are relative and can be used in either `kustomize` or `rawYaml` formats:

  ```yaml
  manifests:
    rawYaml:
      - sensitive/*.yaml
    kustomize:
      paths:
        - components/elasticsearch
  ```

- If adding a new service, try getting a helm object first. If none exists, you may have to encapsulate it with your k8s manifests in `skaffold/components`. We are a k8s shop, so manifest writing may be required from time to time.

### Prerequisites

#### Add Helm repos

The retriever-ingest service's deployment requires pulling in configurations for other services from third-party sources,
e.g. Elasticsearch, OpenTelemetry, and Postgres.

The first time you try to deploy this project to a local Kubernetes, you may need to tell
your local version of `Helm` (a package manager for Kubernetes configurations) where to find those
third-party things, by running something like the following.

```shell
helm repo add \
  nvdp \
  https://nvidia.github.io/k8s-device-plugin

helm repo add \
  zipkin \
  https://zipkin.io/zipkin-helm

helm repo add \
  opentelemetry \
  https://open-telemetry.github.io/opentelemetry-helm-charts

helm repo add \
  nvidia \
  https://helm.ngc.nvidia.com/nvidia

helm repo add \
  bitnami \
  https://charts.bitnami.com/bitnami
```

For the full list of repositories, see the `dependencies` section in [this project's Chart.yaml](../../helm/Chart.yaml).

#### Nvidia GPU Support

In order for the deployed kubernetes pods to access the Nvidia GPU resources the [Nvidia k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) must be installed. There are a multitude of configurations for this plugin but for a straight forward route to start development you can simply run.

```shell
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/deployments/static/nvidia-device-plugin.yml
```

#### Create an image pull secret

You'll also need to provide a Kubernetes Secret with credentials to pull NVIDIA-private Docker images.

For short-lived development clusters, just use your own individual credentials.

```shell
DOCKER_CONFIG_JSON=$(
    cat "${HOME}/.docker/config.json" \
    | base64 -w 0
)

cat <<EOF > ./skaffold/sensitive/imagepull.yaml
apiVersion: v1
kind: Secret
metadata:
  name: nvcrimagepullsecret
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: ${DOCKER_CONFIG_JSON}
EOF
```

An NGC personal API key is needed to access models and images hosted on NGC.
Make sure that you have followed the steps of _[Ensure you have access to NGC](./index.md#ensure-you-have-access-to-ngc)_.
Next, store the key in an environment variable:

```shell
export NGC_API_KEY="<YOUR_KEY_HERE>"
```

Then create the secret manifest with:

```shell
kubectl create secret generic ngcapisecrets \
  --from-literal=ngc_api_key="${NGC_API_KEY}" \
  --dry-run=client -o yaml \
  > skaffold/sensitive/ngcapi.yaml
```

### Deploy the service

Run the following to deploy the retriever-ingest to your cluster.

```shell
skaffold dev \
    -v info \
    -f ./skaffold/nv-ingest.skaffold.yaml \
    --kube-context "kind-nv-ingest-${USER}"
```

<details><summary>explanation of those flags (click me)</summary>

- `-v info` = print INFO-level and above logs from `skaffold` and the tools it calls (like `helm` or `kustomize`)
- `-f ./skaffold/nv-ingest.skaffold.yaml` = use configuration specific to retriever-ingest
- `--tail=false` = don't flood your console with all the logs from the deployed containers
- `--kube-context "kind-${USER}"` = target the specific Kubernetes cluster you created with `kind` above

</details>

`skaffold dev` watches your local files and automatically redeploys the app as you change those files.
It also holds control in the terminal you run it in, and handles shutting down the pods in Kubernetes when you `Ctrl + C` out of it.

You should see output similar to this:

```shell
Generating tags...
 - ...
Checking cache...
 - ...
Tags used in deployment:
 - ...
Starting deploy...
Loading images into kind cluster nodes...
 - ...
Waiting for deployments to stabilize...
Deployments stabilized in 23.08 seconds
Watching for changes...
```

When you run this command, `skaffold dev` finds a random open port on the system and exposes the retriever-ingest service on that port ([skaffold docs](https://skaffold.dev/docs/port-forwarding/)).

You can find that port in `skaffold`'s logs, in a statement like this:

```bash
Port forwarding Service/nv-ingest in namespace , remote port http -> http://0.0.0.0:4503
```

Alternatively, you can obtain it like this:

```shell
NV_INGEST_MS_PORT=$(
    ps aux \
    | grep -E "kind\-${USER} port-forward .*Service/nv-ingest" \
    | grep -o -E '[0-9]+:http' \
    | cut -d ':' -f1
)
```

To confirm that the service is deployed and working, issue a request against the port you set up port-forwarding to above.

```shell
API_HOST="http://localhost:${NV_INGEST_MS_PORT}"

curl \
  -i \
  -X GET \
  "${API_HOST}/health"
```

Additionally, running `skaffold verify` in a new terminal will run verification tests against the service (i.e. [integration tests](https://skaffold.dev/docs/verify/)). These are very lightweight health checks, and should not be confused with actual integration tests.

## Clean Up

To destroy the entire Kubernetes cluster, run the following.

```shell
kind delete cluster \
    --name "${USER}"
```

## Troubleshooting

### Debugging Tools

`kubectl` is the official CLI for Kubernetes, and supports a lot of useful functionality.

For example, to get a shell inside the `nv-ingest-ms-runtime` container in your deployment, run the following:

```shell
NV_INGEST_POD=$(
    kubectl get pods \
        --context "kind-${USER}" \
        --namespace default \
        -l 'app.kubernetes.io/instance=nv-ingest-ms-runtime' \
        --no-headers \
    | awk '{print $1}'
)
kubectl exec \
    --context "kind-${USER}" \
    --namespace default \
    pod/${NV_INGEST_POD} \
    -i \
    -t \
    -- sh
```

For an interactive, live-updating experience, try `k9s`.
To launch it, just run `k9s`.

```shell
k9s
```

You should see something like the following.

![k9s example](./media/k9s-example.png){width=80%}

For details on how to use it, see https://k9scli.io/topics/commands/.

### Installing Helm Repositories

You may encounter an error like this:

> _Error: no repository definition for https://helm.dask.org. Please add the missing repos via 'helm repo add'_

That indicates that your local installation of `Helm` (sort of a package manager for Kubernetes configurations) doesn't know
how to access a remote repository containing Kubernetes configurations.

As that error message says, run `help repo add` with that URL and an informative name.

```shell
helm repo add \
    bitnami \
    https://charts.bitnami.com/bitnami

helm repo add \
    tika \
    https://apache.jfrog.io/artifactory/tika
```

### Getting more logs from `skaffold`

You may encounter an error like this:

```shell
Generating tags...
 - retrieval-ms -> retrieval-ms:f181a78-dirty
Checking cache...
 - retrieval-ms: Found Locally
Cleaning up...
 - No resources found
building helm dependencies: exit status 1
```

Seeing only "building helm dependencies" likely means you ran `skaffold dev` or `skaffold run` in a fairly quiet mode.

Re-run those commands with something like `-v info` or `-v debug` to get more information about what specifically failed.

## References

- Helm quickstart: https://helm.sh/docs/intro/quickstart/
- `kind` docs: https://kind.sigs.k8s.io/
- `skaffold` docs: https://skaffold.dev/docs/
