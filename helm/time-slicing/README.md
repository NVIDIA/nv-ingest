### Configure the NVIDIA Operator

Next we install the NVIDIA GPU operator to make the Kubernetes cluster aware of the GPUs.

```bash
helm install --wait --generate-name --repo https://helm.ngc.nvidia.com/nvidia \
    --namespace gpu-operator --create-namespace \
    gpu-operator --set driver.enabled=false
```

Once this is installed we can check the number of available GPUs.

```console
$ kubectl get nodes -o json | jq -r '.items[] | select(.metadata.name | test("-worker[0-9]*$")) | {name: .metadata.name, "nvidia.com/gpu": .status.allocatable["nvidia.com/gpu"]}'
{
  "name": "one-worker-one-gpu-worker",
  "nvidia.com/gpu": "1"
}
```

### Enable time-slicing

Next we want to enable time-slicing to allow more than one Pod to use this GPU. We do this by creating a time-slicing config file using [`time-slicing-config.yaml`](time-slicing-config.yaml).

```bash
kubectl apply -n gpu-operator -f time-slicing-config.yaml
```

Then update the cluster policy to use this new config.

```bash
kubectl patch clusterpolicies.nvidia.com/cluster-policy \
    -n gpu-operator --type merge \
    -p '{"spec": {"devicePlugin": {"config": {"name": "time-slicing-config", "default": "any"}}}}'
```


Now if we check the number of GPUs available we should see `16`, but note this is still just GPU `0` which we exposed to the node, just sliced into `16`.

```console
$ kubectl get nodes -o json | jq -r '.items[] | select(.metadata.name | test("-worker[0-9]*$")) | {name: .metadata.name, "nvidia.com/gpu": .status.allocatable["nvidia.com/gpu"]}'
{
  "name": "one-worker-one-gpu-worker",
  "nvidia.com/gpu": "16"
}
```
