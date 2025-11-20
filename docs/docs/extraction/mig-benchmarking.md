# Complete Guide: Using MIG for nv-ingest Benchmarking

## Overview

This guide provides step-by-step instructions for configuring NVIDIA MIG (Multi-Instance GPU) and running bo20/bo767 benchmarks on nv-ingest deployed with MIG.

**What is MIG?**
- Hardware-level GPU partitioning
- Dedicated VRAM and compute per instance
- Complete isolation between workloads
- Suitable for multi-tenant deployments

**MIG vs Time-Slicing:**
- MIG: Hardware partitions with guaranteed resources
- Time-Slicing: Temporal sharing with dynamic allocation
- MIG provides better isolation, time-slicing provides better throughput

---

## Prerequisites

### Hardware
- NVIDIA GPUs with MIG support (tested on RTX PRO 6000 Blackwell)
- Minimum 24GB VRAM per GPU recommended
- Multiple GPUs for best results

### Software
- Ubuntu 24.04 LTS
- NVIDIA Driver 580+
- CUDA 13.0+
- MicroK8s 1.32+ with GPU Operator
- NGC API key with staging access (for NeMo OCR v1.2.0-rc2)

### Dataset
- Bo767 dataset downloaded to `/home/nvadmin/bo767/`
- 767 PDFs required for full benchmark
- Bo20 uses first 20 PDFs from bo767

---

## Step 1: Verify MIG Support

Check if your GPUs support MIG:

```bash
# Check GPU model
nvidia-smi --query-gpu=name --format=csv,noheader

# Try to check MIG modes (if MIG supported, this won't error)
nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader

# List available MIG profiles
nvidia-smi mig -lgip
```

**Supported GPUs**: A100, H100, A30, RTX PRO 6000 Blackwell (and others)

**Available MIG Profiles for RTX PRO 6000:**
- `1g.24gb` (ID: 14): 4 instances/GPU, 23.62GB each, 46 SMs
- `2g.48gb` (ID: 5): 2 instances/GPU, 47.38GB each, 94 SMs
- `4g.96gb` (ID: 0): 1 instance/GPU, 95.00GB (full GPU), 188 SMs

---

## Step 2: Clean Up Existing Deployments

Before enabling MIG, remove any existing nv-ingest deployments:

```bash
# Uninstall nv-ingest if deployed
sudo microk8s helm3 uninstall nv-ingest -n nv-ingest 2>/dev/null || true

# Wait for pods to terminate
sleep 30

# Verify no GPU processes running
nvidia-smi
```

---

## Step 3: Enable MIG Mode

**Important**: MIG mode can be enabled without reboot on most systems.

### Enable MIG on All GPUs

```bash
# For this guide, we'll use all 8 GPUs
# (You can use fewer if desired)

for gpu in 0 1 2 3 4 5 6 7; do
  sudo nvidia-smi -i $gpu -mig 1
  echo "✅ MIG enabled on GPU $gpu"
done

# Verify MIG is enabled
nvidia-smi --query-gpu=index,mig.mode.current --format=csv
```

**Expected Output:**
```
0, Enabled
1, Enabled
...
7, Enabled
```

---

## Step 4: Create MIG Instances

### Create 4x MIG 1g.24gb Instances per GPU

This configuration creates **32 MIG instances** total (8 GPUs × 4 instances):

```bash
# Create GPU instances (profile 14 = 1g.24gb)
for gpu in 0 1 2 3 4 5 6 7; do
  sudo nvidia-smi mig -cgi 14,14,14,14 -i $gpu
  echo "✅ Created 4 MIG instances on GPU $gpu"
done

# Create compute instances for each GPU instance
for gpu in 0 1 2 3 4 5 6 7; do
  sudo nvidia-smi mig -cci -i $gpu
  echo "✅ Created compute instances on GPU $gpu"
done
```

### Verify MIG Instances

```bash
# List all MIG GPU instances
sudo nvidia-smi mig -lgi

# List MIG devices
nvidia-smi -L
```

**Expected Output:**
```
GPU 0:
  MIG 1g.24gb Device 0: (UUID: MIG-...)
  MIG 1g.24gb Device 1: (UUID: MIG-...)
  MIG 1g.24gb Device 2: (UUID: MIG-...)
  MIG 1g.24gb Device 3: (UUID: MIG-...)
...
GPU 7:
  MIG 1g.24gb Device 0-3: (UUIDs...)
```

---

## Step 5: Configure Kubernetes for MIG

### Update GPU Operator Cluster Policy

```bash
# Remove any time-slicing configuration and enable MIG
sudo microk8s kubectl patch clusterpolicies.nvidia.com/cluster-policy \
  -n gpu-operator-resources --type merge \
  -p '{"spec": {"mig": {"strategy": "single"}, "devicePlugin": {"config": {"name": "", "default": ""}}}}'

# Restart device plugin to detect MIG devices
sudo microk8s kubectl delete pod -n gpu-operator-resources \
  -l app=nvidia-device-plugin-daemonset

# Wait for device plugin to restart
sleep 90
```

### Verify Kubernetes Detects MIG Devices

```bash
sudo microk8s kubectl get nodes -o json | \
  jq '.items[0].status.allocatable | with_entries(select(.key | contains("nvidia.com")))'
```

**Expected Output:**
```json
{
  "nvidia.com/gpu": "32"
}
```

✅ Kubernetes now sees 32 MIG devices!

---

## Step 6: Deploy nv-ingest on MIG

### Deploy Single nv-ingest Instance

```bash
export NGC_API_KEY='<your-ngc-key-with-staging-access>'
NAMESPACE=nv-ingest

sudo microk8s helm3 install \
  nv-ingest \
  https://helm.ngc.nvidia.com/nvidia/nemo-microservices/charts/nv-ingest-25.9.0.tgz \
  -n ${NAMESPACE} --create-namespace \
  --username '$oauthtoken' \
  --password "${NGC_API_KEY}" \
  --set ngcImagePullSecret.create=true \
  --set ngcImagePullSecret.password="${NGC_API_KEY}" \
  --set ngcApiSecret.create=true \
  --set ngcApiSecret.password="${NGC_API_KEY}" \
  --set nemoretriever-ocr.deployed=true \
  --set nemoretriever-ocr.image.repository="nvcr.io/nvstaging/nim/nemoretriever-ocr-v1" \
  --set nemoretriever-ocr.image.tag="1.2.0-rc2-latest-release-38498041" \
  --set paddleocr-nim.deployed=false \
  --set envVars.OCR_MODEL_NAME="scene_text_ensemble" \
  --set image.repository="nvcr.io/nvidia/nemo-microservices/nv-ingest" \
  --set image.tag="25.9.0" \
  --set redis.image.repository=bitnamilegacy/redis \
  --set redis.image.tag=8.2.1-debian-12-r0 \
  --wait --timeout 20m
```

### Verify Deployment

```bash
# Check all pods are running
sudo microk8s kubectl get pods -n nv-ingest

# Verify MIG device allocation
nvidia-smi

# Expected: 5-7 MIG devices in use by nv-ingest NIMs
```

### Create MinIO a-bucket

```bash
MINIO_POD=$(sudo microk8s kubectl get pod -n nv-ingest | grep minio | awk '{print $1}')
sudo microk8s kubectl exec -n nv-ingest $MINIO_POD -- \
  sh -c 'mc alias set myminio http://localhost:9000 minioadmin minioadmin && \
         mc mb myminio/a-bucket --ignore-existing'
```

---

## Step 7: Run Bo20 Benchmark on MIG

### Deploy Bo20 Test Pod

```bash
cd /home/nvadmin/nv-ingest

# Create bo20 test pod
sudo microk8s kubectl apply -f bo767_benchmark_setup/configs/bo20_eval_pod.yaml

# Monitor progress
sudo microk8s kubectl logs -f bo20-eval -n nv-ingest
```

### Expected Bo20 Results on MIG

```
Files processed: 20/20 (100%)
Total entities: 841
Ingestion time: ~86 seconds
Throughput: ~5.78 pages/sec
E2E time: ~87 seconds
```

### Check Results

```bash
# View metrics
cat /home/nvadmin/nv-ingest/test_results/bo20_full.json | jq '.'

# View log
tail -100 /home/nvadmin/nv-ingest/evaluation/bo20_run_*.log
```

---

## Step 8: Run Bo767 Benchmark on MIG

### Deploy Bo767 Test Pod

```bash
cd /home/nvadmin/nv-ingest

# Create bo767 test pod
sudo microk8s kubectl apply -f bo767_benchmark_setup/configs/bo767_eval_pod.yaml

# Monitor progress (live)
sudo microk8s kubectl logs -f bo767-eval -n nv-ingest

# Or watch log file
tail -f /home/nvadmin/nv-ingest/evaluation/bo767_run_*.log
```

### Expected Runtime

- **Total**: ~37-40 minutes
  - Ingestion: ~35-36 minutes (767 PDFs)
  - Indexing: ~2 minutes
  - Recall testing: ~1-2 minutes

### Expected Bo767 Results on MIG

```json
{
  "files_processed": "767/767 (100%)",
  "total_entities": "85,396",
  "ingestion_time": "~2,133 seconds (~35.6 min)",
  "ingestion_throughput": "~25.7 pages/sec",
  "e2e_time": "~2,264 seconds (~37.7 min)",
  "e2e_throughput": "~24.2 pages/sec",
  "success_rate": "100%",
  "recall_@10": {
    "text": "~0.924",
    "chart": "~0.907",
    "table": "~0.672",
    "multimodal": "~0.893"
  }
}
```

### Monitor MIG Device Usage

```bash
# Watch MIG devices in real-time
watch -n 5 nvidia-smi

# Check specific MIG device usage
nvidia-smi | grep -A 40 'MIG devices:'
```

### Check Results

```bash
# View complete metrics
cat /home/nvadmin/nv-ingest/test_results/bo767_full.json | jq '.'

# View recall scores
cat /home/nvadmin/nv-ingest/test_results/*_recall.json

# View complete log
less /home/nvadmin/nv-ingest/evaluation/bo767_run_*.log
```

---

## Performance Comparison: MIG vs Time-Slicing

### Bo767 Benchmark Results

| Metric | Time-Slicing | MIG | Difference |
|--------|--------------|-----|------------|
| **Ingestion Time** | 31.0 min | 35.6 min | MIG 14% slower |
| **Throughput** | 29.4 pages/sec | 25.7 pages/sec | MIG 14% slower |
| **E2E Time** | 33.0 min | 37.7 min | MIG 14% slower |
| **Success Rate** | 100% | 100% | Same |
| **Recall @10** | 0.89 | 0.89 | Same |
| **Query QPS** | 68.3 | 7.4 | MIG 89% slower |

### Why Time-Slicing is Faster

1. **More virtual GPUs**: 128 (8×16) vs 32 (8×4)
2. **Dynamic resource sharing**: NIMs can burst when GPU available
3. **Better parallelization**: More concurrent inference requests
4. **Flexible allocation**: Not limited by fixed partition sizes

### When to Use MIG Despite Lower Throughput

✅ **Multi-tenant deployments** with strict isolation  
✅ **QoS guarantees** required for SLAs  
✅ **Security/compliance** mandates hardware isolation  
✅ **Billing/chargeback** per GPU resource allocation  
✅ **Predictable performance** more important than peak throughput  

---

## Troubleshooting

### Issue 1: MIG Mode Not Enabled After Reboot

**Symptom**: `nvidia-smi --query-gpu=mig.mode.current` shows "Disabled"

**Solution**: Re-enable MIG mode:
```bash
for gpu in 0 1 2 3 4 5 6 7; do sudo nvidia-smi -i $gpu -mig 1; done
```

### Issue 2: Cannot Create MIG Instances

**Symptom**: "In use by another client" or "Insufficient permissions"

**Solution**:
```bash
# Make sure no processes are using the GPU
sudo nvidia-smi

# If processes exist, stop them or reboot
sudo reboot

# After reboot, create instances before any GPU workload starts
```

### Issue 3: Kubernetes Shows 0 or Wrong Number of GPUs

**Symptom**: `nvidia.com/gpu: "0"` or incorrect count

**Solution**:
```bash
# Check cluster policy
sudo microk8s kubectl get clusterpolicy cluster-policy -n gpu-operator-resources -o yaml | grep -A 5 'mig:'

# Ensure MIG strategy is set
sudo microk8s kubectl patch clusterpolicies.nvidia.com/cluster-policy \
  -n gpu-operator-resources --type merge \
  -p '{"spec": {"mig": {"strategy": "single"}}}'

# Remove time-slicing config
sudo microk8s kubectl patch clusterpolicies.nvidia.com/cluster-policy \
  -n gpu-operator-resources --type merge \
  -p '{"spec": {"devicePlugin": {"config": {"name": "", "default": ""}}}}'

# Restart device plugin
sudo microk8s kubectl delete pod -n gpu-operator-resources -l app=nvidia-device-plugin-daemonset
sleep 90

# Verify
sudo microk8s kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'
```

### Issue 4: nv-ingest Pods CrashLoopBackOff

**Symptom**: Pods repeatedly crashing or not starting

**Possible Causes**:
1. **Insufficient MIG device memory** (24GB per 1g.24gb instance)
2. **Too many pods requesting GPUs** (only 32 available)
3. **Image pull errors**

**Solution**:
```bash
# Check pod events
sudo microk8s kubectl describe pod <pod-name> -n nv-ingest

# Check available MIG devices
sudo microk8s kubectl get nodes -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}'

# Reduce replicas if needed or use larger MIG profile (2g.48gb)
```

### Issue 5: Slow Query Performance

**Symptom**: Query QPS much lower than expected (~7 vs 68 on time-slicing)

**Cause**: MIG instances limit concurrent execution

**This is expected behavior**: MIG provides isolation at cost of parallelization.

**Options**:
1. Accept lower QPS (trade-off for isolation)
2. Use more MIG devices per nv-ingest instance
3. Switch to time-slicing for better query throughput

---

## MIG Configuration Commands Reference

### Basic MIG Commands

```bash
# Enable MIG on GPU 0
sudo nvidia-smi -i 0 -mig 1

# Disable MIG on GPU 0
sudo nvidia-smi -i 0 -mig 0

# List available MIG profiles for GPU 0
nvidia-smi mig -lgip -i 0

# Create MIG instance (profile 14 = 1g.24gb)
sudo nvidia-smi mig -cgi 14 -i 0

# Create multiple instances at once
sudo nvidia-smi mig -cgi 14,14,14,14 -i 0

# Create compute instances
sudo nvidia-smi mig -cci -i 0

# List GPU instances
sudo nvidia-smi mig -lgi

# List compute instances
sudo nvidia-smi mig -lci

# Destroy all compute instances on GPU 0
sudo nvidia-smi mig -dci -i 0

# Destroy all GPU instances on GPU 0
sudo nvidia-smi mig -dgi -i 0
```

### Cleanup: Return to Time-Slicing

```bash
# Destroy all MIG instances
for gpu in 0 1 2 3 4 5 6 7; do
  sudo nvidia-smi mig -dci -i $gpu
  sudo nvidia-smi mig -dgi -i $gpu
done

# Disable MIG mode
for gpu in 0 1 2 3 4 5 6 7; do
  sudo nvidia-smi -i $gpu -mig 0
done

# Verify
nvidia-smi --query-gpu=mig.mode.current --format=csv

# Reconfigure GPU Operator for time-slicing
sudo microk8s kubectl apply -f ../bo767_benchmark_setup/configs/time-slicing-config.yaml

# (See bo767_benchmark_setup docs for complete time-slicing setup)
```

---

## Advanced: Multi-Instance MIG Deployment

**Note**: The nv-ingest Helm chart has hardcoded service names that make multi-instance deployment complex. Use the single-instance approach above unless you need strict multi-tenancy.

### For Multi-Instance (Advanced)

Create custom Helm values files with service name overrides:

```yaml
# values-instance-a.yaml
fullnameOverride: "nv-ingest-a"
redis:
  fullnameOverride: "nv-ingest-redis"
minio:
  fullnameOverride: "nv-ingest-minio"
milvus:
  fullnameOverride: "nv-ingest-milvus"
# ... override all service references
```

Then deploy:
```bash
helm install nv-ingest-a ... -n nv-ingest-a -f values-instance-a.yaml
helm install nv-ingest-b ... -n nv-ingest-b -f values-instance-b.yaml
```

---

## Expected Performance Characteristics

### MIG Performance Profile

**Strengths:**
- ✅ Consistent, predictable latency
- ✅ No interference from other workloads
- ✅ Guaranteed memory allocation
- ✅ Suitable for SLA-based deployments

**Limitations:**
- ⚠️ Lower peak throughput vs time-slicing (~14% slower)
- ⚠️ Reduced query parallelization (~90% lower QPS)
- ⚠️ Fixed resource allocation (less flexible)
- ⚠️ More complex setup

### Throughput Comparison

| Dataset | MIG (pages/sec) | Time-Slicing (pages/sec) | Difference |
|---------|-----------------|-------------------------|------------|
| Bo20 | ~5.78 | ~5.39 | MIG +7% (small dataset) |
| Bo767 | ~25.7 | ~29.4 | Time-Slicing +14% (large dataset) |

**Observation**: On small datasets (bo20), MIG is competitive. On large datasets (bo767), time-slicing's better parallelization wins.

---

## Best Practices for MIG with nv-ingest

### 1. Choose Right MIG Profile

- **1g.24gb**: Good for most NIMs, 4 instances/GPU
- **2g.48gb**: For memory-intensive NIMs, 2 instances/GPU
- **4g.96gb**: Full GPU for Milvus or large models

### 2. Calculate MIG Devices Needed

nv-ingest requires approximately:
- nv-ingest main: 1 GPU
- nemoretriever-ocr: 1 GPU
- nemoretriever-page-elements: 1 GPU
- nemoretriever-table-structure: 1 GPU
- nemoretriever-graphic-elements: 1 GPU
- nvidia-nim-llama-embedqa: 1 GPU
- milvus-standalone: 1 GPU

**Total: ~6-7 MIG devices per nv-ingest instance**

### 3. Monitor MIG Utilization

```bash
# Real-time monitoring
watch -n 2 nvidia-smi

# Check MIG memory usage
nvidia-smi | grep 'MIG 1g.24gb'

# Check processes per MIG device
nvidia-smi | grep -A 20 'Processes:'
```

### 4. Persistence Mode

Enable persistence mode for better MIG performance:

```bash
sudo nvidia-smi -pm 1
```

---

## Quick Reference Scripts

### One-Click MIG Setup Script

```bash
#!/bin/bash
# setup_mig_for_nvingest.sh

# Enable MIG on all GPUs
for gpu in {0..7}; do sudo nvidia-smi -i $gpu -mig 1; done

# Create 4x 1g.24gb instances per GPU (32 total)
for gpu in {0..7}; do
  sudo nvidia-smi mig -cgi 14,14,14,14 -i $gpu
  sudo nvidia-smi mig -cci -i $gpu
done

# Configure Kubernetes
sudo microk8s kubectl patch clusterpolicies.nvidia.com/cluster-policy \
  -n gpu-operator-resources --type merge \
  -p '{"spec": {"mig": {"strategy": "single"}, "devicePlugin": {"config": {"name": "", "default": ""}}}}'

sudo microk8s kubectl delete pod -n gpu-operator-resources -l app=nvidia-device-plugin-daemonset

echo "✅ MIG setup complete. Wait 90s for device plugin to restart."
```

### One-Click MIG Cleanup Script

```bash
#!/bin/bash
# cleanup_mig.sh

# Destroy all MIG instances
for gpu in {0..7}; do
  sudo nvidia-smi mig -dci -i $gpu 2>/dev/null
  sudo nvidia-smi mig -dgi -i $gpu 2>/dev/null
  sudo nvidia-smi -i $gpu -mig 0
done

echo "✅ MIG disabled. Reboot for changes to take full effect."
```

---

## Benchmark Execution Checklist

### Pre-Benchmark

- [ ] MIG enabled on all GPUs
- [ ] 32 MIG instances created and verified
- [ ] Kubernetes detecting MIG devices (32)
- [ ] nv-ingest deployed and all pods running
- [ ] MinIO a-bucket created
- [ ] Bo767 dataset available at `/home/nvadmin/bo767/`

### During Benchmark

- [ ] Monitor `sudo microk8s kubectl logs -f <pod> -n nv-ingest`
- [ ] Watch MIG device usage with `nvidia-smi`
- [ ] Check for errors in nv-ingest service logs

### Post-Benchmark

- [ ] Verify results in `test_results/bo767_full.json`
- [ ] Check log file in `evaluation/bo767_run_*.log`
- [ ] Compare with time-slicing baseline
- [ ] Document any issues encountered

---

## Results Files

After running benchmarks, you'll find:

**Metrics:**
```
/home/nvadmin/nv-ingest/test_results/
├── bo767_full.json          # Complete bo767 metrics
├── bo20_full.json           # Bo20 metrics
├── *_table_recall.json      # Table recall scores
├── *_chart_recall.json      # Chart recall scores
├── *_text_recall.json       # Text recall scores
└── *_multimodal_recall.json # Multimodal recall scores
```

**Logs:**
```
/home/nvadmin/nv-ingest/evaluation/
├── bo767_run_YYYYMMDD_HHMMSS.log  # Full bo767 execution log
└── bo20_run_YYYYMMDD_HHMMSS.log   # Full bo20 execution log
```

---

## Summary

### MIG Setup: ✅ Working
- 32 MIG instances successfully created
- Kubernetes integration working
- nv-ingest deploys and runs on MIG
- 100% success rate on bo767 and bo20

### Performance: Time-Slicing is Faster
- Ingestion: Time-slicing 14% faster
- Queries: Time-slicing 10x better QPS
- **Recommendation**: Use time-slicing for nv-ingest unless strict isolation required

### Use Cases for Each

**Time-Slicing** (Recommended for nv-ingest):
- Single-tenant deployments
- Maximum throughput priority
- Development and testing
- Cost-sensitive deployments

**MIG**:
- Multi-tenant with isolation requirements
- SLA-based deployments needing QoS
- Security/compliance requirements
- Multiple completely isolated nv-ingest instances

---

## Additional Resources

- [NVIDIA MIG User Guide](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/)
- [Kubernetes GPU Operator MIG Documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/gpu-operator-mig.html)
- [nv-ingest Helm MIG Example](https://github.com/NVIDIA/nv-ingest/tree/main/helm/mig)

---

**Guide Version**: 1.0  
**Last Updated**: November 19, 2025  
**Tested On**: 8x RTX PRO 6000 Blackwell, Ubuntu 24.04, MicroK8s 1.32  
**MIG Profile**: 1g.24gb (32 instances total)  
**Status**: ✅ Fully tested and validated
