# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for common service settings in split topology."""

from __future__ import annotations

from tests.test_helm_shared_results import _render, _service_deployments


def test_split_roles_inherit_common_service_settings() -> None:
    documents = _render(
        "--set",
        "topology.mode=split",
        "--set",
        "serviceMonitor.autoEnableInSplitMode=false",
        "--set",
        "service.securityContext.allowPrivilegeEscalation=false",
        "--set",
        "service.securityContext.readOnlyRootFilesystem=true",
        "--set",
        "service.securityContext.capabilities.drop[0]=ALL",
        "--set",
        "service.securityContext.runAsNonRoot=true",
        "--set",
        "service.securityContext.seccompProfile.type=RuntimeDefault",
        "--set",
        "service.podLabels.nvbug=label-sentinel",
        "--set",
        "service.podAnnotations.nvbug=annotation-sentinel",
        "--set",
        "service.priorityClassName=priority-sentinel",
        "--set",
        "service.topologySpreadConstraints[0].maxSkew=1",
        "--set",
        "service.topologySpreadConstraints[0].topologyKey=kubernetes.io/hostname",
        "--set",
        "service.topologySpreadConstraints[0].whenUnsatisfiable=DoNotSchedule",
        "--set",
        "service.topologySpreadConstraints[0].labelSelector.matchLabels.nvbug=label-sentinel",
        "--set",
        "service.envFrom[0].configMapRef.name=env-sentinel",
        "--set",
        "service.extraVolumes[0].name=volume-sentinel",
        "--set",
        "service.extraVolumes[0].emptyDir.medium=Memory",
        "--set",
        "service.extraVolumeMounts[0].name=volume-sentinel",
        "--set",
        "service.extraVolumeMounts[0].mountPath=/volume-sentinel",
        "--set",
        "service.terminationGracePeriodSeconds=77",
    )
    deployments = _service_deployments(documents)

    assert len(deployments) == 3
    for deployment in deployments:
        pod_template = deployment["spec"]["template"]
        pod_spec = pod_template["spec"]
        container = next(item for item in pod_spec["containers"] if item["name"] == "nemo-retriever")

        assert pod_template["metadata"]["labels"]["nvbug"] == "label-sentinel"
        assert pod_template["metadata"]["annotations"]["nvbug"] == "annotation-sentinel"
        assert pod_spec["priorityClassName"] == "priority-sentinel"
        assert pod_spec["terminationGracePeriodSeconds"] == 77
        assert pod_spec["topologySpreadConstraints"] == [
            {
                "labelSelector": {"matchLabels": {"nvbug": "label-sentinel"}},
                "maxSkew": 1,
                "topologyKey": "kubernetes.io/hostname",
                "whenUnsatisfiable": "DoNotSchedule",
            }
        ]
        assert container["securityContext"] == {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
            "readOnlyRootFilesystem": True,
            "runAsNonRoot": True,
            "seccompProfile": {"type": "RuntimeDefault"},
        }
        assert container["envFrom"] == [{"configMapRef": {"name": "env-sentinel"}}]
        assert {"name": "volume-sentinel", "mountPath": "/volume-sentinel"} in container["volumeMounts"]
        assert {"name": "volume-sentinel", "emptyDir": {"medium": "Memory"}} in pod_spec["volumes"]
        component = deployment["metadata"]["labels"]["app.kubernetes.io/component"]
        init_containers = pod_spec.get("initContainers", [])
        if component == "gateway":
            assert init_containers == []
        else:
            wait_for_gateway = next(item for item in init_containers if item["name"] == "wait-for-gateway")
            assert wait_for_gateway["securityContext"] == container["securityContext"]
