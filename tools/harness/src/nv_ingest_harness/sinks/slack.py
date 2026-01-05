# Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Slack sink for posting benchmark results."""

import json
import os
import re
from typing import Any

import requests

from nv_ingest_harness.sinks.base import Sink

_POST_TEMPLATE = """
{
  "username": "nv-ingest Benchmark Runner",
  "icon_emoji": ":rocket:",
  "blocks": [
    {
      "type": "header",
      "text": {
        "type": "plain_text",
        "text": "nv-ingest Nightly Benchmark Summary"
      }
    },
    {
      "type": "section",
      "text": {
        "type": "mrkdwn",
        "text": "$EXECUTIVE_SUMMARY"
      }
    },
    {
      "type": "divider"
    },
    $REPORT_JSON_TEXT
  ]
}
"""

_BLANK_ROW = [
    {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}]},
    {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": " "}]}]},
]

# Default metrics to report for each dataset
DEFAULT_METRICS = [
    "ingestion_time_s",
    "pages_per_second",
    "result_count",
    "total_pages",
]

# Additional metrics for recall datasets
RECALL_METRICS = [
    "recall_multimodal_@5_no_reranker",
    "recall_multimodal_@5_reranker",
]


class SlackSink(Sink):
    """Slack sink that posts table-formatted results using Block Kit."""

    name: str = "slack"

    def __init__(self, sink_config: dict[str, Any]):
        self.sink_config = sink_config
        self.enabled = sink_config.get("enabled", True)
        self.webhook_url = sink_config.get("webhook_url") or os.environ.get("SLACK_WEBHOOK_URL")
        if not self.webhook_url and self.enabled:
            print("Warning: SlackSink enabled but no webhook_url configured. Set SLACK_WEBHOOK_URL env var.")
            self.enabled = False

        self.default_metrics = sink_config.get("default_metrics", DEFAULT_METRICS)
        self.session_name: str | None = None
        self.env_data: dict[str, Any] = {}
        self.results_to_report: list[dict[str, Any]] = []

    def initialize(self, session_name: str, env_data: dict[str, Any]) -> None:
        self.session_name = session_name
        self.env_data = env_data
        self.results_to_report = []

    def process_result(
        self,
        result: dict[str, Any],
        entry_config: dict[str, Any] | None = None,
    ) -> None:
        if entry_config and "additional_metrics" in entry_config:
            result = dict(result)
            result["_additional_metrics"] = entry_config["additional_metrics"]
        self.results_to_report.append(result)

    def finalize(self) -> None:
        if not self.enabled:
            print("SlackSink: Not enabled, skipping post.")
            return
        self._post()
        print("SlackSink: Posted results to Slack.")

    def _post(self) -> None:
        indent = "-    "

        rows = []
        all_passed = all(r.get("success", False) for r in self.results_to_report)
        passed_count = sum(1 for r in self.results_to_report if r.get("success", False))
        total_count = len(self.results_to_report)
        overall_status = (
            f"✅ success ({passed_count}/{total_count})"
            if all_passed
            else f"❌ FAILED ({passed_count}/{total_count} passed)"
        )
        rows.append(self._two_column_row_bold("OVERALL STATUS", overall_status))

        for result in self.results_to_report:
            dataset = result.get("dataset", "unknown")
            success = result.get("success", False)
            status_str = "✅ success" if success else "❌ FAILED"
            rows.append(self._two_column_row_bold(f"{indent}{dataset}", status_str))

        rows.append(_BLANK_ROW)
        rows.append(self._two_column_row_bold("ENVIRONMENT", " "))
        for key, val in self.env_data.items():
            if key in {"packages_txt"}:
                continue
            fkey, fval = self._format_metric_value(key, val)
            rows.append(self._two_column_row(f"{indent}{fkey}", fval))

        rows.append(_BLANK_ROW)
        rows.append(self._two_column_row_bold("RESULTS", " "))
        for result in self.results_to_report:
            dataset = result.get("dataset", "unknown")
            success = result.get("success", False)
            status_str = "✅ success" if success else "❌ FAILED"
            rows.append(self._two_column_row_bold(dataset, status_str))

            metrics = result.get("metrics", {})
            metrics_to_report = list(self.default_metrics)
            for recall_metric in RECALL_METRICS:
                if recall_metric in metrics:
                    metrics_to_report.append(recall_metric)

            for metric_name in metrics_to_report:
                if metric_name in metrics:
                    metric_val = metrics[metric_name]
                    fkey, fval = self._format_metric_value(metric_name, metric_val)

                    if metric_val is not None:
                        if metric_name == "result_count":
                            expected = result.get("expected_result_count")
                            if expected:
                                fval = f"{metric_val}/{expected}"
                        elif metric_name == "total_pages":
                            expected = result.get("expected_total_pages")
                            if expected:
                                fval = f"{metric_val}/{expected}"

                    rows.append(self._two_column_row(f"{indent}{fkey}", fval))

            requirements_status = result.get("requirements_status", [])
            if requirements_status:
                all_met = all(r.get("status") == "pass" for r in requirements_status)
                if all_met:
                    rows.append(self._two_column_row(f"{indent}All requirements met", "✅"))
                else:
                    for req in requirements_status:
                        if req.get("status") != "pass":
                            rows.append(
                                self._two_column_row(
                                    f"{indent}{req.get('metric', 'unknown')}", f"❌ {req.get('message', 'failed')}"
                                )
                            )

            rows.append(_BLANK_ROW)

        if self.results_to_report:
            rows.pop(-1)

        table_dict = {"type": "table", "rows": rows}
        report_json_text = json.dumps(table_dict, indent=2)
        exec_summary = f"Session: `{self.session_name}`"
        message_values = {
            "REPORT_JSON_TEXT": report_json_text,
            "EXECUTIVE_SUMMARY": exec_summary,
        }
        payload = self._substitute_template(_POST_TEMPLATE, message_values).strip()
        response = requests.post(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if not response.ok:
            print(f"SlackSink: Failed to post (status={response.status_code}): {response.text}")

    @staticmethod
    def _substitute_template(template_str: str, values: dict[str, str]) -> str:
        def replacer(match: re.Match[str]) -> str:
            var_with_dollar = match.group(0)
            varname = var_with_dollar[1:]
            return str(values.get(varname, var_with_dollar))

        return re.sub(r"\$[A-Za-z0-9_]+", replacer, template_str)

    @staticmethod
    def _two_column_row(left: str, right: str) -> list[dict[str, Any]]:
        return [
            {
                "type": "rich_text",
                "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": left}]}],
            },
            {
                "type": "rich_text",
                "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": right}]}],
            },
        ]

    @staticmethod
    def _two_column_row_bold(left: str, right: str) -> list[dict[str, Any]]:
        return [
            {
                "type": "rich_text",
                "elements": [
                    {"type": "rich_text_section", "elements": [{"type": "text", "text": left, "style": {"bold": True}}]}
                ],
            },
            {
                "type": "rich_text",
                "elements": [{"type": "rich_text_section", "elements": [{"type": "text", "text": right}]}],
            },
        ]

    @staticmethod
    def _format_metric_value(metric: str, value: Any) -> tuple[str, str]:
        if value is None:
            return (metric, "N/A")
        if metric.endswith("_s"):
            seconds = float(value)
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            formatted = f"{seconds:.2f}s"
            if hours > 0 or minutes > 0:
                formatted += " ("
                if hours > 0:
                    formatted += f"{hours}h : "
                formatted += f"{minutes:02}m : {secs:05.2f}s)"
            return (metric, formatted)
        if metric.startswith("recall_"):
            return (metric, f"{float(value):.3f}")
        if metric == "pages_per_second":
            return (metric, f"{float(value):.2f}")
        return (metric, str(value))
