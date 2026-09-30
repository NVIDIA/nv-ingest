# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pluggable configuration layer for evaluation pipelines.

Provides a single YAML/JSON file as the source of truth for an entire
evaluation run -- datasets, models, retrieval sources, scoring, and
execution parameters.

Primary entry points:
    ``load_eval_config(path)``       -- read and validate a config file
    ``build_eval_chain(config)``     -- single-model ``>>`` graph chain
    ``build_eval_pipeline(config)``  -- multi-model ``QAEvalPipeline``

Two config formats are supported.  The **new** ``models`` + ``evaluations``
schema lets you define models once and compose generator/judge combos
with per-combo run counts::

    models:
      nemotron-lightning:
        model: "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
        api_key: "${NVIDIA_API_KEY}"
      nemotron-lightning-judge:
        model: "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
        api_key: "${NVIDIA_API_KEY}"
        temperature: 0.1
        max_tokens: 4096

    evaluations:
      - generator: "nemotron-lightning"
        judge: "nemotron-lightning-judge"
        runs: 5

    execution:
      top_k: 5
      max_workers: 8

The **legacy** ``generators`` + ``judge`` format is still accepted and
auto-normalised internally::

    generators:
      - name: "nemotron"
        model: "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
        api_key: "${NVIDIA_API_KEY}"

    judge:
      model: "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b"
      api_key: "${NVIDIA_API_KEY}"
      temperature: 0.1
      max_tokens: 4096

Environment variables are expanded in string values: ``${VAR}`` resolves
to ``os.environ["VAR"]`` at load time. Secrets never live in the config.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_retriever.tools.evaluation.orchestrator import QAEvalPipeline
    from nemo_retriever.graph.pipeline_graph import Graph

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def check_unresolved_env(value: str | None, field: str, context: str) -> None:
    """Raise if *value* still contains an unexpanded ``${VAR}`` reference.

    Call this on ``api_key`` (or any secret) after ``_expand_env_vars``
    to fail fast with a clear message instead of sending a literal
    ``${VAR}`` string to an API endpoint.
    """
    if value and _ENV_VAR_RE.search(value):
        var = _ENV_VAR_RE.search(value).group(1)  # type: ignore[union-attr]
        raise ValueError(
            f"{field} for {context} contains unresolved env var ${{{var}}}. " f"Set it with: export {var}=<value>"
        )


_VALID_SECTION_SETS = (
    frozenset(("generators", "judge")),
    frozenset(("models", "evaluations")),
)


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ``${VAR}`` in string values."""
    if isinstance(value, str):

        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                logger.warning("Environment variable %s is not set", var_name)
                return match.group(0)
            return env_val

        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _normalize_config(config: dict) -> dict:
    """Normalise new ``models``/``evaluations`` format into legacy keys.

    If the config uses the new format, this synthesises ``generators``
    and ``judge`` entries so that :func:`build_eval_chain` and
    :func:`build_eval_pipeline` keep working without changes.

    The ``evaluations`` list is always present after normalisation
    (synthesised from ``generators`` + ``judge`` when using legacy format).

    Raises
    ------
    ValueError
        If ``evaluations`` specifies multiple distinct judges.
        :func:`build_eval_chain` and :func:`build_eval_pipeline` support
        only a single judge per invocation; use
        :func:`~nemo_retriever.evaluation.runner.run_eval_sweep` for
        heterogeneous sweeps -- it iterates ``evaluations`` and selects
        the correct judge per combo.
    """
    if "models" in config and "evaluations" in config:
        models = config["models"]
        evals = config["evaluations"]
        if not evals:
            raise ValueError("'evaluations' list is empty -- need at least one generator/judge combo")

        seen_generators: dict[str, dict] = {}
        for entry in evals:
            gen_key = entry["generator"]
            if gen_key not in models:
                raise ValueError(f"Evaluation references unknown model {gen_key!r}. " f"Available: {list(models)}")
            judge_key = entry["judge"]
            if judge_key not in models:
                raise ValueError(f"Evaluation references unknown model {judge_key!r}. " f"Available: {list(models)}")
            if gen_key not in seen_generators:
                seen_generators[gen_key] = {"name": gen_key, **models[gen_key]}

        config.setdefault("generators", list(seen_generators.values()))
        first_judge_key = evals[0]["judge"]
        distinct_judges = {e["judge"] for e in evals}
        if len(distinct_judges) > 1:
            raise ValueError(
                f"Config has {len(distinct_judges)} distinct judges "
                f"{sorted(distinct_judges)}. "
                "build_eval_chain() and build_eval_pipeline() support only a "
                "single judge per invocation. Use run_eval_sweep() for "
                "heterogeneous judges -- it iterates the evaluations list "
                "and uses the correct judge per-combo."
            )
        config.setdefault("judge", models[first_judge_key])

    elif "generators" in config and "judge" in config:
        generators = config["generators"]
        judge_cfg = config["judge"]
        judge_name = judge_cfg.get("name", "judge")
        config.setdefault("models", {})
        for gen in generators:
            name = gen.get("name", gen["model"])
            config["models"][name] = {k: v for k, v in gen.items() if k != "name"}
        config["models"][judge_name] = {k: v for k, v in judge_cfg.items() if k != "name"}
        config.setdefault(
            "evaluations", [{"generator": g.get("name", g["model"]), "judge": judge_name} for g in generators]
        )

    return config


def load_eval_config(path: str) -> dict:
    """Load eval config from YAML (``.yaml``/``.yml``) or JSON (``.json``) file.

    Supports ``${VAR}`` env var expansion in string values (recursive).
    YAML requires ``pyyaml`` (core dependency of nemo-retriever). JSON uses stdlib.

    Parameters
    ----------
    path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Parsed and env-var-expanded configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is unsupported or required sections are missing.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for YAML config files. "
                "It is a core dependency of nemo-retriever; reinstall with: "
                "pip install nemo-retriever"
            ) from exc
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    elif suffix == ".json":
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {suffix!r}. " "Use .yaml, .yml, or .json.")

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(raw).__name__}")

    config = _expand_env_vars(raw)

    if not any(req <= config.keys() for req in _VALID_SECTION_SETS):
        accepted = " or ".join(str(sorted(s)) for s in _VALID_SECTION_SETS)
        raise ValueError(
            f"Config must contain one of these section sets: {accepted}. " f"Got top-level keys: {sorted(config)}"
        )

    config = _normalize_config(config)

    return config


def build_eval_chain(
    config: dict,
    model_name: str | None = None,
) -> "Graph":
    """Construct a ``>>`` chain from config for single-model evaluation.

    If *model_name* is specified, uses that generator from config.
    If ``None``, uses the first generator listed.

    Returns a :class:`Graph`::

        RetrievalLoaderOperator >> QAGenerationOperator >> JudgingOperator >> ScoringOperator

    Parameters
    ----------
    config : dict
        Parsed config from :func:`load_eval_config`.
    model_name : str, optional
        Generator name to use. Defaults to the first in ``config["generators"]``.

    Returns
    -------
    Graph
        A chainable graph ready for ``.execute(None)``.
    """
    from nemo_retriever.tools.evaluation.generation import QAGenerationOperator
    from nemo_retriever.tools.evaluation.judging import JudgingOperator
    from nemo_retriever.tools.evaluation.retrieval_loader import RetrievalLoaderOperator
    from nemo_retriever.operators.graph_ops.scoring_operator import ScoringOperator

    generators = config["generators"]
    if not generators:
        raise ValueError("Config must have at least one generator")

    if model_name is not None:
        gen_cfg = next((g for g in generators if g.get("name") == model_name), None)
        if gen_cfg is None:
            available = [g.get("name", "?") for g in generators]
            raise ValueError(f"Generator {model_name!r} not found. Available: {available}")
    else:
        gen_cfg = generators[0]

    execution = config.get("execution", {})
    retrieval = config.get("retrieval", {})
    dataset = config.get("dataset", {})
    judge_cfg = config["judge"]

    retrieval_type = retrieval.get("type", "file")
    if retrieval_type != "file":
        raise ValueError(
            f"build_eval_chain() only supports retrieval.type='file', got {retrieval_type!r}. "
            "For LanceDB retrieval, use FileRetriever.from_lancedb() with run_eval_sweep()."
        )
    retrieval_json = retrieval.get("file_path", "")
    ground_truth_source = dataset.get("source", "")

    loader = RetrievalLoaderOperator(
        retrieval_json=retrieval_json,
        ground_truth_csv=ground_truth_source,
        data_dir=dataset.get("data_dir"),
        query_column=dataset.get("query_column", "query"),
        answer_column=dataset.get("answer_column", "answer"),
        top_k=execution.get("top_k", 5),
    )

    default_timeout = execution.get("timeout", 120.0)

    gen_op = QAGenerationOperator(
        model=gen_cfg["model"],
        api_base=gen_cfg.get("api_base"),
        api_key=gen_cfg.get("api_key"),
        temperature=gen_cfg.get("temperature", 0.0),
        top_p=gen_cfg.get("top_p"),
        max_tokens=gen_cfg.get("max_tokens", 4096),
        extra_params=gen_cfg.get("extra_params"),
        num_retries=gen_cfg.get("num_retries", 3),
        timeout=gen_cfg.get("timeout", default_timeout),
        max_workers=execution.get("max_workers", 8),
        reasoning_enabled=gen_cfg.get("reasoning_enabled", True),
    )

    judge_op = JudgingOperator(
        model=judge_cfg["model"],
        api_base=judge_cfg.get("api_base"),
        api_key=judge_cfg.get("api_key"),
        extra_params=judge_cfg.get("extra_params"),
        num_retries=judge_cfg.get("num_retries", 3),
        timeout=judge_cfg.get("timeout", default_timeout),
        temperature=judge_cfg.get("temperature"),
        max_tokens=judge_cfg.get("max_tokens"),
        max_workers=execution.get("max_workers", 8),
    )

    scoring_op = ScoringOperator()

    return loader >> gen_op >> judge_op >> scoring_op


def build_eval_pipeline(config: dict) -> "QAEvalPipeline":
    """Construct a multi-model ``QAEvalPipeline`` from config.

    Uses all generators listed in config for multi-model sweeps.

    Parameters
    ----------
    config : dict
        Parsed config from :func:`load_eval_config`.

    Returns
    -------
    QAEvalPipeline
        A fully configured pipeline ready for ``.evaluate(qa_pairs)``
        or ``.process(df)``.
    """
    from nemo_retriever.models.llm.clients import LLMJudge, LiteLLMClient
    from nemo_retriever.tools.evaluation.orchestrator import QAEvalPipeline
    from nemo_retriever.tools.evaluation.retrievers import FileRetriever

    generators = config["generators"]
    if not generators:
        raise ValueError("Config must have at least one generator")

    execution = config.get("execution", {})
    retrieval = config.get("retrieval", {})
    judge_cfg = config["judge"]
    default_timeout = execution.get("timeout", 120.0)

    retrieval_type = retrieval.get("type", "file")
    if retrieval_type == "file":
        retriever = FileRetriever(file_path=retrieval["file_path"])
    elif retrieval_type == "lancedb":
        raise ValueError(
            "retrieval.type='lancedb' requires the caller to build the retriever "
            "via FileRetriever.from_lancedb() and pass it to run_eval_sweep(retriever=...). "
            "build_eval_pipeline() cannot construct it because it needs qa_pairs."
        )
    else:
        raise ValueError(f"Unsupported retrieval type: {retrieval_type!r}. " "Supported: 'file', 'lancedb'.")

    llm_clients: dict[str, LiteLLMClient] = {}
    for gen_cfg in generators:
        name = gen_cfg.get("name", gen_cfg["model"])
        llm_clients[name] = LiteLLMClient.from_kwargs(
            model=gen_cfg["model"],
            api_base=gen_cfg.get("api_base"),
            api_key=gen_cfg.get("api_key"),
            temperature=gen_cfg.get("temperature", 0.0),
            top_p=gen_cfg.get("top_p"),
            max_tokens=gen_cfg.get("max_tokens", 4096),
            extra_params=gen_cfg.get("extra_params"),
            num_retries=gen_cfg.get("num_retries", 3),
            timeout=gen_cfg.get("timeout", default_timeout),
            reasoning_enabled=gen_cfg.get("reasoning_enabled", True),
        )

    judge = LLMJudge.from_kwargs(
        model=judge_cfg["model"],
        api_base=judge_cfg.get("api_base"),
        api_key=judge_cfg.get("api_key"),
        extra_params=judge_cfg.get("extra_params"),
        num_retries=judge_cfg.get("num_retries", 3),
        timeout=judge_cfg.get("timeout", default_timeout),
        temperature=judge_cfg.get("temperature"),
        max_tokens=judge_cfg.get("max_tokens"),
    )

    return QAEvalPipeline(
        retriever=retriever,
        llm_clients=llm_clients,
        judge=judge,
        top_k=execution.get("top_k", 5),
        max_workers=execution.get("max_workers", 8),
        include_chunks_in_results=execution.get("include_chunks_in_results", True),
        chunk_char_limit=execution.get("chunk_char_limit", 500),
    )
