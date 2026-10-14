# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint retrieval templates shared by admission and local inference.

Only the supported Mistral3 retrieval profile uses this path. Legacy text
embedders retain their prefix-only contract. No generation wrappers are added.
"""

from functools import lru_cache
from pathlib import Path

from nemo_retriever.models.embed_model_spec import _local_config_path


@lru_cache(maxsize=16)
def load_retrieval_template(model_id: str, revision: str | None, cache_dir: str | None = None) -> str:
    local_config = _local_config_path(model_id)
    if local_config is not None:
        path = local_config.parent / "chat_template.jinja"
    else:
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model_id, "chat_template.jinja", revision=revision, cache_dir=cache_dir))
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"Embedding model {model_id!r} requires a nonempty retrieval chat template.")
    return template


@lru_cache(maxsize=16)
def _compile(template: str):
    from jinja2 import TemplateError
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise TemplateError(message)

    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    environment.globals["raise_exception"] = raise_exception
    return environment.from_string(template)


def render_retrieval_input(template: str, prefix: str, text: str, *, image: bool = False) -> str:
    """Apply the checkpoint template once to a prefix and raw content.

    Preserve the historical acceptance of already-prefixed input, but never
    infer modality from tokens embedded in user text.
    """
    content = str(text)
    if prefix and content.startswith(prefix):
        content = content[len(prefix) :]
    user_content = [{"type": "image"}, {"type": "text", "text": content}] if image else content
    messages = [{"role": "system", "content": prefix}, {"role": "user", "content": user_content}]
    return _compile(template).render(messages=messages, add_generation_prompt=False)
