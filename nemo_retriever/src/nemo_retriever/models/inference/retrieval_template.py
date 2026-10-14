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
    """Load the required ``chat_template.jinja`` retrieval asset as UTF-8 text.

    Parameters
    ----------
    model_id : str
        Local checkpoint directory containing ``config.json`` and
        ``chat_template.jinja``, or a Hugging Face Hub repository ID. Local
        paths use the checkpoint resolver's home-directory expansion.
    revision : str or None
        Hub revision passed to the downloader; ignored for local directories.
        Callers supply the checkpoint's resolved immutable revision when pinned.
    cache_dir : str or None
        Optional Hugging Face download cache directory. Local files are read
        directly. Successful loads are cached by the three function arguments.

    Returns
    -------
    str
        Nonempty template source, preserving its original whitespace. The
        source is compiled and rendered separately by render_retrieval_input.

    Raises
    ------
    ValueError
        If a local path is invalid, its config is missing, or the template
        contains only whitespace.
    OSError, UnicodeError
        If the template file cannot be read as UTF-8.
    huggingface_hub.errors.HfHubHTTPError
        If the repository, revision, or required asset cannot be downloaded.
    huggingface_hub.errors.LocalEntryNotFoundError
        If the required asset is unavailable both remotely and in the cache.
    """
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

    Parameters
    ----------
    template : str
        Jinja template source, normally loaded from ``chat_template.jinja``.
        It receives ``messages`` and ``add_generation_prompt=False``.
    prefix : str
        Checkpoint query or document prefix, passed as the system message.
        One exact leading occurrence is removed from text if already present.
    text : str
        Raw user content, converted to a string before rendering. Embedded
        token-like text never selects image modality.
    image : bool
        If True, represent user content as an image placeholder followed by a
        text block; otherwise pass a plain text user message. This function
        does not decode or attach image bytes.

    Returns
    -------
    str
        Text rendered by the checkpoint template, without an additional
        generation wrapper or a second prefix/template pass.

    Raises
    ------
    jinja2.TemplateError
        If the template cannot be compiled or raises an error during rendering.
    """
    content = str(text)
    if prefix and content.startswith(prefix):
        content = content[len(prefix) :]
    user_content = [{"type": "image"}, {"type": "text", "text": content}] if image else content
    messages = [{"role": "system", "content": prefix}, {"role": "user", "content": user_content}]
    return _compile(template).render(messages=messages, add_generation_prompt=False)
