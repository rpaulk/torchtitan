# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from renderers import build_training_sample, Message, Renderer, ToolSpec

from torchtitan.components.data.dataset import (
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.components.data.sources import (
    HuggingFaceRandomAccessSource,
    HuggingFaceStreamingSource,
)
from torchtitan.components.data.types import DatasetBuildContext
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.renderer import RendererConfig
from torchtitan.components.tokenizer import HuggingFaceTokenizer


logger = logging.getLogger(__name__)


def _read_text(sample: dict[str, Any]) -> str:
    return sample["text"]


class TextProcessor(SampleProcessor):
    """Tokenizes plain text into next-token input and label pairs."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        text_fn: Callable[[dict[str, Any]], str] = _read_text

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._tokenizer = context.tokenizer
        self._text_fn = config.text_fn

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        del rng
        input_ids = np.asarray(
            self._tokenizer.encode(self._text_fn(sample), add_bos=True, add_eos=True),
            dtype=np.int64,
        )
        if len(input_ids) < 2:
            return None
        return TextSequence(
            input_ids=input_ids[:-1],
            labels=input_ids[1:],
        )


def _require_token_prefix(full_tokens: list[int], prompt_tokens: list[int]) -> None:
    """Raise if prompt_tokens is not an exact prefix of full_tokens.

    ChatProcessor locates the prompt/response boundary by re-rendering the
    prompt alone and requiring it to tokenize to a prefix of the full
    conversation. That holds only when rendering the prompt with
    ``add_generation_prompt=True`` produces a textual prefix of the full render
    and the tokenizer does not merge characters across that seam. Both are
    properties of the template and tokenizer together rather than of an
    individual sample.

    Raise instead of dropping the sample: a mismatch means the label boundary
    is unknown, and because the cause is systematic it would fire for most
    samples, so dropping would silently train on a fraction of the dataset.
    The overflow path drops because an oversized example really is per-sample.
    """
    if full_tokens[: len(prompt_tokens)] != prompt_tokens:
        raise ValueError(
            "Prompt tokens are not an exact prefix of the full conversation "
            "tokens, so the prompt/response boundary cannot be located. "
            "ChatProcessor requires that rendering the prompt with "
            "add_generation_prompt=True yields a textual prefix of the full "
            "render, and that the tokenizer does not merge characters across "
            "that seam. A template that rewrites earlier turns when later ones "
            "are present, or turn separators that only merge in context, break "
            "this assumption."
        )


class ChatProcessor(SampleProcessor):
    """Tokenizes chat samples and masks labels outside assistant responses."""

    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        messages_fn: Callable[[dict[str, Any]], list[Message]]
        tools_fn: Callable[[dict[str, Any]], list[ToolSpec] | None] | None = None
        renderer: RendererConfig | None = None
        """Model renderer; None uses the tokenizer's single-turn chat template."""

    def __init__(self, config: Config, *, context: DatasetBuildContext) -> None:
        self._tokenizer = context.tokenizer
        self._max_context_length = context.max_context_length
        self._messages_fn = config.messages_fn
        self._tools_fn = config.tools_fn
        self._logged_first_sample = False
        self._renderer = None
        if config.renderer is None:
            if config.tools_fn is not None:
                raise ValueError("tools_fn requires a renderer")
            if context.tokenizer.eos_id is None:
                raise ValueError(
                    "Tokenizer does not have an eos_id set. "
                    "ChatProcessor requires a tokenizer with a valid EOS token."
                )
            self._eos_id = context.tokenizer.eos_id
        else:
            if not isinstance(context.tokenizer, HuggingFaceTokenizer):
                raise ValueError("Chat renderers require a HuggingFaceTokenizer.")
            self._renderer = config.renderer.build(tokenizer=context.tokenizer)

    @staticmethod
    def _validate_messages(messages: list[Message]) -> None:
        """Validate that messages are a single-turn [user, assistant] pair."""
        if len(messages) != 2:
            raise ValueError(
                f"Expected single-turn [user, assistant], got {len(messages)} messages"
            )
        if messages[0]["role"] != "user":
            raise ValueError(
                f"First message must be 'user', got '{messages[0]['role']}'"
            )
        if messages[1]["role"] != "assistant":
            raise ValueError(
                f"Second message must be 'assistant', got '{messages[1]['role']}'"
            )

    def __call__(
        self, sample: dict[str, Any], rng: np.random.Generator
    ) -> TextSequence | None:
        """Tokenize a chat sample and mask prompt labels.

        Returns None if the sample exceeds `max_context_length`, avoiding
        training on truncated responses.

        Uses the renderer's loss mask when configured; otherwise uses prefix
        re-tokenization to find the single-turn prompt/response boundary.
        """
        del rng
        messages = self._messages_fn(sample)
        if self._renderer is not None:
            tools = self._tools_fn(sample) if self._tools_fn is not None else None
            return self._tokenize_with_renderer(
                messages,
                tools=tools,
                renderer=self._renderer,
            )

        self._validate_messages(messages)

        full_text = self._tokenizer.apply_chat_template(messages)
        # Strip extra newline and ensure the sequence ends with EOS without duplicates
        full_text = full_text.rstrip("\n")
        full_tokens = self._tokenizer.encode(full_text, add_bos=True, add_eos=False)
        if full_tokens[-1] != self._eos_id:
            full_tokens.append(self._eos_id)

        sequence = self._create_sequence(full_tokens, full_text=full_text)
        if sequence is None:
            return None

        # Find prompt/response boundary by tokenizing just the user message
        # with add_generation_prompt=True.
        prompt_text = self._tokenizer.apply_chat_template(
            messages[:1], add_generation_prompt=True
        )
        prompt_tokens = self._tokenizer.encode(prompt_text, add_bos=True, add_eos=False)
        _require_token_prefix(full_tokens, prompt_tokens)
        prompt_len = len(prompt_tokens)
        sequence.labels[: max(prompt_len - 1, 0)] = IGNORE_INDEX
        return sequence

    def _tokenize_with_renderer(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        renderer: Renderer,
    ) -> TextSequence | None:
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Chat samples must end with an assistant message.")
        segments = self._thinking_segments(messages, renderer)
        if segments is None:
            return self._render_document(messages, tools=tools, renderer=renderer)

        # One document per user turn, packed into one row. Positions restart at
        # 0 per document, which packing, attention masks, and Mamba's seq_idx
        # all treat as a document boundary.
        documents = []
        for prefix, trained in segments:
            document = self._render_document(
                prefix, tools=tools, renderer=renderer, trained=trained
            )
            if document is not None:
                documents.append(document)
        if not documents:
            return None
        return TextSequence(
            input_ids=np.concatenate([d.input_ids for d in documents]),
            labels=np.concatenate([d.labels for d in documents]),
            positions=np.concatenate(
                [np.arange(len(d.input_ids), dtype=np.int64) for d in documents]
            ),
        )

    @staticmethod
    def _thinking_segments(
        messages: list[Message], renderer: Renderer
    ) -> list[tuple[list[Message], set[int]]] | None:
        """Split at user turns when the template strips earlier reasoning.

        A template that keeps reasoning only for the current tool cycle renders
        earlier assistant turns without it, so one render would train those
        turns without reasoning. Each segment instead renders the history up to
        one user turn's responses and trains only those responses, which is
        exactly what the model sees at inference. Returns None when one render
        is already faithful.
        """
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if len(user_indices) < 2:
            return None
        if not any(
            m["role"] == "assistant" and (m.get("reasoning_content") or m.get("reasoning"))
            for m in messages[: user_indices[-1]]
        ):
            return None
        retention = getattr(renderer, "effective_thinking_retention", None)
        if retention is None:
            raise ValueError(
                f"{type(renderer).__name__} does not report "
                "effective_thinking_retention; cannot tell whether multi-turn "
                "reasoning history must be split."
            )
        if retention == "all":
            return None
        if retention != "tool_cycle":
            raise ValueError(f"Unsupported thinking retention {retention!r}.")

        # Shallow copies make message identity unique for role_to_mask.
        messages = [dict(m) for m in messages]
        bounds = [*user_indices[1:], len(messages)]
        segments = []
        for start, end in zip(user_indices, bounds):
            while end > start and messages[end - 1]["role"] != "assistant":
                end -= 1
            if end > start:
                segments.append(
                    (messages[:end], {id(m) for m in messages[start:end]})
                )
        return segments

    def _render_document(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None,
        renderer: Renderer,
        trained: set[int] | None = None,
    ) -> TextSequence | None:
        # TODO(data-sft-supervision): Support per-turn loss weighting.
        kwargs = {}
        if trained is not None:
            kwargs["role_to_mask"] = lambda m: id(m) in trained
        rendered = build_training_sample(
            renderer,
            messages,
            tools=tools,
            ensure_final_stop=True,
            **kwargs,
        )
        if rendered.multi_modal_data is not None:
            raise ValueError("ChatProcessor supports text-only samples.")

        sequence = self._create_sequence(rendered.token_ids)
        if sequence is None:
            return None

        # Shift the mask with the labels: label j predicts token j + 1.
        sequence.labels[~np.asarray(rendered.loss_mask[1:], dtype=bool)] = IGNORE_INDEX
        return sequence

    def _create_sequence(
        self, full_tokens: list[int], *, full_text: str | None = None
    ) -> TextSequence | None:
        if not self._logged_first_sample:
            if full_text is None:
                full_text = self._tokenizer.decode(
                    full_tokens, skip_special_tokens=False
                )
            logger.info(f"[ChatProcessor] First sample full:\n{full_text}")
            self._logged_first_sample = True

        # TODO(data-sft-overflow): Consider truncating oversized examples instead.
        # Causal loss remains valid for the retained response prefix.
        # Drop oversized examples rather than truncating.
        if len(full_tokens) - 1 > self._max_context_length:
            logger.debug(
                "Dropping sample: token count exceeds "
                f"max_context_length={self._max_context_length}"
            )
            return None

        tokens = np.asarray(full_tokens, dtype=np.int64)
        return TextSequence(
            input_ids=tokens[:-1],
            labels=tokens[1:].copy(),
        )


DATASETS: dict[str, SingleDatasetConfig] = {
    "c4": SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="allenai/c4",
            name="en",
            split="train",
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    ),
    "c4_test": SingleDatasetConfig(
        source=HuggingFaceRandomAccessSource.Config(
            path="json",
            split="train",
            load_dataset_kwargs={
                "data_files": "tests/assets/c4_test/data.json",
            },
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    ),
    "c4_validation": SingleDatasetConfig(
        source=HuggingFaceStreamingSource.Config(
            path="allenai/c4",
            name="en",
            split="validation",
        ),
        processor=TextProcessor.Config(),
        post_filters=(lambda sample: sample is not None,),
    ),
}
