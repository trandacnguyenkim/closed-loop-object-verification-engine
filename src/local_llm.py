"""
local_llm.py — Shared local Qwen chat helper (no network API calls).

This module loads a Hugging Face causal LM once per process and exposes a
small helper to generate JSON responses from chat-style prompts.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

_DEFAULT_MODEL = "Qwen/Qwen3-14B-Instruct"


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object found in *text*."""
    # Qwen3 prepends <think>...</think> reasoning blocks before the JSON output.
    # Strip them so we don't accidentally match a { inside the thinking block.
    if "<think>" in text and "</think>" in text:
        text = text[text.rfind("</think>") + len("</think>"):]

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start token found in model output.")

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])

    raise ValueError("No complete JSON object found in model output.")


class LocalQwenChat:
    """Singleton-like local Qwen wrapper for chat + JSON generation."""

    _tokenizer = None
    _model = None
    _loaded_model_name = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("LOCAL_LLM_MODEL", _DEFAULT_MODEL)
        self._ensure_loaded()

    @classmethod
    def _dtype_for_device(cls) -> torch.dtype:
        if torch.cuda.is_available():
            # BF16 is preferred on modern accelerators when available.
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32

    def _ensure_loaded(self) -> None:
        if (
            LocalQwenChat._model is not None
            and LocalQwenChat._tokenizer is not None
            and LocalQwenChat._loaded_model_name == self.model_name
        ):
            return

        dtype = self._dtype_for_device()
        print(f"[LocalQwen] Loading local model: {self.model_name} (dtype={dtype})")

        LocalQwenChat._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        LocalQwenChat._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        LocalQwenChat._loaded_model_name = self.model_name
        print("[LocalQwen] Model ready.")

    @property
    def tokenizer(self):
        return LocalQwenChat._tokenizer

    @property
    def model(self):
        return LocalQwenChat._model

    def generate_json(
        self,
        system_prompt: str,
        user_messages: list[str],
        max_new_tokens: int = 700,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """
        Generate a JSON dict from a system prompt plus one or more user turns.
        Retries if parsing fails.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": "user", "content": msg} for msg in user_messages)

        for attempt in range(max_retries):
            try:
                chat_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        do_sample=temperature > 0,
                        temperature=temperature,
                        top_p=0.9,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
                raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                return _extract_first_json_object(raw)

            except Exception as exc:
                print(f"[LocalQwen] Attempt {attempt + 1} failed: {exc}")
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError("Unexpected JSON generation failure.")