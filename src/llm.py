"""
Local LLM backends.

Two implementations behind one interface:

  OllamaBackend       - easiest to run, good for prompt iteration on a laptop
  TransformersBackend - runs on a GPU cluster, gives batching and exact control
                        over quantisation, needed for the full annotation run

Everything runs locally. No post text leaves the machine, which is the condition
the ethics route depends on.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    latency_s: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None


class Backend(ABC):
    """Common interface so the annotation loop is backend-agnostic."""

    name: str

    @abstractmethod
    def generate(self, system: str, prompt: str, max_tokens: int = 256) -> Generation:
        ...

    def generate_batch(
        self, system: str, prompts: list[str], max_tokens: int = 256
    ) -> list[Generation]:
        """Default is sequential. Override where the backend can do better."""
        return [self.generate(system, p, max_tokens) for p in prompts]


class OllamaBackend(Backend):
    """
    Talks to a local Ollama server (default http://localhost:11434).

    Setup:
        curl -fsSL https://ollama.com/install.sh | sh
        ollama pull llama3.1:8b
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
        seed: int = 42,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.name = f"ollama:{model}"

    def generate(self, system: str, prompt: str, max_tokens: int = 256) -> Generation:
        import requests

        started = time.perf_counter()
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "system": system,
                "prompt": prompt,
                "stream": False,
                "options": {
                    # Temperature 0 with a fixed seed. Annotation should be
                    # reproducible; sampling variation here would show up later
                    # as label noise that is impossible to distinguish from
                    # genuine model uncertainty.
                    "temperature": self.temperature,
                    "seed": self.seed,
                    "num_predict": max_tokens,
                },
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        return Generation(
            text=payload.get("response", ""),
            latency_s=time.perf_counter() - started,
            prompt_tokens=payload.get("prompt_eval_count"),
            output_tokens=payload.get("eval_count"),
        )


class TransformersBackend(Backend):
    """
    Runs the model in-process via HuggingFace transformers.

    Use this for the full annotation run. Supports 4-bit quantisation so an 8B
    model fits comfortably on a single 16GB GPU, and batches properly, which
    matters when annotating tens of thousands of posts.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit: bool = True,
        device_map: str = "auto",
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.name = f"transformers:{model_id}"
        self._model = None
        self._tokenizer = None
        self._load_in_4bit = load_in_4bit
        self._device_map = device_map

    def _ensure_loaded(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # bfloat16 needs Ampere or newer. A Colab T4 is Turing, where bf16 is
        # unaccelerated and the 4-bit kernels are unreliable, so fall back to
        # float16 there.
        dtype = torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            dtype = torch.float16

        kwargs = {"device_map": self._device_map, "torch_dtype": dtype}
        if self._load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Decoder-only models need left padding for correct batched generation.
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    def _build_chat(self, system: str, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, system: str, prompt: str, max_tokens: int = 256) -> Generation:
        return self.generate_batch(system, [prompt], max_tokens)[0]

    def generate_batch(
        self, system: str, prompts: list[str], max_tokens: int = 256
    ) -> list[Generation]:
        import torch

        self._ensure_loaded()
        texts = [self._build_chat(system, p) for p in prompts]
        encoded = self._tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=4096
        ).to(self._model.device)

        started = time.perf_counter()
        with torch.no_grad():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started

        results = []
        input_len = encoded["input_ids"].shape[1]
        for i in range(len(prompts)):
            new_tokens = generated[i][input_len:]
            results.append(
                Generation(
                    text=self._tokenizer.decode(new_tokens, skip_special_tokens=True),
                    # Wall time split across the batch. Fine for throughput
                    # figures; do not quote it as single-post latency.
                    latency_s=elapsed / len(prompts),
                    prompt_tokens=int(encoded["attention_mask"][i].sum()),
                    output_tokens=int((new_tokens != self._tokenizer.pad_token_id).sum()),
                )
            )
        return results


def get_backend(kind: str = "ollama", **kwargs) -> Backend:
    if kind == "ollama":
        return OllamaBackend(**kwargs)
    if kind == "transformers":
        return TransformersBackend(**kwargs)
    raise ValueError(f"Unknown backend {kind!r}. Options: ollama, transformers")
