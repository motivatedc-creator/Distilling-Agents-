from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from .models import PatchResponse


class PatchWorker(Protocol):
    def generate_patch(self, *, issue: str, context: str, error_log: str, attempt: int) -> PatchResponse: ...


class VLLMWorker:
    """Qwen worker accessed through vLLM's OpenAI-compatible API."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8001/v1",
        api_key: str = "-",
        model: str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        timeout: float = 120.0,
        max_output_tokens: int = 1_800,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.last_prompt_tokens: int | None = None

    def generate_patch(self, *, issue: str, context: str, error_log: str, attempt: int) -> PatchResponse:
        system = (
            "You are a constrained code-repair worker. Produce the smallest correct repository patch. "
            "Do not change public APIs unless the issue explicitly requires it. "
            "Your response is machine-consumed and must match the supplied JSON schema."
        )
        # Character caps are deliberately conservative for the 8,192-token serving window.
        # Exact prompt token usage is recorded from vLLM when the server returns usage data.
        issue_context = issue[:3_000]
        repository_context = context[:10_500]
        retry_context = (error_log.strip() or "No previous attempt has been executed.")[-3_000:]
        user = f"""ISSUE\n{issue_context}\n\nREPOSITORY CONTEXT\n{repository_context}\n\nPREVIOUS ATTEMPT RESULT\n{retry_context}\n\nATTEMPT\n{attempt}\n\nReturn one unified git diff in the diff field. Paths must be repository-relative and use a/ and b/ prefixes."""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=self.max_output_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "patch_response",
                    "strict": True,
                    "schema": PatchResponse.model_json_schema(),
                },
            },
        )
        usage = getattr(completion, "usage", None)
        self.last_prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("worker returned an empty response")
        return PatchResponse.model_validate_json(content)
