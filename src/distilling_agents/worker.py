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
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def generate_patch(self, *, issue: str, context: str, error_log: str, attempt: int) -> PatchResponse:
        system = (
            "You are a constrained code-repair worker. Produce the smallest correct repository patch. "
            "Do not change public APIs unless the issue explicitly requires it. "
            "Your response is machine-consumed and must match the supplied JSON schema."
        )
        retry_context = error_log.strip() or "No previous attempt has been executed."
        user = f"""ISSUE\n{issue}\n\nREPOSITORY CONTEXT\n{context}\n\nPREVIOUS ATTEMPT RESULT\n{retry_context}\n\nATTEMPT\n{attempt}\n\nReturn one unified git diff in the diff field. Paths must be repository-relative and use a/ and b/ prefixes."""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "patch_response",
                    "strict": True,
                    "schema": PatchResponse.model_json_schema(),
                },
            },
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("worker returned an empty response")
        return PatchResponse.model_validate_json(content)
