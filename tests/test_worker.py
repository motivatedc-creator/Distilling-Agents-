from __future__ import annotations

from types import SimpleNamespace

from distilling_agents.worker import VLLMWorker


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"diff":"x"}'))],
            usage=SimpleNamespace(prompt_tokens=4321),
        )


class FakeClient:
    def __init__(self) -> None:
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_worker_caps_prompt_sections_and_reserves_output_tokens() -> None:
    worker = VLLMWorker(max_output_tokens=1_800)
    fake = FakeClient()
    worker.client = fake

    response = worker.generate_patch(
        issue="i" * 20_000,
        context="c" * 30_000,
        error_log="e" * 20_000,
        attempt=1,
    )

    assert response.diff == "x"
    assert fake.completions.kwargs is not None
    assert fake.completions.kwargs["max_tokens"] == 1_800
    user_message = fake.completions.kwargs["messages"][1]["content"]
    assert len(user_message) <= 18_500
    assert worker.last_prompt_tokens == 4321
