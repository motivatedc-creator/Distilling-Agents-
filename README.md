# Distilling Agents

A deliberately small local coding-agent harness built around one principle:

> **The execution harness matters more than model size.**

V1 gives a local Qwen2.5-Coder worker a narrow job: produce a structured unified diff. Python owns repository access, patch validation, test execution, retry limits, and Git isolation.

## V1 flow

```text
issue description
      ↓
temporary git worktree
      ↓
retrieve small repo context
      ↓
Qwen2.5-Coder via vLLM
      ↓
JSON-schema constrained {"diff": "..."}
      ↓
git apply --check
      ↓
apply patch
      ↓
run deterministic tests
   ↙         ↘
 pass        fail
  ↓            ↓
return diff   reset worktree
               ↓
          retry (max 3)
               ↓
             BLOCKED
```

## What the worker can and cannot do

The model **does not receive arbitrary shell access**. It does not run `os.system`, install packages, commit, push, or touch your source checkout.

The harness owns these deterministic operations:

- list tracked files
- read repository files
- search tracked code
- validate a unified diff
- apply the diff
- run one preconfigured test command
- capture `git diff`
- reset a failed attempt

The CLI creates a detached temporary Git worktree, so the source checkout is not modified.

## Requirements

- Windows 11 + WSL2/Ubuntu is recommended for NVIDIA/vLLM setups
- Python 3.11+
- Git
- NVIDIA GPU capable of running the selected local model
- vLLM serving an OpenAI-compatible endpoint

## Install the harness

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Start the local worker

The initial target is `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`.

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct-AWQ \
  --host 127.0.0.1 \
  --port 8001 \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-seqs 1
```

The client uses vLLM's OpenAI-compatible `response_format: json_schema` structured output API, not the removed `guided_json` interface.

## Run a repair

Point the harness at a **Git repository with a deterministic test command and dependencies already available to the process**:

```bash
distill-agent /path/to/repo \
  --issue "add(2, 3) incorrectly returns -1; fix it without changing the public API" \
  --test-command "python -m pytest -q"
```

Successful output contains the final diff. Failed runs stop after the configured attempt budget and return `blocked` rather than looping forever.

## Central Pashto voice

Distilling Agents can optionally synthesize status speech through an isolated OmniVoice installation. Central Pashto (`pst`) is the default language.

```bash
distill-speak "سلام، څنګه یې؟" --play
```

The coding harness does not install or import OmniVoice directly. See `docs/voice.md` for WSL2/CUDA setup, voice cloning, playback, and GPU-contention notes. Voice is disabled for the coding benchmark.

## Current scope

Implemented now:

- LangGraph execution loop
- strict patch response schema
- safe patch validation
- temporary Git worktree isolation
- deterministic tests
- retry budget / blocked state
- unit and integration-style tests using a synthetic buggy repo
- optional isolated OmniVoice speech adapter with Central Pashto default
- standalone `distill-speak` command
- GitHub Actions CI

Not implemented yet:

- DeepSeek-R1 orchestrator
- Docker execution sandbox
- real GitHub issue/webhook ingestion
- automatic PR creation
- semantic repo indexing/RAG
- marketing engine
- telemetry dashboards
- automatic GPU model handoff between Qwen and OmniVoice

Those are intentionally postponed until the core repair loop proves reliable.

## Prototype success gate

Before expanding the architecture, build a benchmark of 10 synthetic bugs and require:

- at least 8/10 fixed without human intervention
- zero source-checkout modifications
- zero invalid patches reaching execution
- zero infinite loops
- every failed run terminates with a useful blocked result
