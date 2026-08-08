# OmniVoice Central Pashto Voice Integration — Design

Date: 2026-08-08
Status: Approved design, pending implementation plan
Target repo: `motivatedc-creator/Distilling-Agents-`
External dependency: `k2-fsa/OmniVoice`

## Goal

Add a reusable local speech layer to Distilling Agents using OmniVoice, with Central Pashto as the default agent voice.

The feature has two user-facing modes:

1. Automatic agent status speech after a coding run finishes.
2. A standalone `distill-speak` command that synthesizes arbitrary text.

The default language is Central Pashto, OmniVoice language ID `pst`.

## Non-goals for V1

V1 will not:

- run OmniVoice inside the coding-agent Python environment;
- keep Qwen/vLLM and OmniVoice resident on the GPU simultaneously;
- automatically stop and restart vLLM to reclaim VRAM;
- add a long-lived TTS server or daemon;
- make voice output part of benchmark scoring;
- require voice synthesis for a coding task to succeed;
- add automatic translation from English to Pashto with another model;
- add voice cloning as the default mode before a reference voice is explicitly configured.

## External capability assumptions

OmniVoice supports Central Pashto with language ID `pst` and exposes both a Python API and `omnivoice-infer` CLI with explicit language selection.

OmniVoice supports auto voice, voice design, and voice cloning. V1 will use auto voice by default because OmniVoice documents voice design as primarily trained on Chinese and English, making it a weaker default for Central Pashto.

OmniVoice recommends an isolated/fresh environment and depends on its own PyTorch, Torchaudio, Transformers, Accelerate, and audio stack. Distilling Agents must therefore treat it as an external executable integration rather than importing it into the core package environment.

OmniVoice is Apache-2.0 licensed.

## Architecture

```text
coding issue
    |
    v
Distilling Agents / Qwen repair loop
    |
    +---- full technical result ---> terminal / JSON
    |
    v
Voice summary formatter
    |
    v
OmniVoice adapter
    |
    v
isolated OmniVoice environment / process
    |
    v
WAV output + optional local playback
```

The voice layer is downstream from the coding result. It is not part of the LangGraph repair decision loop.

## Components

### 1. `VoiceConfig`

A small configuration object owned by Distilling Agents.

Initial fields:

- `enabled: bool`
- `language: str = "pst"`
- `model: str = "k2-fsa/OmniVoice"`
- `omnivoice_command: str = "omnivoice-infer"`
- `output_dir: Path`
- `speed: float = 1.0`
- `num_step: int = 32`
- optional `ref_audio: Path | None`
- optional `ref_text: str | None`

Configuration may be provided by CLI arguments and environment variables. No persistent global configuration format is required for V1.

### 2. `OmniVoiceAdapter`

A narrow process adapter responsible only for turning text into a WAV file.

It will:

- construct an argument-vector invocation of `omnivoice-infer`;
- always pass explicit `--language pst` by default;
- pass the configured model, output path, speed, and generation step count;
- optionally pass `--ref_audio` and `--ref_text` when voice cloning is configured;
- execute without `shell=True`;
- enforce a synthesis timeout;
- capture bounded stdout/stderr;
- verify that the expected WAV file exists and is non-empty;
- return a structured success/failure result instead of raising into the coding workflow.

It will not manage CUDA processes, install packages, download models itself, or terminate vLLM.

### 3. Agent status speech

After `run_agent()` returns, the CLI will keep printing the existing technical JSON result exactly as the source of truth.

If voice is enabled, a short human-readable Central Pashto status phrase will then be selected from deterministic templates.

Examples of intended meaning:

- passed: the work is finished and tests passed;
- blocked: the task stopped after the available attempts;
- voice unavailable: coding result remains valid and a local warning is printed.

The automatic status layer must not send large diffs, logs, stack traces, filenames, SHAs, or raw test output to TTS.

V1 will use a small curated set of Pashto status strings instead of adding a translation model. This keeps behavior deterministic and prevents translation quality from contaminating coding reliability.

### 4. `distill-speak`

Add a second package script:

```bash
distill-speak "ستا سو حال دی"
```

Default behavior:

- language: `pst`;
- model: `k2-fsa/OmniVoice`;
- auto voice;
- output: generated WAV under a local output directory;
- print the output path and synthesis result.

Useful overrides:

```bash
distill-speak "..." --language pst --output out.wav
```

Voice cloning support:

```bash
distill-speak "..." \
  --ref-audio pakhto-reference.wav \
  --ref-text "..."
```

The same adapter used by automatic agent speech must back this command. There must not be two independent OmniVoice implementations.

### 5. Optional playback

Synthesis and playback are separate concerns.

V1 may support a best-effort `--play` option implemented through a small platform adapter, but WAV generation is the required capability. A missing audio player must never cause synthesis to be reported as failed.

## GPU and process policy

The target machine has one NVIDIA GPU and the existing Qwen/vLLM worker is expected to consume most available VRAM.

Therefore V1 will not promise simultaneous Qwen and OmniVoice residency.

The adapter will behave as follows:

- if OmniVoice can synthesize successfully, produce speech;
- if CUDA allocation fails because vLLM still owns VRAM, report voice as unavailable;
- do not kill, suspend, or restart the coding model automatically.

A future GPU supervisor may coordinate model handoff only after the 10-bug coding benchmark is complete and only if voice usage justifies the additional complexity.

## Failure isolation

Voice is strictly non-critical.

For every coding run:

```text
coding result = passed or blocked
voice result  = spoken / skipped / unavailable
```

The coding result must never change because TTS failed.

Expected voice failures include:

- `omnivoice-infer` not installed;
- model download unavailable;
- CUDA out of memory;
- synthesis timeout;
- invalid reference audio;
- process exits non-zero;
- WAV missing or empty;
- playback unavailable.

All such failures are converted into bounded diagnostic messages and the process exits according to the coding result, not the voice result.

## Benchmark policy

The 10-bug Qwen benchmark will run with automatic voice disabled.

This keeps benchmark metrics restricted to:

- retrieval;
- patch generation;
- patch validation;
- worktree safety;
- deterministic tests;
- retry/termination behavior.

TTS latency, model downloads, CUDA contention, and audio failures must not influence the benchmark pass rate.

## Environment strategy

Do not add OmniVoice and its heavy dependencies to the main `distilling-agents` dependency list.

Recommended local layout:

```text
Distilling-Agents-/.venv/          # coding harness
~/.local/omnivoice-venv/           # isolated TTS environment
```

The user installs OmniVoice in the dedicated environment following OmniVoice's documented PyTorch/CUDA requirements. Distilling Agents is configured with the path to that environment's `omnivoice-infer` executable when it is not already on `PATH`.

This boundary allows OmniVoice versions, Torch builds, or CUDA dependencies to change without destabilizing the coding harness.

## Security and safety boundaries

The voice adapter follows the existing harness philosophy:

- no `shell=True`;
- model text is data, not executable command text;
- executable path is configured, not generated by the LLM;
- output paths are controlled by the harness;
- bounded process timeout;
- bounded logs;
- no arbitrary command execution exposed to the coding worker;
- reference audio is read-only input;
- voice cloning is only used with audio the user has permission to use.

## Testing strategy

Core CI must not require a GPU, OmniVoice, or model download.

Unit tests will use a fake executable/process boundary to verify:

1. Central Pashto `pst` is passed by default.
2. standalone text produces the expected argument vector.
3. voice-clone arguments are included only when configured.
4. non-zero OmniVoice exit becomes a structured unavailable result.
5. timeout becomes a structured unavailable result.
6. missing/empty WAV is rejected.
7. automatic speech failure does not change an agent `passed` result.
8. automatic speech failure does not change an agent `blocked` result.
9. benchmark mode disables automatic speech.
10. CLI remains backward compatible when voice is disabled.

A local manual acceptance test on the target NVIDIA machine will be required before claiming real Central Pashto synthesis works end-to-end.

## Acceptance criteria

V1 is complete when all of the following are true:

- `distill-speak` exists;
- Central Pashto (`pst`) is its default language;
- it can call an isolated OmniVoice installation and produce a non-empty WAV;
- the main coding CLI can optionally speak a short Central Pashto result summary;
- voice cloning inputs can be supplied explicitly;
- TTS failures never crash or change the coding result;
- test coverage works in GitHub Actions without GPU/model downloads;
- the 10-bug coding benchmark remains voice-disabled;
- README documents setup, isolation, expected VRAM contention, and usage;
- no automatic vLLM shutdown/restart logic is included in V1.

## Future extensions

Only after the core coding benchmark and V1 voice integration are proven:

- persistent cloned Pakhtoon voice profile;
- a user-approved Pashto reference recording encoded once and reused across sessions;
- controlled GPU model handoff supervisor;
- richer Pashto event vocabulary;
- configurable notification policies;
- local speech input / ASR if desired;
- long-running TTS service if startup latency becomes a measured problem.
