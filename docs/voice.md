# Central Pashto Voice with OmniVoice

Distilling Agents can synthesize speech through an isolated [OmniVoice](https://github.com/k2-fsa/OmniVoice) installation. The default language is Central Pashto, OmniVoice language ID `pst`.

Voice is intentionally outside the coding-agent environment. A TTS failure never changes a coding result, and the coding benchmark does not use voice.

## Environment boundary

Keep the two Python environments separate:

```text
Distilling-Agents-/.venv/          # coding harness
~/.local/omnivoice-venv/           # OmniVoice + Torch/audio stack
```

The integration baseline below matches the upstream OmniVoice 0.2.1 setup reviewed on 2026-08-08. Recheck upstream requirements before future upgrades.

```bash
python3 -m venv ~/.local/omnivoice-venv
source ~/.local/omnivoice-venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install omnivoice==0.2.1
```

Install Distilling Agents in its own environment separately. Do not add OmniVoice, Torch, Torchaudio, Transformers, or Accelerate to the main project dependency list.

## Standalone Central Pashto synthesis

Generate a WAV:

```bash
distill-speak "سلام، څنګه یې؟" \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer"
```

Generate and play it locally:

```bash
distill-speak "سلام، څنګه یې؟" \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer" \
  --play
```

`pst` is used unless `--language` overrides it. Generated files default to `~/.cache/distilling-agents/voice/`.

## Voice cloning

OmniVoice can use an explicitly supplied reference recording:

```bash
distill-speak "کار بشپړ شو" \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer" \
  --ref-audio /path/to/pashto-reference.wav \
  --ref-text "د ريفرنس غږ متن"
```

Only clone audio that you have permission to use. A short clean Pashto reference is preferable for Pashto output.

## Coding-agent status voice

Automatic status speech is opt-in and happens only after `run_agent()` has finalized the technical result:

```bash
distill-agent /path/to/repo \
  --issue "fix the bug" \
  --voice \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer"
```

The terminal JSON remains the source of truth. Automatic TTS receives only a short deterministic Pashto status phrase; it never receives diffs, logs, filenames, SHAs, or raw test output.

Voice failures are reported separately and never alter the coding exit code.

## RTX 4060 / 8 GB VRAM note

On a single 8 GB GPU, a running Qwen/vLLM server may occupy too much VRAM for OmniVoice. V1 deliberately does not kill, suspend, or restart vLLM.

If OmniVoice hits CUDA out-of-memory, the coding run remains valid and voice is reported as unavailable. For a manual voice test, stop vLLM first and run `distill-speak` again.

Automatic GPU handoff is intentionally postponed until the coding benchmark has proven the core repair harness.

## Playback under WSL2

The playback adapter checks for local players in this order:

1. `ffplay`
2. `paplay`
3. `aplay`

One concrete setup path is FFmpeg:

```bash
sudo apt update
sudo apt install -y ffmpeg
```

Manual player check:

```bash
ffplay -nodisp -autoexit /tmp/distilling-agents-pst.wav
```

Playback failure does not invalidate a successfully generated WAV.

## Manual Central Pashto acceptance test

Real audio cannot be verified by GitHub Actions. Run this on the target NVIDIA machine, with vLLM stopped if VRAM requires it:

```bash
distill-speak "سلام، څنګه یې؟ کار بشپړ شو." \
  --omnivoice-command "$HOME/.local/omnivoice-venv/bin/omnivoice-infer" \
  --output /tmp/distilling-agents-pst.wav \
  --play
```

Acceptance requires all of the following:

1. the command exits `0`;
2. `/tmp/distilling-agents-pst.wav` exists and is non-empty;
3. audible speech is produced through a supported player;
4. the listener recognizes the speech as Central Pashto;
5. no coding-agent/vLLM process is modified or terminated by the voice layer.

If the WAV is valid but pronunciation or voice quality is poor, record that as a model-quality finding rather than a harness failure.

## CI boundary

CI verifies the process adapter, language defaults, playback failure handling, coding-result isolation, and CLI behavior using fakes. CI must not install OmniVoice, download model weights, require CUDA, or require a sound device.
