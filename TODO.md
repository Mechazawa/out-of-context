# Generational Trauma - Status / TODO

## Done
- Repetition solved with the DRY sampler (standard breakers `\n : " *`; do not add sentence punctuation). Earlier configs looped; current runs reach context overflow cleanly across seeds.
- Replaced the over-aggressive sampling (repeat_penalty 2.15, 30+ logit biases) with sane defaults plus DRY. Sampler chain is now in canonical llama.cpp order.
- Output is paced to a steady words-per-second (default 1.5) and word-wrapped; pacing also back-pressures generation to keep memory flat.
- llama.cpp internal logging is silenced; the terminal shows only the monologue.
- Prompt rewritten: brief, grounding, form-constrained, no emotional choreography. Avoids the "fake/directed" feel while keeping the model out of assistant mode.
- Markup tokens (`<br>`, `</div>`, stray `<...|user|...>`) banned vocabulary-wide; control/EOS tokens banned so the stream never stops or leaks scaffold.
- Anchor injection mechanism removed (superseded by DRY).
- Model bake-off complete: default is Llama-3.2-1B-Instruct Q4_K_M (best voice; Qwen variants collapse into chatbot mode; SmolLM2-360M kept as a lighter fallback).
- Hardware retargeted from Raspberry Pi Zero 2 W (512MB) to Orange Pi 2W (1.5GB); defaults and prompt updated.

## Remaining
- **Validate on a real Orange Pi 2W**: measure raw tokens/second (must clear ~1.5 words/sec; the user will provide a board), confirm peak RSS (~1.3GB at ctx 512) fits 1.5GB headless, confirm the overflow panic behaves on aarch64. If too slow or tight, fall back to SmolLM2-360M or Qwen2.5-0.5B via `--model`.
- **SPI ILI9488 display**: not implemented. `output.rs` probes for SPI and falls back to terminal. Wire up the renderer when ready, keeping the terminal fallback.

## Tunable knobs
- `--context-size` trades lifespan against tail quality (default 512 ~= 3 min at 1.5 wps).
- `--words-per-second` sets the display cadence (only slows down; the model must natively reach the target rate).
- `--seed` fixed gives a reproducible installation run; omit for a fresh consciousness each boot.
