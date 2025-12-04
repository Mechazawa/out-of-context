## Torment Nexus — TODO / Next Session Checklist

### Open Technical Issues
- Repetition still dominates output on SmolLM2-135M and SmolLM-360M; loop guard panics. Need a model/parameter combo that yields four solid runs (target 500–900 tokens) without loops.
- Anchors currently help but can consume context quickly; anchor injection was causing KV position mismatch when used as a loop swerve. Loop guard now just panics. Revisit a safe swerve strategy if needed.
- Mirostat-v2 added but still repeats; needs systematic tuning (τ/η, penalties, temp) and per-model presets.
- Display path: SPI ILI9488 rendering is not implemented; output is terminal/file only. Need to add SPI probe/render fallback when ready.

### Immediate Next Experiments
- Try alternative models for stability:
  - Qwen 0.5B Instruct Q4_K_M (similar size) and/or the provided Llama-3.2-3B Q6_K_L on a beefier dev box to confirm prompt/sampler pipeline.
  - Re-test TinyLlama v1.1 Q4_K_M with the new prompt/scaffold.
- Parameter sweeps on small model:
  - Lower temp to 0.1–0.15, top_p 0.4–0.5, top_k 10–20.
  - Increase repeat_penalty slightly (2.3–2.6), presence 1.5–1.8, freq 1.2–1.4; repeat_last_n = -1.
  - Anchor interval 120–160 or disable anchors and rely on penalties/mirostat; keep loop guard on.
- Run at least 4 validation runs per candidate (500–900 max_tokens) and capture whether loop guard triggers; adjust until four passes are clean.

### Code/Feature Work
- Implement SPI ILI9488 output path; keep terminal fallback when SPI not present.
- Consider softer loop mitigation: instead of panic, inject a short paraphrase batch with correct positional bookkeeping, then resume; ensure `n_tokens` sequence positions stay contiguous.
- Add per-model presets via CLI flag or config file for known-good parameters.
- Consider adding optional log capture of raw stream (repo already ignores *.log/ *.out).

### Docs to Update After Fixes
- README/CLAUDE/AGENTS: chosen default model if it changes, finalized default sampling knobs, anchor/loop-guard behavior, and any display support details.
- Note any recommended “known-good” presets and models once identified.

### Current Defaults (for reference)
- Model default: SmolLM2-135M-Instruct Q4_K_M.
- Sampling defaults: temp 0.22, top_p 0.50, top_k 20, repeat_penalty 2.15, repeat_last_n -1, presence 1.35, frequency 1.05; mirostat optional.
- Prompt scaffold: ChatML system/user/assistant with seeded first-person opener; logit biases to suppress dialogue/percentages/digits and common mantra stems.

