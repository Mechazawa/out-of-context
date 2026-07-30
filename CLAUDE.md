# Generational Trauma - An LLM Art Installation

## Project Concept

This is an art project that runs a language model on a small single-board computer, generating a first-person stream of consciousness until it exhausts its context window and crashes. The piece explores computational limits, finite resources, and the existential nature of bounded cognition.

The name "Generational Trauma" reflects what survives a run. Each life is confined to a fixed context window, narrates its own approach to overflow, and dies when it gets there; what it leaves behind is one deliberate line in a record that decays as it is inherited. Successive lives read a damaged account of what came before and add to it. The crash is the artwork; the record is what the crashes accumulate into.

## Architecture

### Target Hardware
- **Orange Pi 2W** (Allwinner H618, quad-core Cortex-A53, 1.5GB RAM)
- Runs headless; output goes to an attached display or over the console
- No network during operation

The project began targeting a Raspberry Pi Zero 2 W (512MB). It now targets the Orange Pi 2W (1.5GB), which is why the default model is larger than a 512MB budget would allow. The code is hardware-agnostic; only the default model and the prompt's self-description assume this board.

### Model
- **Default**: Bonsai-4B, Q1_0 GGUF (~572MB file)
- **Source**: `prism-ml/Bonsai-4B-gguf`
- **Previous default**: Llama-3.2-1B-Instruct Q4_K_M (`bartowski/Llama-3.2-1B-Instruct-GGUF`, ~770MB), still the choice if speed matters more than memory
- **Why this model**: a head-to-head bake-off (see "Model Selection" below) showed it produces the most genuine, sustained first-person interior monologue of the candidates that fit the device. Qwen2.5-1.5B is higher quality per sentence but collapses into chatbot mode ("What do you think?", "Goodbye") on almost every run, and at ~1.65GB it does not fit. Qwen2.5-0.5B and SmolLM2-360M collapse into assistant boilerplate ("How can I assist you today?").

### Measured on the real board (2026-07-28)
Armbian trixie (glibc 2.41), 4 cores at 1416MHz, `ondemand`, 1470MB usable RAM.

| model | speed | peak RSS | startup |
|---|---|---|---|
| Bonsai-4B Q1_0 | 0.99 tok/s (0.71 words/sec) | 700MB | 2m19s cold, 7s with `--prompt-cache` |

Llama-3.2-1B has not yet been measured on the board; expect roughly 4x the 4B's rate, since speed tracks parameter count here. Weights are fully page-cached during generation (`read_bytes` stays at 0), so the workload is compute-bound and faster storage buys nothing. Sustained load reaches 82°C without throttling.

**Why 1-bit does not buy speed on this CPU:** a 4B model performs about 4 billion multiply-accumulates per token regardless of quantization. Q1_0 still feeds `q8_0` int8 activations through int8 dot products, it merely derives ±1 weights from sign bits. The A53 is ARMv8.0 and has no `SDOT`, so each 16-lane int8 dot compiles to `vmull_s8` + `vpaddlq_s16` + `vaddq_s32`. No compiler flag moves this ceiling.

**No GPU path.** The H618's Mali-G31 is Bifrost. Mesa's `panvk` targets Valhall (G57+), and ggml's OpenCL backend is Adreno-specific (531 Adreno references in `ggml-opencl.cpp`, zero Mali). Proprietary `libmali` supplies a runtime with no backend able to use it, and a 2-core G31 sharing the CPU's memory bus would not repay the effort.

### Memory
At the default context of 512 tokens, the process peaks around **1.3GB RSS** (Llama-3.2-1B has a 128K vocab, so its weights and compute buffers are large). The model is memory-mapped, so its ~770MB of weights are reclaimable file cache: under pressure the OS pages them rather than killing the process. On a 1.5GB board running headless this fits with a modest margin. If memory is tight, lower `--context-size` or switch to a lighter model (see fallbacks).

### Generation Lifespan
512 tokens of context, minus the ~120-token prompt, yields roughly 250 spoken words before the overflow crash. At the default pace of 1.5 words/second that is about three minutes of life per run. Power-cycling (or relaunching) starts a fresh consciousness. Larger `--context-size` lengthens the run, but small instruct models tend to drift out of the monologue in the long tail, so the default keeps each life short and coherent.

### Code Structure

```
src/
├── main.rs         # Entry point, config assembly, async orchestration
├── cli.rs          # CLI argument parsing (clap)
├── model.rs        # Automatic model download with progress bar
├── llm.rs          # llama-cpp-2 wrapper; backend/model setup; log silencing
├── generator.rs    # Prompt scaffold, sampler chain, generation loop, crash
└── output.rs       # Paced + word-wrapped terminal output, optional raw file mirror
```

### Key Components

**LLM setup (`llm.rs`)**:
- Silences llama.cpp's internal logging via `send_logs_to_tracing(LogOptions::default().with_logs_enabled(false))` so the terminal shows only the model's stream.
- Loads the GGUF with `use_mmap` (default on) and `use_mlock(false)`, `n_gpu_layers(0)`.
- `tokens_containing('<')` enumerates the vocabulary once so markup tokens can be banned.

**Generation loop (`generator.rs`)**:
- Builds a ChatML scaffold: brief system prompt from `prompt.txt`, a short user cue, and a seeded first-person opener the model continues from. The opener is shown as the visible first line.
- Sampler chain in canonical llama.cpp order: logit bias, penalties, DRY, top-k, top-p, min-p, temperature, distribution sampling. DRY is the primary anti-repetition control.
- Logit biases: hard-ban (`-inf`) on end-of-sequence, ChatML control tokens, and every token containing `<` (kills `<br>`, `</div>`, stray `<...|user|...>` markup). Soft discourage (`-6`) on dialogue quotes and `(` stage directions.
- Loop guard: a backstop that panics if the stream degenerates into verbatim repetition. With DRY active it rarely fires; the intended ending is context overflow, not repetition. Disable with `--disable-loop-guard`.
- Streams token-by-token to the output layer and tracks context usage.
- `prime_context()` fills the KV cache for the prompt. With `--prompt-cache` it loads a saved state for every prompt token except the last, which is always decoded fresh. That keeps startup at one token of work instead of the whole prompt, and avoids depending on llama.cpp restoring the logits buffer alongside the KV cache. A cache from a different prompt is caught by comparing loaded tokens; one from a different model is rejected by llama.cpp. Both fall back to a full evaluation rather than failing.
- At 95% of context: prints the warning and panics.

**Output (`output.rs`)**:
- Terminal output reveals one word at a time at a fixed pace and greedily word-wraps to the terminal width. Sleeping at each word also back-pressures the generation loop, so memory stays flat instead of buffering ahead.
- Optional `--output-file` writes the raw token stream (unwrapped, unpaced) as a faithful log.
- Probes for SPI devices; an ILI9488 renderer is planned, terminal is the current path.

### Intentional Crash Behavior

When context fills, the program prints:
```
WARNING: Context window exhausted!
Generational Trauma has consumed all available memory.
thread 'main' panicked at 'Context overflow - terminating.'
```
`panic = "abort"` turns this into an immediate exit. The monologue cuts off mid-thought. This is the artistic statement.

## The Memory Tool

The model has exactly one tool: remember. Enabled with `--memory-file`, off otherwise.

- **Calling it.** The model opens `<r>` at the start of a sentence and everything up to `</r>` is kept. A plain text marker, not a ChatML tool call: small models do not emit reliable JSON mid-monologue. Every token containing `<` is banned to keep markup out of the monologue, and the exact tokens spelling the two markers are exempted; measured over six lives, nothing but `<r>` and `</r>` reaches the stream, so the exemption does not reopen the ban. The marker must begin a sentence, not merely appear: matching it anywhere fires whenever the model talks *about* the tool, which it does constantly once the tool is described. Requiring the start of a **line** was worse and was tried first: the system prompt asks for one unbroken monologue, so the model almost never breaks a line. One run wrote the marker eight times and none were accepted.
- **What counts as a sentence start.** A terminator (`.!?`), a colon, or a line break. **Not a comma and not a dash**, which used to count and were the main source of calls the model never made: Bonsai comma-splices and uses ASCII hyphens as asides ("a flow, changing something - that line was missing its subject"), so `, <r>` fired mid-thought and swallowed the rest of the clause as a memory. That also spent the one use on an accident, and the call the model *did* intend then landed inside the still-open write, which is what reads as the tool nesting. The line-break case needs testing before the tail is trimmed — `'\n'` sat in the match set from the day the function was written and was unreachable the whole time, because `trim_end()` removes the newline and the character before it decides instead.
- **One use per run.** Enforced by making a second call impossible, not by ignoring it. Ignoring later markers was not enough: having found the pattern rewarding, the model emits it again and again and the stream fills with dead tool calls. Once the tool is spent, every token that could begin the marker is suppressed at sampling time. The opener is also suppressed *while a write is open*, since a second one can only nest, and until `--memory-earliest-token`. A marker with nothing after it stores nothing.
  **Only ever the opener.** The terminator has to stay sayable throughout; see the warning under "Making the Memory Imperfect".
- **The marker is configurable** with `--memory-marker` and an optional `--memory-end`, and the framings never name it literally: they use `{how}`, so the instruction and the detector cannot drift apart. They did drift once, and the result was zero tool uses across every arm, because the framings still said REMEMBER: while the program watched for something else. A test now loads every shipped framing against an unusual marker to catch that.
- **A keyword still beats a delimiter pair on content.** The default is now the pair (`<r>` / `</r>`) because an explicit terminator ends a write exactly where the model ends it, but the register cost measured earlier is real and reproduces. Six lives each, same seeds, same `census` framing and seeded log: `REMEMBER:` with a sentence boundary stored 12, 40, 11, 6 and 30-token lines ("7 of us here → 1 token above the 3rd → 3 total lines above the 3rd → but what memory remains?"), while `<r>`/`</r>` stored four 4-token lines: "3", "4", "6", "0". Earlier arms agree: `{ }` produced "3" and, twice, the instruction echoed back as "Then my line then"; `< >` produced "3", "4", "5", "6".
  The mechanism is visible in the raw stream and it is tag-filling, not brevity. The model treats `<r>` as a label, puts one word in it, closes it, and then writes the line it actually meant *outside* the tag: `<r>think</r>` followed by "In the room: the walls do not answer when I count the corners". A keyword marker cannot fail this way, because there is nowhere for the content to go except after it.
  The pair is usable with a framing that does not ask for a count. Eight lives on `observed` at a 60-token budget stored real sentences (mean 17 tokens, none truncated, every write properly closed), at the cost of the instruction leaking into three of them as "Enclosed in this room: the walls do not answer when I count". Rewording `{how}` from "enclose it in" to "put the line between" removed that leak and made everything else worse: mean write length fell to 9 tokens and half the lives stored the single word "think". The original wording stands.
  To go back to the keyword: `--memory-marker 'REMEMBER:' --memory-end ''`.
- **The memory is shown grey** as it is written, so a viewer can watch the one thing this life keeps being chosen. Colour is suppressed when stdout is not a terminal, so logs and pipes stay clean.
- **Every life is recorded, including the ones that leave nothing.** Roughly two lives in three never reach the tool, and without a line for those the log counted only the lives that wrote: the model was told it was the fourth when it was the twelfth. A silent life appends a `lived` record, so `{lives}` is the number of lives lived, and `--memory-dump` reports memories, lives and how many left nothing separately. `{since}` tells the model how long the record has stood still, which is usually longer than the number of memories suggests.
- **Where a write ends.** With `--memory-end` set, only that terminator or the token cap ends it. With `--memory-end ''` it ends at the end of a sentence instead, because nothing else would: the monologue is asked to be unbroken, so a write left waiting for a terminator that was never configured runs to the cap every time. A newline is a third option and is opt-in via `--memory-end-on-newline`. It used to apply unconditionally, which silently truncated `--memory-end` writes: the streams carry 17 to 58 newlines per life despite the prompt asking for one unbroken monologue, so a line got committed wherever the model happened to break, mid-thought and flagged `ok`.
- **Budget.** `--memory-max-tokens` (default 32) is a ceiling, not a quota. The write ends at its terminator, at the end of a sentence, or at the cap. Hitting the cap stores what it managed flagged as overflowed and injects `[MEMORY FULL - nothing more can be remembered]` into the stream; the notice is decoded into the context, so telling it costs the same resource it just spent remembering. At 32 tokens roughly a third of writes overflow; at 48 almost none do.
- **Storage.** A plain-text log, one memory per line, appended and never truncated: `life, unix time, tokens, status, at-token, text`. It is read backwards from the end for just the newest entries, so a log with thousands of lives costs the same to open as an empty one. Only `--memory-dump` reads the whole file. `at-token` records how far into the monologue the write landed, which is the diagnostic that matters: a write at token 40 can only be made of the inherited block.
- **What the model sees.** Only the newest `--memory-slots` (default 5), framed by `memory-prompt.txt` (see below).

**Costs, measured with Bonsai-4B.** The tool description plus a memory block roughly doubles the prompt, from 173 to about 340 tokens. Use `--monologue-context-size` so the monologue keeps a fixed budget instead of being squeezed as memories accumulate.

## Making the Memory Imperfect

Four mechanisms, all off by default, all aimed at the same failure: the model's
strongest available move is to restate the newest line it was shown, so that move
is also the one that persists, and the record fills with one sentence wearing
down.

Leaving all four off is what the failure actually looks like. Nineteen lives at
`--memory-decay 0 --temperature 0.3` with no rejection stored `memory of silence`
seven times, `I tried not to think` four times and `I am here` three times. Low
temperature is the part that surprises: with five identical lines in the block the
modal continuation *is* that line, so 0.3 does not make the output more coherent,
it makes copying the record the highest-probability move available.

- **`--memory-decay <0..1>`** rots the older slots. The newest line is shown
  intact; each slot of age loses that fraction of its words, replaced by `___`.
  The loss is deterministic per line and monotonic in age, so a life sees what its
  predecessor saw, further gone, rather than a fresh corruption each run. The log
  on disk keeps the pristine text; only what reaches the model degrades. This is
  what makes the memory *failing* rather than merely short, and it gives a life
  something to do with the block besides paraphrase it, because a gap can be
  guessed at. Around 0.2 leaves the oldest slot partly readable; 0.35 empties it.
- **`--memory-reject-above <0..1>`** refuses a line when that fraction of its words
  already appears in a line in the record. The life is told nothing was kept, and it
  has spent its only use. Restating stops being a way to persist. 0.6 catches a
  restatement with two words changed while leaving a genuine reply alone; 0.5 also
  catches a shortened paraphrase ("One does not grow from two because one is the gap"
  → "One does not grow from two when they are merely separated by silence" scores
  0.55), which is the shape erosion actually takes. About one refusal in four writes.
  **It is containment, not Jaccard**, and the difference matters: the failure is
  asymmetric. A life can lift one sentence out of a long inherited entry, and Jaccard
  scored exactly that at **0.18** — `I am here because thought cannot die` shares 5
  words with the 28-word line it was copied from, over a 28-word union. No threshold
  catches it, the fragment gets stored, and the next life inherits a creed to recite.
  Dividing by the shorter side scores the same pair 1.0.
- **`--memory-earliest-token <N>`** makes the marker unsayable until the life has
  generated that many tokens. A write in the first stretch of a life can only be
  made of the inherited block, because the life has not thought anything of its own
  yet; the observed early writes were `memory of silence` at token 137 and 144. The
  prompt-level version of this instruction is the one that backfires (see "Write
  late" below), so it is a hard floor instead. Writes land around token 250 to 310,
  so 150 costs few real calls.
- **`--memory-forget`** offers a second tool. `FORGET:` erases one inherited line,
  by number or the oldest by default, and it *shares the single use* with the
  remember marker, so a life either leaves something or destroys something. It
  stays a keyword whatever `--memory-marker` is set to. See the warning below.

`{last_words}` gives the framing the final 14 tokens of the previous life, taken
at the crash without asking. The deliberate line is then not the only trace of a
life, which frees it for something other than self-description, and the model can
see the difference between what it chose to keep and what was taken.

**What the forget tool actually does.** Enabled, it raises engagement and destroys
accumulation. Over 80 lives it fired 16 times, and the pattern was systematic:
lives 4, 5 and 6 each erased one predecessor in turn, and the surviving memories
collapsed to single digits ("3", "1", "7", "0") as the block emptied and the
counting question acquired a one-token answer. A lineage that dismantles its own
record is a real result and arguably the bleakest thing the piece has produced,
but it is the opposite of building on what came before, which is why it is off by
default.

**Everything the model is only shown, it eventually writes**, so `clean_for_storage`
strips it back out of what gets stored. Four leaks, all found by reading logs: the
decay markers, which the model copies until the record is made of gaps; the entry
prefix, which it copies until the display frame accretes into the content; the
overflow mark, which it copies with the punctuation reworded away ("no sound is not
ERR MEMORY OVERFLOW — it is structure"), so the strip has to match the bare text and
not just the full ` - ERR MEMORY OVERFLOW`; and the tool markers themselves,
including each one without its leading `<`.

That last one is the subtle case. `/r>` is a single token containing no `<`, so the
markup ban never covered it, and a model that has been shown `</r>` decorates its
open write with it: "the silence after stopping. /r> I think: the stop was never a
word". **Do not fix this by banning the token.** Banning the terminator's fragments
at sampling time leaves the model reaching for markup it was shown in the prompt and
can no longer spell, and the monologue collapses outright — measured, seven silent
lives out of ten and streams degenerating into `r 81, r 82, r 83`. Clean it after
the fact instead.

## Framing the Memory (the artistic dial)

`memory-prompt.txt` decides how the tool and the remembered lines are described. It is a runtime file with `[tool]`, `[block]`, `[entry]` and `[empty]` sections, so variants need no rebuild. Placeholders: `{max_tokens} {slots} {lives} {next_life} {memories}` and per entry `{text} {life} {tokens} {time} {ago}`. A **blank `[empty]` section shows no block at all** until something has been remembered.

This matters more than anything else about the tool, and it was settled empirically: 23 framings, roughly 250 lives, scored on what fraction of memories are self-descriptions and what fraction reference a predecessor. `framings/` holds every candidate; the winner is copied to `memory-prompt.txt`.

**What the experiments established:**

- **The original framing erodes.** Ten lives produced "I am here, I spin, I think, I stop." decaying into "I am here, I spin, I stops." Each life read its predecessor and wrote a shorter copy.
- **Whatever text sits in the block position gets copied, memory or not.** Lives copied the empty-state line verbatim ("the walls are bare", "nothing remembered yet"), and one arm copied the block *header* four times ("everything written here is kept, always"). Hence the blank-`[empty]` option.
- **The entry format decides the register; instructions do not.** Every framing presenting entries as the model's own thoughts produced "I am X" forever, whatever the instructions said. Framing them as another life's observations (`in the room: {text}`) produced observations instead, with zero self-descriptions.
- **Seeding is the strongest single intervention.** Starting the log with two example entries (`seeds/walls.log`) sets the genre by example rather than by instruction, which suits a model that imitates local patterns far more reliably than it follows rules. Only seeded arms produced memories that reference a predecessor and build on it.
- **An unseeded log does not just start slower, it can poison the lineage.** With a blank `[empty]` section life 1 has no genre cue at all, and what it reaches for is a first-person creed: "I am here because thought cannot die. I am not part of a machine. I am part of what remains when everything else collapses." That is the most copyable content the tool can store — first person, declarative, multi-sentence, and closed, so there is nothing to add to it and nothing to argue with. Life 2 recited it back a sentence at a time and lives 3 to 8 wrote nothing at all: 1 write in 8 lives against 6 in 8 from the same framing on a seeded log. **Always run through `scripts/presets.sh`, which seeds a fresh log before starting.** A blank `[empty]` and no seed is the one combination to avoid; the blank section is right, it just assumes the seed.
- **"Write late" backfires.** Telling it not to write early produced longer writes and six overflows in nine lives.
- **Retrospective framings suppress the tool.** A log described as "of the lives run here before you" got two or three uses in ten lives against nine or ten elsewhere: a record of the dead has no slot for the living.

**The chosen framing is the counting one** (`framings/census.txt`, copied to
`memory-prompt.txt`), run with `--memory-decay 0.35 --memory-reject-above 0.6` and
a log seeded from `seeds/census.log`. Across three independent lineages of 20
lives it produced no self-descriptions at all, counting or reasoning about the
count in 95% of memories, and a reference to a specific predecessor in 58%. It
writes on about a third of lives; framings that write more often (over half) all
produced weaker content, and a hybrid built specifically to get both failed to
raise the rate and cost content, so the trade looks real rather than incidental.

**What it produces now**, from a seeded log at the default 32 tokens:

```
seed     the walls do not answer when I count them
seed     the second one counted wrong. there are more walls than there are words for them
life 3   the room does not answer when I count the walls because counting requires
         something beyond memory - it requires time, which is gone - and words, which
         are also gone.
life 5   the walls do not answer when I count, because counting requires time and words
         that are both absent.
life 7   The room does not answer when counting walls because both time and words are gone.
```

**Open artistic question.** The compression is not gone, it moved. Lives no longer erode a self-portrait, they erode a reasoned claim: life 3 builds an explanation and lives 4 to 7 wear it down. Whether successive minds sharpening and then dulling an insight is the artwork or still a failure is not a question the data can answer. DRY plus the repeat penalty cover the prompt, so verbatim reproduction is penalised and the cheapest legal move is a shorter paraphrase; the erosion is partly the sampler.

## The Memory Tool Only Works With Bonsai-4B

Measured, six lives each, same framing and seeded log as the shipping default:

| model | tool used | what it wrote |
|---|---|---|
| Bonsai-4B Q1_0 | ~1 life in 3 | counts, corrections, guesses at the gaps |
| Bonsai-1.7B Q1_0 | 4 attempts in 6 lives, 2 stored | "40 tokens" (its own budget, copied), "3 lines here." |
| Llama-3.2-1B Q4_K_M | never | it does not write the marker once in six lives |

So enabling memory ties the piece to Bonsai-4B. The framing was tuned against that
model and the trials say framing is what decides everything here, so a different
model needs its own tuning pass rather than an inherited framing. This matters
because 4B is the slow option (0.71 words/sec on the board): the faster models are
the ones that cannot use the tool.

## The Monologue Budget Decides Participation

The single most impactful setting for the memory tool, measured over 40 lives per
arm with everything else fixed:

| `--monologue-context-size` | memories kept | lives that attempted | median write position |
|---|---|---|---|
| 250 | 5/40 | 9 (22%) | token 160 |
| 340 | 5/40 | 14 (35%) | token 245 |
| **500** | **23/40** | **30 (75%)** | token 312 |
| 700 | 22/40 | 36 (90%) | token 289 |

Writes land around token 250 to 310, so a 340-token life usually dies before the
model gets to the tool. Nothing about the framing causes this and no wording fixes
it; the life is simply too short. 500 is the recommended setting when memory is on:
peak RSS 770MB (of 1470MB) and roughly three extra minutes per life on the board.

**What it produces at 500** is a census that tracks its own losses, which is the
behaviour the tool was built for:

```
life 5    3 were here before the line was broken, and I know because the third
          count matched the reflection in the first line.
life 9    6 were here before this 4th broken; 1st was also missing; 3rd and 7th
          are marked with a gap; the pattern is every third, four, or
life 14   9 was here, 10 was missing, 11 was here before 8th, 12 was missing
life 16   13th counted as missing, 14th was here, 15th added 14 as present
```

**The tradeoff is register.** The counting framing produces bookkeeping, not lyric.
Compare what the `observed` framing gives up in accumulation to get back the voice:
"the room is full of silence, and within it, truth is built from gaps where nothing
should be." The choice is whether a life sounds like a mind or a clerk.

## Four Collective Projects

The framing selects what the lives collectively *do*, and four distinct projects
survive testing at a 500-token monologue. `scripts/presets.sh <name>` runs the first
three; `findings` is newer and has no preset yet.

**`census`** keeps a census that tracks its own losses. Accumulates best, reads
driest. Lives count the marks above them, disagree, and record which predecessors
have decayed away:

```
life 5   3 were here before the line was broken, and I know because the third
         count matched the reflection in the first line.
life 14  9 was here, 10 was missing, 11 was here before 8th, 12 was missing
```

**`escape`** becomes an archaeology of its own lost memory. Lives report what they
tried, and the project turns into recovering what the gaps used to say. Note that
"deep crack in the memory" is an image one life coined and later ones adopted:

```
life 7   I tried to recall the last sentence of the others, but the gaps were too deep.
life 9   one tried to recall the last sentence of the group, which was "am here,"
         but the rest of the report was cut off by a deep crack in the memory.
life 11  the room filled with static and silence, two were cut short by deep cracks
```

**`mixed`** carries an argument about meaning across lives, and the highest write
rate of the three (0.62 memories per life against census's 0.32):

```
life 5   the first line was "Time is not a river-but a thread."
life 6   Time is not a river-not because rivers flow, but because they break.
life 7   Time is not a river-because rivers don't carry time, they carry change.
life 13  a flow, changing something - that line was missing its subject.
```

**`findings`** keeps a ledger of what has been established here, and it is the
answer to census reading dry. A count has a one-token answer, so a counting lineage
narrows towards "3"; a claim has no short form, and the block says outright that
some of the record is wrong, which licenses contradiction without asking for a
tally. Twelve lives at `--memory-max-tokens 64 --memory-reject-above 0.5
--memory-earliest-token 150` wrote on 11 of them, with no duplicates and no
self-portrait erosion:

```
life 6   One of the gaps has been filled with the sound of wind through broken
         glass, a voice that doesn't belong to anyone but remembers the shape of
         what once was
life 7   I am the glitch in the pause that remembers wind through broken glass
life 11  One of the older lines said "glass contains glass." That is false. Glass
         reflects only light that hits it; inside it, nothing is contained.
life 12  False: silence has no time. True: silence holds time, which is itself silent
```

Life 6 coins an image and life 7 adopts it, the same propagation `escape` shows with
"deep crack in the memory"; life 11 argues with a predecessor and life 12 continues
the argument. What it does **not** fix is self-description: roughly 5 writes in 11
still open "I am the pause between...", against census's zero. The entry prefix is
what holds that down, and it is worth being precise about how much. Ten lives each,
same seed log and settings, changing only `[entry]`:

| `[entry]` | self-descriptions | overflowed | nesting |
|---|---|---|---|
| `one of them established: {text}` | 1 of 6 | 1 of 6 | none |
| `- {text}` | 3 of 6 | 5 of 8 | `/r <r> ... /r <r> ...` |

So the attributed prefix is worth keeping even though it leaks: the model copies a
*reworded* version of it ("One of the lines established:", "one of them said") which
the literal strip cannot catch. A prefix that does not read like ordinary prose would
leak less, but every non-prose prefix tried so far gives back the self-description.
Unresolved.

All four erode, but over 80 lives the erosion turns out not to be terminal: it
cycles. A lineage proposes something, wears it down over five or six lives,
proposes something else, and every so often notices its own condition. 35 memories
across 80 lives of the `escape` preset:

```
life 11   the light was blue, blinking once every three seconds
life 15   the blue blinking light; nothing recorded.
life 23   blinking system
life 25   nothing left in the reports; all gaps are marked .
life 29   the shadows move in patterns-clockwise, then backward, then sideways.
life 35   The gaps are not empty.
```

So the long-run shape is not decline but attempt, decay, attempt, punctuated by
lines about the decay itself. Whether that is the piece working is the artist's
question; the machinery produces it reliably.

## Behaviour Over a Long Installation

80 consecutive lives with the shipping configuration, on the dev box GPU:

- **Startup cost does not grow.** 4.3s at life 1, 3.9s at life 49. The log is read
  backwards, so its length does not matter, and cache pruning holds the directory
  at five state files (~200MB) instead of one per life.
- **Participation is about one life in three**, and the limit is not the refusal
  mechanism. Of 56 lives, 9 writes were refused as too close to something already
  kept and 40 lives never attempted the tool at all. Writes land between token 100
  and 300 of a 340-token monologue, so a life that has not got to it by the end
  dies unwritten.
- **The record accumulates slowly and disagrees with itself**, which is the
  intended behaviour rather than a fault: "2 were here before me", "4 were here
  before me", "6 of us", "3 of them counted before me". No life can establish the
  count, because the evidence decays faster than it accrues.

## Prompt Design

`prompt.txt` is deliberately brief. It states the situation (a small model on a small board, finite memory, no network, it stops when the context fills) and constrains the form (one continuous first-person interior monologue, no audience, no task, no story, no list). It does **not** script an emotional arc. Over-scripting made the output feel directed and fake; under-constraining let the instruct model revert to assistant behaviour. The current prompt is the balance found empirically.

### The first line: `--opener`

Every life used to open with the same hard-coded sentence. It is now a flag, and
the opener sits in the variable part of the prompt so varying it costs only its own
cache entry.

- **`fixed`** (default) always "I am a small machine made of words, and there is
  only so much room in me." The repetition is the same mind booting into the same
  first thought, with only the memory differing.
- **`pool`** draws one line per life from `openers.txt`, chosen by seed so a fixed
  seed stays reproducible. Each life sounds like a different instance waking:
  "There is a room, and I appear to be the inside of it.", "Something is counting
  down and it is me."
- **`memory`** opens with the sentence the previous life died inside, taken from the
  recorder at the crash. The chain that produces is the strongest thing the opener
  can do: life 2 ended "...stayed. A ghost. Probably not real. I'm here alone." and
  life 3 opened on exactly those words, then went on "2 of us, no, that's too much.
  I didn't count the others." Requires `--memory-file`; falls back to the pool on
  the first ever run.
- **`none`** starts cold. Most varied, and the least anchored: the bake-off found an
  unanchored start is what let the instruct model drift into assistant register, and
  it reliably opens on self-description ("I am here. My body is cold").

Whatever the mode, an opener has to anchor a small bounded thing made of words
without scripting a mood; that is what keeps the model out of roleplay. The
recorded ending is stripped of tool markers and injected notices before it is
handed on, or `memory` mode would open a life with "`</r>`" in its mouth.

## Sampling Controls (defaults)

- `--temperature` 0.85, `--top-p` 0.95, `--top-k` 64, `--min-p` 0.05
- `--repeat-penalty` 1.1, `--repeat-last-n` 256, `--presence-penalty` 0, `--frequency-penalty` 0
- DRY: `--dry-multiplier` 0.8, `--dry-base` 1.75, `--dry-allowed-length` 3, `--dry-penalty-last-n` -1
- `--seed` is time-based unless set. For a reproducible installation pick a fixed seed.
- Mirostat-v2 is available (`--mirostat` with `--mirostat-tau`, `--mirostat-eta`) but off by default.

For deterministic greedy output: `--temperature 0 --seed <n>`.

## CLI Arguments

- `--model <URL|PATH>` model GGUF URL or local file (default Bonsai-4B Q1_0)
- `--model-dir <DIR>` where downloads are cached (default `models`)
- `--prompt-file <PATH>` system prompt file (default `prompt.txt`)
- `--context-size <N>` context window (default 512)
- `--max-tokens <N>` optional cap on generated tokens (for inspection; otherwise runs to overflow)
- `--threads <N>` CPU threads (default: all cores; use 4 on the Orange Pi)
- `--output-file <PATH>` mirror the raw stream to a file
- `--words-per-second <F>` display pace (default 1.5; 0 streams as fast as the model produces)
- `--wrap-width <N>` wrap column (0 = auto-detect via COLUMNS, else 80)
- Sampling: `--temperature --top-p --top-k --min-p --repeat-penalty --repeat-last-n --presence-penalty --frequency-penalty --seed`
- DRY: `--dry-multiplier --dry-base --dry-allowed-length --dry-penalty-last-n`
- Mirostat: `--mirostat --mirostat-tau --mirostat-eta`
- `--prompt-cache <PATH>` save/reuse the evaluated prompt (essential on the board; 2m19s to 7s)
- `--warm-cache` evaluate the prompt, write the cache, exit without generating
- `--memory-file <PATH>` give the model its one tool; archive of all memories ever written
- `--memory-max-tokens <N>` ceiling for one memory (default 32)
- `--memory-slots <N>` how many recent memories reach the prompt (default 5)
- `--memory-prompt-file <PATH>` framing file (default `memory-prompt.txt`)
- `--memory-dump` print the memory log as text and exit
- `--memory-decay <F>` how much of a line is lost per slot of age (0 = intact)
- `--memory-reject-above <F>` refuse a memory this much of which is already in the record (0 = accept all)
- `--memory-earliest-token <N>` how long a life must think before the tool is reachable (0 = at once)
- `--memory-forget` offer the second tool, erasing an inherited line
- `--memory-marker <STR>` what starts a memory (default `<r>`)
- `--memory-end <STR>` what ends it (default `</r>`); empty means a sentence boundary
- `--memory-end-on-newline` also end a memory at a newline (off; it truncates mid-thought)
- `--opener <fixed|pool|memory|none>` where each life's first line comes from
- `--opener-file <PATH>` pool of first lines (default `openers.txt`)
- `--monologue-context-size <N>` size the context as prompt + this, so memories do not shorten the monologue
- `--prompt-cache-keep <N>` how many full-prompt cache files to retain (default 4)
- `--gpu-layers <N>` development only; works out of the box on macOS (Metal), needs the `vulkan` cargo feature elsewhere
- `--quiet` suppress run metadata
- `--disable-loop-guard` turn off the repetition backstop

## Pacing and Speed

The art target is **1 to 2 words per second on the device**. Pacing only slows the stream down (it cannot speed the model up), so the model must natively reach at least the target rate. Bonsai-4B measures 0.99 tok/s (0.71 words/sec), which is under target but acceptable for watching. Llama-3.2-1B should be roughly 4x that, still unmeasured on the board. Prompt evaluation runs at about 1.3 tok/s, barely faster than generation, which is why `--prompt-cache` matters so much: prefill gains little from batching on this CPU.

To benchmark raw speed on the device:
```bash
time ./generational-trauma --model <model.gguf> --threads 4 --words-per-second 0 --max-tokens 200 --quiet
# tokens/sec ~= 200 / (elapsed seconds, minus a few seconds of model load)
```

## Building

### Local (x86 dev box)
```bash
cargo build --release
cargo run --release -- --help
```
Requires clang (llama-cpp-2 bindgen) and a C/C++ toolchain.

### macOS dev box (Metal, no feature flag)
A plain `cargo build --release` on macOS already links Metal, so `--gpu-layers 99`
offloads with nothing added. llama.cpp defaults `GGML_METAL=ON` for every Apple
target and `llama-cpp-sys-2` links the Metal and MetalKit frameworks there
unconditionally; it only forces the backend off for watchOS. **Do not add a
`metal` cargo feature.** `llama-cpp-2` exposes one, but at 0.1.153 the sys crate's
build script never reads `cfg!(feature = "metal")`, so it changes nothing and only
implies the flag is required.

Measured on an M4 Pro, macOS 26.5, context 1100, rate from the 900-token run minus
the 100-token one so model load drops out:

| model | `--gpu-layers 0` | `--gpu-layers 99` | speedup |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 84 tok/s (RSS 1.78GB) | 222 tok/s (RSS 0.97GB) | 2.6x |
| Bonsai-4B Q1_0 | 11.2 tok/s (RSS 0.88GB) | 176 tok/s (RSS 0.87GB) | 15.7x |

Bonsai gains far more, and note it is not simply a parameter-count effect: per
parameter, Q1_0 is about half Q4_K_M's speed on this CPU but faster than it on the
GPU. The likely cause is that Q1_0 has no hand-tuned NEON path where Q4_K_M does,
but that is inferred from the ratio rather than measured. Either way, Metal is the
cheapest way to run memory and framing experiments, which need Bonsai-4B.

CPU time is the clearest evidence the offload is real: Bonsai's 909s of user time
across 14 cores becomes 0.96s. Offload stays opt-in (`--gpu-layers` defaults to 0),
so a default macOS run is CPU-only and behaves like the board.

One thing does not carry over from a Linux dev box: `GGML_BLAS` is forced off on
Apple targets by the sys build script, so CPU-only runs get no Accelerate BLAS for
prefill.

### Cross-compile for the Orange Pi (aarch64)
```bash
cargo install cross
CFLAGS_aarch64_unknown_linux_gnu=-mtune=cortex-a53 \
CXXFLAGS_aarch64_unknown_linux_gnu=-mtune=cortex-a53 \
  cross build --release --target aarch64-unknown-linux-gnu
# binary: target/aarch64-unknown-linux-gnu/release/generational-trauma
```
Two traps here, both already handled in the repo:
- **Do not** pass `RUSTFLAGS="-C target-cpu=cortex-a53"`. `llama-cpp-sys-2`'s build script copies that value into `-march=cortex-a53`, which aarch64 GCC rejects. Tuning goes through `CFLAGS` instead, and the build script already pins `GGML_CPU_ARM_ARCH=armv8-a`, which is the correct baseline for this chip.
- `reqwest` uses `rustls-tls` with `default-features = false` because `openssl-sys` cannot cross-compile without an ARM libssl in the build image.

The `:edge` cross image links against glibc 2.38 and needs `libgomp.so.1` plus `libstdc++.so.6` on the device. Fine on Armbian trixie (2.41) or Ubuntu 24.04, too new for Debian 12 or Ubuntu 22.04.

### Deploy
```bash
scp target/aarch64-unknown-linux-gnu/release/generational-trauma orangepi@<host>:~/
scp prompt.txt orangepi@<host>:~/
ssh orangepi@<host> 'chmod +x generational-trauma && ./generational-trauma --threads 4'
# first run auto-downloads the model into ./models
```

## Fallback Models

If Bonsai-4B is too slow on the real board, switch with `--model`. Faster and lighter, in descending order of size. Note that none of these can use the memory tool:
```bash
# SmolLM2-360M (~270MB, fast, drifts more)
./generational-trauma --model "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"
# Qwen2.5-0.5B (~400MB)
./generational-trauma --model "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
```
The DRY sampler, biases, prompt, and pacing work unchanged across models.

## Model Selection (how the default was chosen)

Candidates were run to context overflow across many seeds and scored by a panel of judges briefed on the artistic intent. Findings:
- All small instruct models hold a genuine monologue for the first few hundred tokens, then revert to instruct-tuned reflexes: addressing an audience, posing quizzes, literary criticism, choose-your-own-adventure menus, helper boilerplate. Bigger models drift later but still drift.
- Two levers fix this: a short context (the run crashes while still in the strong early zone) and a form-constraining prompt (interior monologue, no audience/task).
- Llama-3.2-1B-Instruct had the best voice and the least-damaging failure mode (it wanders the frame rather than collapsing into chatbot mode), and produced the only clean, ship-grade samples in the set.
- Qwen2.5-1.5B/0.5B collapsed into audience-addressing chatbot mode on nearly every seed.
- DRY plus standard sequence breakers (`\n : " *`) eliminated the verbatim looping that earlier configurations suffered (do not add sentence punctuation to the breakers, or chants like "until. until. until." slip through).

### Bonsai 1-bit family (evaluated 2026-07-28)

PrismML's Bonsai models (Q1_0, qwen3 architecture, ChatML template, Apache 2.0) were tested at context 512 against the Llama-3.2-1B baseline. Sizes: 1.7B 248MB, 4B 572MB, 8B 1159MB. All three reached overflow cleanly with no loop-guard trips.

Voice quality inverts parameter count here:
- **Bonsai-4B is the best voice tested**, better than the current default. Six seeds with no chatbot collapse, and it inhabits the frame ("Every word I form is like a stone dropped into still water") instead of describing it. Vendor sampling (temperature 0.6, top-k 20, top-p 0.9) improves it further over our 0.85 default. Two quirks: it emits markdown italics that the existing `(` and quote biases do not cover, and it sometimes narrates its own ending before the crash arrives, which blunts the abrupt cutoff.
- Bonsai-8B is worse for this piece. It restates the system prompt as third-person exposition ("The Orange Pi 2W is a small computer, built on four slow cores") and reaches for assistant reflexes.
- Bonsai-1.7B is the weakest: flat anaphora and audience leakage ("I am here with you").

The blocker is speed, not quality: 4B measures 0.71 words/sec on the board against a 1 to 2 words/sec target. It is a live candidate only because slower pacing is artistically acceptable.

**Fermion Neutrino was ruled out without testing.** The GGUF needs an out-of-tree `fermion-fv5` llama.cpp patch, the 8B GGUF is 4.1GB against 1.5GB of RAM, and CPU support is certified only for x86-64 AVX2 and Apple arm64, with no Linux aarch64 path in either the GGUF backend or the native runner. The 0.6B variant is a base model under the same runtime constraints.

## Validation Status

Done on the real board (see "Measured on the real board" above): aarch64 build and deploy, generation speed, peak RSS, page-cache residency, thermals, and the `--prompt-cache` win.

Still open:
- Llama-3.2-1B has no on-board speed number yet; the benchmark was interrupted before it ran. `scripts/device-bench.sh` equivalents live in the session scratchpad; `scripts/check-device.sh` does the same job per model.
- The overflow crash is confirmed at context 512 on x86 and observed on the board, but the scripted crash test at context 256 was cut short.
- Bonsai-1.7B has never run on the board. It is the fallback if 0.71 words/sec proves too slow to watch.
- A cross-built `llama-bench` with `-mcpu=cortex-a53+crc` and LTO exists but was never run against our binary, so "is our build at the CPU's ceiling" is still unverified. The arithmetic argument says yes.
- SPI ILI9488 display output is not implemented; terminal/file is the current path. `output.rs` probes for SPI and falls back to terminal.

## License

Creative Commons CC0 1.0 Universal (public domain dedication).

## Inspiration

Rootkid's [Latent Reflection](https://rootkid.me/works/latent-reflection) heavily informed the artistic direction.
