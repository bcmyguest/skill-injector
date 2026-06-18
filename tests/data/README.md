# anthropic/skills prompt corpus

A collection of prompts for exercising ski's ranker against the skill library at
[github.com/anthropics/skills](https://github.com/anthropics/skills) — the 17
skills shipped by the `anthropic-agent-skills` marketplace plugin
(`~/.claude/plugins/marketplaces/anthropic-agent-skills/skills/`).

Two uses:

1. **Inject test** — does ski pick the right skill for a realistic request? The
   runner reports top-1 accuracy.
2. **Usage test** — each prompt is a realistic task for that skill, so the same
   list doubles as copy-paste prompts for driving the skills by hand in Claude
   Code or opencode.

## Files

| file | what |
|---|---|
| `anthropic_skills_prompts.tsv` | narrow corpus (17-skill anthropic lib): `<skill-id>\t<kind>\t<prompt>` |
| `popular_skills_prompts.tsv` | realistic corpus for a ~47-skill index (see below) |
| `phrase_trigger_prompts.tsv` | exact-trigger corpus for the phrase channel: positives that type a skill's literal quoted trigger, plus negatives sharing single phrase tokens (precision stress) |
| `run-anthropic-prompts.sh` | scores every prompt via `ski why`, prints accuracy + misses |

`kind` is `direct` (explicit domain words), `task` (indirect/realistic),
`confusable` (two skills compete), `negative` (should match nothing,
`skill-id = (none)`), or `borderline` (observe-only; excluded from headline
metrics).

## Precision corpus + in-process eval (`popular_skills_prompts.tsv`)

The narrow 17-skill anthropic library is *misleading* for over-injection: indirect
prompts have no good match there, so real matches score like noise. The realistic
corpus unions those 17 with the 31 highest-installed community skills from
skills.sh (`/var/tmp/ski-eval/.claude/skills`), giving a ~47-skill index where a
genuine knee exists between real matches and unrelated programming prompts. It
carries 43 positives (paraphrased/indirect across the library) and 52 negatives
(real C/C++/JVM/algorithms/theory/other-language/dev-env prompts that should match
*nothing*) — so a fired injection on a negative is a true false positive.

`examples/eval.rs` runs the **real two-stage decision** (stage-1 cosine, or the
cross-encoder when ambiguous) in-process — one model load, not the per-prompt
subprocess of `run-anthropic-prompts.sh` — and prints a confusion matrix (recall
on positives, false-positive rate on negatives) plus per-prompt score dumps:

```sh
SKI_ROOTS="/var/tmp/ski-eval/.claude/skills:$HOME/.claude/plugins/marketplaces/anthropic-agent-skills" \
  cargo run --example eval -- tests/data/popular_skills_prompts.tsv -v
```

Use it to tune the gate (`rerank_min`, `min_similarity`) against a realistic
distractor set instead of overfitting a scalar to a handful of prompts.

`SKI_PHRASE_BOOST=<f>` overrides the phrase-channel boost for one run (`0.0`
disables the channel), so the same corpus can be scored with and without it to
isolate the phrase channel's effect — e.g. on `phrase_trigger_prompts.tsv`:

```sh
SKI_PHRASE_BOOST=0.0  cargo run --example eval -- tests/data/phrase_trigger_prompts.tsv
SKI_PHRASE_BOOST=0.20 cargo run --example eval -- tests/data/phrase_trigger_prompts.tsv
```

## Run

```sh
# from the repo root
./tests/data/run-anthropic-prompts.sh        # scoped: only the 17 skills compete
./tests/data/run-anthropic-prompts.sh -g     # global: every installed skill competes
./tests/data/run-anthropic-prompts.sh -v     # also print each prediction
```

- **Scoped** sets `SKI_ROOTS` to the marketplace dir, so the run measures
  discrimination *within* the library.
- **Global** uses ski's default roots, measuring real injection behaviour with
  every other installed skill acting as a distractor.

Inspect one prompt's full ranking:

```sh
SKI_ROOTS=~/.claude/plugins/marketplaces/anthropic-agent-skills \
  cargo run -q -- why "merge these three PDFs into one" --top 8
```

`SKI_ROOTS` is a colon-separated root override (opt-in; unset = ski's normal
roots). It scopes `index`, `why`, `hook`, and the rest of the CLI alike.

## Reading the numbers

The offline lane (`SKI_OFFLINE=1`) uses the **bag-of-words embedder**, which
matches on lexical overlap only. Expect it to:

- handle `direct` prompts that reuse the skill's own vocabulary,
- miss most `task` prompts (synonyms, no shared tokens — e.g. "polished letter as
  a Word file" doesn't lexically reach `docx`), and
- let a few `negative`/distractor skills (`template-skill`, etc.) sneak above the
  `min_similarity` floor.

So **treat the bag-of-words accuracy as a lexical-floor baseline, not a quality
bar.** The semantic gains land with the default `fastembed` (bge) lane; re-run both
modes and re-tune `min_similarity` / `score_margin` there. The corpus is the fixed
target the two embedders are compared against.
