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
| `anthropic_skills_prompts.tsv` | the corpus: `<skill-id>\t<kind>\t<prompt>` |
| `run-anthropic-prompts.sh` | scores every prompt via `ski why`, prints accuracy + misses |

`kind` is `direct` (explicit domain words), `task` (indirect/realistic),
`confusable` (two skills compete), or `negative` (should match nothing,
`skill-id = (none)`).

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

The default build uses the **offline bag-of-words embedder**, which matches on
lexical overlap only. Expect it to:

- handle `direct` prompts that reuse the skill's own vocabulary,
- miss most `task` prompts (synonyms, no shared tokens — e.g. "polished letter as
  a Word file" doesn't lexically reach `docx`), and
- let a few `negative`/distractor skills (`template-skill`, etc.) sneak above the
  `min_similarity` floor.

So **treat the accuracy as a lexical-floor baseline, not a quality bar.** The
semantic gains land with the `--features fastembed` (bge) lane; re-run both modes
and re-tune `min_similarity` / `score_margin` there. The corpus is the fixed
target the two embedders are compared against.
