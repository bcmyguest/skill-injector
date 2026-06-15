# skill-inject — opencode adapter

The opencode twin of the Claude hooks adapter. It bridges opencode plugin events
to the `ski` binary so the right skill is embedded, ranked, and injected locally
on every prompt — no cloud decision, no tool-call round trip.

`ski.ts` is a thin bridge; all ranking, injection text, and session bookkeeping
live in the Rust binary. It fails open: if `ski` is missing or errors, the plugin
does nothing.

## What it wires

| opencode hook                          | ski command                         | purpose                                                          |
| -------------------------------------- | ----------------------------------- | --------------------------------------------------------------- |
| `chat.message`                         | `ski hook --host opencode`          | rank the prompt; stash the matching directive for the turn      |
| `experimental.chat.system.transform`   | —                                   | inject the stashed directive into the **system prompt**         |
| `tool.execute.after`                   | `ski observe --host opencode`       | record skills the model loaded itself (Read of SKILL.md, etc.)  |
| `event` (`session.compacted`)          | `ski session-start --host opencode` | clear the loaded-ledger so skills re-inject after compaction    |
| plugin load                            | `ski session-start --host opencode` | incremental reindex (picks up new / edited skills)              |

The directive is injected as **additional context** through the system prompt,
not as a message part — it must read as injected guidance, never as text the user
typed. `chat.message` (which sees the prompt) and `experimental.chat.system.transform`
(which sees the system prompt) hand off through an in-memory per-session stash.

The hook resolves `ski` from `PATH`, then `~/.local/bin`, then `~/.cargo/bin` —
the same order as the Claude adapter's `ski-bootstrap.sh`.

## Install

1. Build the binary (the default build ships the bge embedder + reranker; add
   `--no-default-features` for the offline bag-of-words lane):

   ```bash
   cargo install --path <repo>
   ```

2. Make opencode load the plugin. Either drop the file into opencode's
   auto-loaded plugin dir:

   ```bash
   mkdir -p ~/.config/opencode/plugin
   cp <repo>/opencode/ski.ts ~/.config/opencode/plugin/ski.ts
   ```

   …or reference it from `~/.config/opencode/opencode.json`:

   ```json
   {
     "plugin": ["<repo>/opencode/ski.ts"]
   }
   ```

`ski init -g opencode` does this for you — it writes the bundled plugin straight to
`~/.config/opencode/plugin/ski.ts` (the first option above), no manual copy needed.

## Skill roots

`ski` reads opencode skills from `~/.config/opencode/skills` (plus the shared
roots). Override with `SKI_ROOTS` (colon-separated).
