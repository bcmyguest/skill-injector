import type { Plugin } from "@opencode-ai/plugin"
import { existsSync } from "node:fs"
import { homedir } from "node:os"

// skill-inject — opencode adapter.
//
// Embeds the user prompt locally, ranks it against the skill corpus, and injects
// the matching skill's directive so the model loads it — no cloud decision, no
// tool-call round trip. All ranking, injection-text, and session bookkeeping
// lives in the `ski` binary (`ski hook|observe|session-start --host opencode`);
// this plugin is a thin event -> stdin -> stdout bridge, the opencode twin of the
// Claude hooks adapter.
//
// FAIL OPEN: if `ski` can't be found, or any call errors, the plugin does
// nothing and never blocks a prompt or a tool call. `ski` owns the same
// contract (errors -> empty output, exit 0).

type Json = Record<string, unknown>

export const SkiOpencodePlugin: Plugin = async ({ $, directory }) => {
  // Resolve `ski` once: PATH first, then the usual cargo / local bins — the
  // same search order as the Claude adapter's ski-bootstrap.sh.
  const resolveSki = async (): Promise<string | null> => {
    try {
      const onPath = (await $`which ski`.quiet().nothrow().text()).trim()
      if (onPath) return onPath
    } catch {
      // `which` failed — fall through to the explicit candidates.
    }
    for (const cand of [`${homedir()}/.local/bin/ski`, `${homedir()}/.cargo/bin/ski`]) {
      if (existsSync(cand)) return cand
    }
    return null
  }

  const ski = await resolveSki()
  if (!ski) {
    console.warn(
      "[ski] `ski` binary not found on PATH, ~/.local/bin, or ~/.cargo/bin — skill-inject disabled. " +
        "Install it: cargo install --path <repo> --features fastembed",
    )
    return {}
  }

  // Run `ski <args>` feeding `payload` as JSON on stdin; return trimmed stdout
  // ("" on any error — fail open). stdin is fed by redirecting a Blob: Bun's
  // shell has no writable `.stdin` stream, and `< ${ReadableStream}` panics, but
  // `< ${Blob}` is supported.
  const runSki = async (args: string[], payload: Json): Promise<string> => {
    try {
      const stdin = new Blob([JSON.stringify(payload)])
      const res = await $`${ski} ${args} < ${stdin}`.quiet().nothrow()
      return res.stdout.toString().trim()
    } catch {
      return ""
    }
  }

  // Best-effort reindex on load so a session picks up new / edited skills.
  // Reindex needs no session id; a bare `startup` source triggers it without
  // clearing any ledger.
  await runSki(["session-start", "--host", "opencode"], { source: "startup" })

  // Directive computed in `chat.message`, consumed in the system-prompt
  // transform of the same turn (keyed by session). chat.message can't reach the
  // system prompt, so the two hooks hand off through here.
  const pending = new Map<string, string>()

  return {
    "chat.message": async (input, output) => {
      // The prompt is the user's text parts; skip any synthetic part so
      // re-ranking never feeds on injected content.
      const prompt = output.parts
        .flatMap((p) => (p.type === "text" && !p.synthetic ? [p.text] : []))
        .join("\n")
        .trim()
      // Always clear last turn's directive first so a no-match turn can't drain
      // a stale one.
      pending.delete(input.sessionID)
      if (!prompt) return

      const raw = await runSki(["hook", "--host", "opencode"], {
        prompt,
        session_id: input.sessionID,
        cwd: directory,
      })
      if (!raw) return

      let decision: { inject?: string; skills?: string[] }
      try {
        decision = JSON.parse(raw)
      } catch {
        return // malformed output -> inject nothing.
      }
      const inject = decision.inject?.trim()
      if (inject) pending.set(input.sessionID, inject)
    },

    "experimental.chat.system.transform": async (input, output) => {
      // Inject as additional context via the system prompt, not as a message
      // part: the directive must read as injected guidance, never as text the
      // user typed. (Appending to the user's part also caused that; pushing a
      // new part hits opencode's `prt`-id validation, which fails the whole
      // prompt.) Drain the directive stashed for this session this turn.
      if (!input.sessionID) return
      const inject = pending.get(input.sessionID)
      if (!inject) return
      pending.delete(input.sessionID)
      output.system.push(inject)
    },

    "tool.execute.after": async (input) => {
      // Observe skills the model pulled in itself so the hook's dedup won't
      // re-inject them. Normalize opencode's tool name + args to the same
      // {tool_name, tool_input:{file_path, skill}} shape `ski observe` reads for
      // Claude (a `Read` of a SKILL.md, or a skill invocation by name).
      const tool = String(input.tool ?? "").toLowerCase()
      const args = (input.args ?? {}) as Json
      const tool_input: { file_path?: string; skill?: string } = {}
      let tool_name = ""
      if (tool === "read") {
        tool_name = "Read"
        tool_input.file_path = String(args.filePath ?? args.file_path ?? args.path ?? "")
      } else if (tool.includes("skill")) {
        tool_name = "Skill"
        tool_input.skill = String(args.skill ?? args.name ?? "")
      } else {
        return
      }
      await runSki(["observe", "--host", "opencode"], {
        session_id: input.sessionID,
        tool_name,
        tool_input,
      })
    },

    event: async ({ event }) => {
      // A compaction restarts the session from a summary; clear the loaded
      // ledger so the relevant skills inject again into the fresh context.
      if (event.type === "session.compacted") {
        await runSki(["session-start", "--host", "opencode"], {
          session_id: event.properties.sessionID,
          source: "compact",
        })
      }
    },
  }
}

export default SkiOpencodePlugin
