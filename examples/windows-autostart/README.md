# Optional: run a Laya engine at login (Windows)

**You do not need this for `laya-mcp`.** The MCP server spawns its own `laya daemon` child on
first use, because an MCP stdio server cannot attach to a foreign process. This is for the case
where you *also* want a warm HTTP endpoint for callers that are not MCP clients — `curl`, cron
scripts, the Decision Studio UI at `http://localhost:8131/`.

## Install

1. Edit the three paths at the top of `laya-serve.cmd` (`LAYA_HOME`, `EXE`, `MODEL`).
2. Put both files where convenient and point the `.vbs` at your `laya-serve.cmd` path.
3. Copy the `.vbs` into your Startup folder: press `Win+R`, run
   `shell:startup`, and drop it in.

The same convention as any other login launcher (`OpenClaw Node.vbs`, `headroom-proxy-start.vbs`):
a thin `.vbs` runs a `.cmd` with window style `0`, so no console flashes and the long-lived
server inherits the hidden console.

## What you get

```bash
curl http://127.0.0.1:8131/health
# {"status": "ok", "device": "cuda:0", "family": "multilingual", "model": "laya-multilingual", ...}

curl -X POST http://127.0.0.1:8131/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":{"body":"二重に請求されました"},"questions":{"dept":{"type":"choice",
       "instructions":"Which team?","criteria":{"billing":"invoices, refunds","sales":"pricing"}}}}'
```

Measured cost: **~260 MiB VRAM resident** (Q8_0 on an RTX 3090), cold start ~2 s.

## Behaviour worth knowing

- **Idempotent.** The port check runs first, so a second login (or running it by hand) exits
  without starting a twin engine. Verified: same PID after a second launch.
- **Two logs, on purpose.** The engine holds `serve-autostart.log` open for its entire life, so
  the launcher's own breadcrumbs go to `launcher.log` — appending status to a file another process
  has locked fails with *"being used by another process"*, which is precisely the path that runs
  on every login after the first.
- **`laya-stop.cmd`** (in the parent folder) kills whatever listens on the port: `laya-stop.cmd`,
  or `set LAYA_PORT=8132 & laya-stop.cmd`.
- **Test without disturbing a live engine:** `set LAYA_PORT=8132` then run the `.cmd` directly.

## macOS / Linux equivalent

Same idea, one file instead of two:

```bash
# ~/.config/systemd/user/laya.service  (or a LaunchAgent plist on macOS)
[Unit]
Description=Laya System-1 decision engine
[Service]
ExecStart=%h/ggmlc/laya serve %h/models/laya_multilingual_q8_0.gguf --port 8131 --device auto --cuda-graph
Restart=on-failure
[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now laya.service
```

Again: optional. The MCP server does not read this endpoint.