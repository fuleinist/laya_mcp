# Optional: run a Laya engine at login (macOS)

**You do not need this for `laya-mcp`.** The MCP server starts its own `laya daemon` child on
first use, because an MCP stdio server cannot attach to a foreign process. This recipe is for the
case where you *also* want a warm HTTP endpoint for callers that are not MCP clients: `curl`, cron
scripts, the Decision Studio UI at `http://localhost:8131/`.

Requires the arm64 Metal build, `laya-macos-arm64-metal.tar.gz` from the
[ggmlc releases](https://github.com/monatis/ggmlc/releases). Intel Macs must use the CPU build and
`--device cpu`.

## Install

1. Edit the four `/Users/you` paths in `com.laya.serve.plist`: the binary, the model, and the two
   log paths.
2. Validate the file. `plutil -lint` must print `OK`.

   ```bash
   plutil -lint com.laya.serve.plist
   ```

3. Copy the file into your LaunchAgents directory and load it.

   ```bash
   mkdir -p ~/Library/LaunchAgents
   cp com.laya.serve.plist ~/Library/LaunchAgents/
   launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.laya.serve.plist
   ```

4. Check the endpoint.

   ```bash
   curl http://127.0.0.1:8131/health
   # {"status": "ok", "device": "metal", "family": "multilingual", "model": "laya-multilingual", ...}
   ```

## Manage it

```bash
launchctl kickstart -k gui/$(id -u)/com.laya.serve   # restart
launchctl bootout gui/$(id -u)/com.laya.serve        # stop and unload
tail -f ~/Library/Logs/laya-serve.log                # follow the log
```

## Behaviour worth knowing

- **No `--cuda-graph`.** A CUDA graph does not exist on Metal. The plist passes `--device metal`
  and stops there.
- **`KeepAlive` restarts on failure only.** `SuccessfulExit` is false, so a clean `bootout` stays
  stopped, and a crash comes back by itself.
- **`RunAtLoad` starts it at login.** The engine stays resident for the session.
- **One log file, on purpose.** The engine writes its own messages. A separate launcher script
  would add a second file to read for no gain, unlike the Windows `.vbs` + `.cmd` pair, which
  exists only to hide the console.
- **`--device auto` also works.** On Apple Silicon it selects Metal. The explicit `metal` value
  makes the intent readable in the plist.

Again: optional. The MCP server does not read this endpoint.