"""Loading provider keys from a file, without ever putting them on a command line or in a log.

The backends read their key from an environment variable, which is the right default for both a
service and an eval run. `--env-file` exists because the alternative in practice is exporting
secrets in a shell history or passing them as an argument, where they end up in `ps` output.
Existing environment values always win, so an explicit export still overrides the file.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | os.PathLike[str]) -> int:
    """Load KEY=VALUE lines into os.environ; returns how many keys were set. Values are not echoed."""
    loaded = 0
    for line in Path(path).expanduser().read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded