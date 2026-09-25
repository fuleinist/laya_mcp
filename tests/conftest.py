"""Session-wide test defaults.

`laya_mcp_server` defaults `LAY_USAGE_LOG` to `~/.laya-mcp/usage.jsonl`, so without this every
tool call made by the suite — including the ~30 tests that pass a fake daemon — would append to the
operator's real usage log, mixing test traffic into a record meant for real agent calls. (Measured
before the guard existed: one `pytest tests/` run added 25 records.)

Tests that assert *on* the log point `srv.USAGE_LOG` at a `tmp_path` file of their own, which
overrides this.
"""

import pytest


@pytest.fixture(autouse=True)
def _keep_the_suite_out_of_the_real_usage_log(monkeypatch):
    import laya_mcp_server as srv

    monkeypatch.setattr(srv, "USAGE_LOG", None)