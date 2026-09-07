
## 03.2-07: intermittent KunRunner teardown hang in tests/test_factor_stream.py

Observed once during 03.2-07 execution: a full `uv run pytest -q` run wedged
indefinitely (>10 min) with the main thread parked in
`kun::StreamContext::~StreamContext()` inside `KunRunner.abi3.so`, reached via
`subtype_dealloc`. Killing and re-running the identical command completed in
18s, and `tests/test_factor_stream.py` alone passes in 2.3s.

Pre-existing and unrelated to this plan's changes (no file this plan touches is
reachable from the KunQuant stream path). Logged rather than fixed per the
executor scope boundary. If it recurs, the lead is the C++ stream-context
destructor's thread join, not the Python test.
