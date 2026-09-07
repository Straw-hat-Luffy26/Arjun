"""Sarathi graph sidecar.

Listens for JSON-RPC 2.0 frames on stdin and answers on stdout, one JSON object
per line. No sockets, so no firewall prompt and nothing to bind -- the same
shape as `sidecars/memory_engine_sidecar/main.py`, deliberately, because that
transport is already trusted here and a second convention would be a second
thing to get wrong.

Run it directly for a smoke test:

    ARJUN_REBEL_DIR=... python main.py
    {"jsonrpc":"2.0","id":1,"method":"graph.ping","params":{}}
"""

import json
import os
import sys
import traceback

sidecar_dir = os.path.dirname(os.path.abspath(__file__))
if sidecar_dir not in sys.path:
    sys.path.insert(0, sidecar_dir)

from router import GraphActionRouter  # noqa: E402


def main() -> None:
    router = GraphActionRouter()
    sys.stdout.flush()

    for line in sys.stdin:
        if not line:
            break
        line_str = line.strip()
        if not line_str:
            continue

        try:
            req = json.loads(line_str)
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {}) or {}

            try:
                result = router.dispatch(method, params)
                response = {"jsonrpc": "2.0", "id": req_id, "result": result}
            except Exception as ex:
                # The traceback goes in `data` rather than to stderr, because a
                # sidecar failure the parent cannot see is a sidecar failure
                # nobody can debug.
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32603,
                        "message": str(ex),
                        "data": traceback.format_exc(),
                    },
                }
        except Exception as json_err:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"parse error: {json_err}"},
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
