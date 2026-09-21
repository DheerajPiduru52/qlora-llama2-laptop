"""Terminal chat against the running server. Same as `qlora chat`.

    python scripts/chat_cli.py [--url http://127.0.0.1:8080] [--temperature 0.7]
"""

from qlora_pipeline.bootstrap import load_env

load_env()

import argparse  # noqa: E402

from qlora_pipeline.config import load_config  # noqa: E402
from qlora_pipeline.serve import chat_repl  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="serve.yaml")
    ap.add_argument("--url", default=None)
    ap.add_argument("--temperature", type=float, default=None)
    a = ap.parse_args()
    chat_repl(load_config(a.config), url=a.url, temperature=a.temperature)
