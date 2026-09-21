"""Load test against the running server. Same as `qlora load-test`.

    python scripts/load_test.py [--url http://127.0.0.1:8080] [--n 20] [--concurrency 1]
"""

from qlora_pipeline.bootstrap import load_env

load_env()

import argparse  # noqa: E402
import json  # noqa: E402

from qlora_pipeline.config import load_config  # noqa: E402
from qlora_pipeline.serve import load_test  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="serve.yaml")
    ap.add_argument("--url", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    a = ap.parse_args()
    s = load_test(load_config(a.config), url=a.url, n=a.n, concurrency=a.concurrency)
    print(json.dumps({k: v for k, v in s.items() if k != "requests"}, indent=2))
