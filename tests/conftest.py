"""Test session setup: project env, offline Hub (cached tokenizer only), so network blips can't skip tests."""

import os

from qlora_pipeline.bootstrap import load_env

load_env()
os.environ["HF_HUB_OFFLINE"] = "1"  # set before huggingface_hub / transformers are imported by any test
