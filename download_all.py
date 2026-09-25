#!/usr/bin/env python
"""Fetch everything a PACT run needs, into this bundle's ``.cache/``.

Nothing else in this folder touches the network, so a machine that has run
this script once can train offline (``HF_HUB_OFFLINE=1``).

Typical uses:

    # laptop: tokenizer only - enough for preflight and for building the
    # view cache (a few hundred MB instead of ~18 GB)
    python download_all.py --tokenizer-only

    # server: the pinned base weights
    python download_all.py            # the default 2x48 GB config

    # optional auxiliary public data for the generalisation ablation
    python download_all.py --public boolq vitaminc --public-limit 3000

    # verify a cache without contacting the hub
    python download_all.py --check
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact import BUNDLE_ROOT, DATA_ROOT  # noqa: E402
from pact.config import PactConfig  # noqa: E402

CACHE = BUNDLE_ROOT / ".cache"
TOKENIZER_PATTERNS = ["tokenizer*", "*.json", "*.jinja", "*.model", "*.txt", "*.py"]
WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json"]


def download_model(model_id, revision, tokenizer_only, token=None, log=print):
    from huggingface_hub import snapshot_download

    patterns = TOKENIZER_PATTERNS if tokenizer_only else TOKENIZER_PATTERNS + WEIGHT_PATTERNS
    log(f"[download] {model_id} @ {revision or 'main'} "
        f"({'tokenizer only' if tokenizer_only else 'full weights'})")
    path = snapshot_download(model_id, revision=revision or None,
                             cache_dir=str(CACHE / "huggingface" / "hub"),
                             allow_patterns=patterns, token=token)
    log(f"[download] -> {path}")
    return Path(path)


def record(payload, path=None):
    path = Path(path or CACHE / "downloads.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return path


def check(cfg, log=print):
    """Report what is already cached, without any network access."""
    status = {"cache": str(CACHE), "downloads": None, "view_caches": [], "public": []}
    downloads = CACHE / "downloads.json"
    if downloads.exists():
        status["downloads"] = json.loads(downloads.read_text(encoding="utf-8"))
    views = CACHE / "views"
    if views.exists():
        for manifest in sorted(views.glob("*/manifest.json")):
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            status["view_caches"].append({"path": str(manifest.parent),
                                          "views": payload.get("views"),
                                          "complete": payload.get("complete")})
    public = CACHE / "public"
    if public.exists():
        for path in sorted(public.glob("*.jsonl")):
            status["public"].append({"path": str(path),
                                     "rows": sum(1 for _ in path.open(encoding="utf-8"))})
    status["data_files"] = {
        name: (DATA_ROOT / "data" / name).exists()
        for name in ("train.jsonl", "eval.jsonl", "manifest.json")}
    log(json.dumps(status, indent=2))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path,
                        default=BUNDLE_ROOT / "configs" / "server_2x48gb.json")
    parser.add_argument("--model-id", help="Override the config's base model")
    parser.add_argument("--revision", help="Override the config's pinned revision")
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--public", nargs="*", default=[],
                        help="Auxiliary datasets to convert: boolq paws vitaminc multinli")
    parser.add_argument("--public-split", default="train")
    parser.add_argument("--public-limit", type=int, default=3000)
    parser.add_argument("--check", action="store_true", help="Report the cache and exit")
    arguments = parser.parse_args()

    cfg = PactConfig.load(arguments.config) if arguments.config.exists() else PactConfig()
    if arguments.check:
        check(cfg)
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    summary = {}
    if not arguments.skip_model:
        model_id = arguments.model_id or cfg.model.model_id
        revision = arguments.revision if arguments.revision is not None else cfg.model.revision
        path = download_model(model_id, revision, arguments.tokenizer_only, token)
        summary["model"] = {"model_id": model_id, "revision": revision,
                            "path": str(path), "tokenizer_only": arguments.tokenizer_only}
    if arguments.public:
        from pact.public_data import CONVERTERS, write_rows

        summary["public"] = []
        for name in arguments.public:
            if name not in CONVERTERS:
                raise SystemExit(f"Unknown dataset {name}; choose from {sorted(CONVERTERS)}")
            rows = CONVERTERS[name](split=arguments.public_split, limit=arguments.public_limit)
            path = write_rows(rows, CACHE / "public" / f"{name}.jsonl")
            summary["public"].append({"dataset": name, "rows": len(rows), "path": str(path)})
            print(json.dumps(summary["public"][-1]))
    record(summary)
    print(json.dumps(summary, indent=2))
    print("\nNext:  python train_all.py")


if __name__ == "__main__":
    main()
