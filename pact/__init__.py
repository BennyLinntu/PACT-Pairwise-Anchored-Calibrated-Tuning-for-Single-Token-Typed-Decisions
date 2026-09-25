"""PACT: Pairwise-Anchored Calibrated Tuning for single-token typed decisions.

This folder is self-contained. Copy ``new_methods/`` anywhere - a server, a
container, a scratch disk - and it carries everything a run needs:

    new_methods/
      nimble/     byte-identical copies of the upstream modules the prompt
                  contract depends on (parallel_schema, evaluate_pilot,
                  schema_data). Vendored so that the prompt, the answer codes
                  and the contract hash cannot drift from what the adapter was
                  trained against.
      data/       the frozen train.jsonl / eval.jsonl / manifest.json
      pact/       the method
      configs/    run plans
      .cache/     everything produced at run time (weights, views, runs)

Importing this package puts the bundle first on ``sys.path``, so ``nimble``
resolves to the vendored copy whether or not the parent repository is present.
"""

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
#: The self-contained bundle: the directory that holds pact/, nimble/, data/.
BUNDLE_ROOT = PACKAGE_ROOT.parent
#: Where the frozen dataset lives: the bundle if it carries one, else the
#: parent repository (useful when working inside a checkout).
DATA_ROOT = BUNDLE_ROOT if (BUNDLE_ROOT / "data" / "train.jsonl").exists() \
    else BUNDLE_ROOT.parent
#: Backwards-compatible alias used by older call sites.
REPO_ROOT = DATA_ROOT

for _path in (BUNDLE_ROOT, BUNDLE_ROOT.parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# The bundle must win over a checkout of the same name.
sys.path.remove(str(BUNDLE_ROOT))
sys.path.insert(0, str(BUNDLE_ROOT))

__version__ = "0.2.0"


def bundle_path(relative):
    """Resolve a path that belongs to this bundle (caches, configs, outputs)."""
    path = Path(relative)
    return path if path.is_absolute() else BUNDLE_ROOT / path


def data_path(relative):
    """Resolve a path that belongs to the frozen dataset."""
    path = Path(relative)
    return path if path.is_absolute() else DATA_ROOT / path
