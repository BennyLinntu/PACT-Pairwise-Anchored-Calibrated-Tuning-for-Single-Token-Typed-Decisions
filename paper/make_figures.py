#!/usr/bin/env python
"""Redraw the figures from a finished run.

``train_all.py`` already does this at the end of a sweep (they land in
``results/figures/`` and are mirrored next to this file). Use this script only
to redraw them, or to draw them for a different run:

    python paper/make_figures.py --run .cache/runs/pact-9b/pact_full/seed-17

Needs matplotlib; without it the numbers are still written as JSON.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pact import BUNDLE_ROOT  # noqa: E402
from pact.figures import build  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="A .../<run>/seed-<n> directory")
    parser.add_argument("--output-dir", help="The sweep directory holding results_aggregate.json")
    parser.add_argument("--out-dir", default=str(BUNDLE_ROOT / "paper" / "figures"))
    parser.add_argument("--bins", type=int, default=15)
    arguments = parser.parse_args()

    run_dir = Path(arguments.run)
    output_dir = Path(arguments.output_dir) if arguments.output_dir else run_dir.parents[1]
    written = build(run_dir, output_dir, arguments.out_dir, arguments.bins)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
