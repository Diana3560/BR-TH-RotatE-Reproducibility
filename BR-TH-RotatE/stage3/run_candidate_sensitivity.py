from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from stage3 import candidate_runner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-3 candidate-space sensitivity for D0/D1/A0/D2"
    )
    parser.add_argument("--dataset", choices=("ownkg",), default="ownkg")
    parser.add_argument(
        "--stage",
        choices=("preflight", "freeze", "run", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--model", choices=candidate_runner.CANDIDATE_MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu; default uses base config")
    args = parser.parse_args()

    if args.stage == "preflight":
        result = candidate_runner.preflight(ROOT, args.dataset)
    elif args.stage == "freeze":
        if args.model is not None or args.seed is not None:
            parser.error("--model/--seed are not accepted for freeze")
        result = candidate_runner.freeze_protocol(ROOT, args.dataset)
    elif args.stage == "run":
        result = candidate_runner.run_sensitivity(
            ROOT,
            args.dataset,
            model_name=args.model,
            seed=args.seed,
            device=args.device,
        )
    elif args.stage == "summarize":
        if args.model is not None or args.seed is not None:
            parser.error("--model/--seed are not accepted for summarize")
        result = candidate_runner.summarize(ROOT, args.dataset)
    else:
        if args.model is not None or args.seed is not None:
            parser.error("--model/--seed are not accepted for all")
        result = candidate_runner.run_all(ROOT, args.dataset, device=args.device)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
