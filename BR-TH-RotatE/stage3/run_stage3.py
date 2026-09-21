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

from stage3 import baseline_runner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage-3 external baselines: RatE + CompoundE + capacity-matched TH-RotatE"
    )
    parser.add_argument("--dataset", choices=("ownkg", "paper4"), default="ownkg")
    parser.add_argument(
        "--stage",
        choices=("preflight", "screen", "freeze", "test", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--model", choices=baseline_runner.BASELINE_MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu; default uses base config")
    args = parser.parse_args()

    if args.stage == "preflight":
        result = baseline_runner.preflight(ROOT, args.dataset)
    elif args.stage == "screen":
        if args.seed is not None:
            parser.error("--seed is not used by Validation screening (the frozen validation seed is used)")
        result = baseline_runner.validation_screen(
            ROOT, args.dataset, model_name=args.model, device=args.device
        )
    elif args.stage == "freeze":
        if args.model is not None or args.seed is not None:
            parser.error("--model/--seed are not accepted for freeze")
        result = baseline_runner.freeze_protocol(ROOT, args.dataset)
    elif args.stage == "test":
        result = baseline_runner.fixed_test(
            ROOT,
            args.dataset,
            model_name=args.model,
            seed=args.seed,
            device=args.device,
        )
    elif args.stage == "summarize":
        if args.model is not None or args.seed is not None:
            parser.error("--model/--seed are not accepted for summarize")
        result = baseline_runner.summarize(ROOT, args.dataset)
    else:
        if args.model is not None or args.seed is not None:
            parser.error("--model/--seed are not accepted for all")
        result = baseline_runner.run_all(ROOT, args.dataset, device=args.device)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
