from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and verify an SDDP energy-storage policy.")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--validation-replications", type=int, default=2000)
    parser.add_argument("--training-seed", type=int, default=2026)
    parser.add_argument("--validation-seed", type=int, default=100000)
    args = parser.parse_args()
    result, _, _ = run_benchmark(
        iterations=args.iterations,
        validation_replications=args.validation_replications,
        training_seed=args.training_seed,
        validation_seed=args.validation_seed,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
