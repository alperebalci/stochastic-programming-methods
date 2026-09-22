import json
import subprocess
import sys

from sddp_storage import StorageConfig, run_benchmark


def tiny_config() -> StorageConfig:
    return StorageConfig(
        price_supports=((30.0, 60.0), (25.0, 75.0), (35.0, 65.0)),
        probabilities=((0.5, 0.5),) * 3,
        capacity=4.0,
        initial_energy=2.0,
        charge_limit=1.5,
        discharge_limit=1.5,
        salvage_price=30.0,
    )


def test_benchmark_reports_exact_and_statistical_metrics() -> None:
    result, _, _ = run_benchmark(
        tiny_config(),
        iterations=12,
        validation_replications=40,
        training_seed=2,
        validation_seed=300,
    )
    assert result.final_lower_bound <= result.exact_objective + 1e-6
    assert result.exact_policy_value >= result.exact_objective - 1e-6
    assert result.policy_ci95_low <= result.policy_mean <= result.policy_ci95_high


def test_cli_emits_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sddp_storage",
            "--iterations",
            "3",
            "--validation-replications",
            "20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert "exact_objective" in payload
    assert "final_lower_bound" in payload
