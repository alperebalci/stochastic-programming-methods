from sddp_storage import run_benchmark

result, _, _ = run_benchmark(iterations=50, validation_replications=1000)
for key, value in result.to_dict().items():
    print(f"{key}: {value}")
