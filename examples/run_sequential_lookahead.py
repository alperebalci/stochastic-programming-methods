from stoch_planning.sequential import StochasticProductionLookaheadPolicy


if __name__ == "__main__":
    policy = StochasticProductionLookaheadPolicy()
    for first_demand, second_demand in ((4.0, 5.0), (4.0, 9.0), (8.0, 5.0), (8.0, 9.0)):
        result = policy.simulate(first_demand, second_demand)
        print(
            f"path=({first_demand:.0f}, {second_demand:.0f}) "
            f"capacity_expansion={result.capacity_expansion:.3f} "
            f"production=({result.stage2_production:.3f}, {result.stage3_production:.3f}) "
            f"final_inventory={result.final_inventory:.3f} "
            f"realized_cost={result.realized_cost:.3f}"
        )
