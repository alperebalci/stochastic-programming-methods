# Stochastic Dual Dynamic Programming for Multistage Energy Storage

A transparent Python implementation of **stochastic dual dynamic programming (SDDP)** for a finite-horizon energy-storage problem with stagewise-independent discrete electricity prices.

The repository is designed to expose the mechanics that are hidden inside production SDDP libraries: state propagation, forward sampling, backward dual cuts, lower bounds, policy simulation, and exact small-instance verification against the full scenario-tree deterministic equivalent.

It is an educational research implementation, not a replacement for SDDP.jl or an industrial power-system planning package.

## Why this is genuinely multistage

The battery is operated over six sequential stages. At each stage the current price is observed before the charge/discharge decision is made. The decision changes the energy state passed to the next stage, and future prices are unknown when the current decision is chosen.

The policy is therefore adaptive:

```text
initial energy
     |
observe price_0 -> choose charge/discharge -> energy_1
                                      |
                              observe price_1
                                      |
                              choose decision
                                      |
                                     ...
```

A two-stage stochastic program cannot represent this repeated observe-decide-transition structure without collapsing the information pattern.

## Storage model

For stage `t`, incoming energy `e_t`, observed price `p_t`, charge `c_t`, discharge `d_t`, and outgoing energy `e_{t+1}`:

```text
minimize    p_t c_t - p_t d_t + future_cost(e_{t+1})

subject to  e_{t+1} - eta_c c_t + d_t / eta_d = e_t
            0 <= e_{t+1} <= E_max
            0 <= c_t <= C_max
            0 <= d_t <= D_max
```

At the final stage, the future-cost term is replaced by linear terminal salvage:

```text
-salvage_price * e_T
```

All prices in the teaching instance are positive. With round-trip losses, simultaneous charging and discharging is therefore dominated without introducing binary mode variables. The resulting model remains a convex linear multistage stochastic program.

The default instance has:

```text
horizon                 6 stages
price outcomes          3 per stage
storage capacity        10
initial energy           5
charge limit             4 per stage
discharge limit          4 per stage
charge efficiency        0.95
discharge efficiency     0.90
terminal salvage price   35
```

Price distributions vary by stage but are independent across stages. This is a finite, stagewise-independent uncertainty model.

## SDDP approximation

Let `V_t(e)` denote expected optimal cost from stage `t` onward before the stage-`t` price is observed. SDDP approximates this convex value function from below by affine cuts:

```text
V_t(e) >= alpha_k + beta_k e
```

The stage LP contains a future-cost variable `theta` and the accumulated cuts:

```text
theta >= alpha_k + beta_k e_{t+1}    for every stored cut k
```

The approximation is piecewise linear and becomes tighter as cuts are added.

### Safe initialization

A nonterminal stage needs at least one lower bound on future cost so that its LP is bounded. The implementation starts from a deliberately conservative constant cut. It assumes, unrealistically favorably, that the battery could earn the maximum stage price times the discharge limit at every remaining stage and also receive the maximum possible terminal salvage. The negative of that maximum possible revenue is therefore a valid lower bound.

This initialization is loose by design but mathematically safe.

## Forward pass

Each SDDP iteration samples one complete price path. The current piecewise-linear future-value approximations are used to make a decision at each stage, and the resulting incoming states are stored:

```text
sample price path
      |
      v
solve stage 0 with current V_1 approximation
      |
      v
store e_1
      |
      v
solve stage 1 with current V_2 approximation
      |
     ...
```

The sampled trajectory is not itself an optimality certificate. Its important role during training is to identify states at which the value functions should be refined.

## Backward pass and dual cuts

The backward pass visits the sampled states in reverse stage order. At a trial incoming state `e_bar`, the implementation solves the stage LP for **every price outcome** in that stage.

The state transition is written with `e_bar` on the right-hand side:

```text
e_next - eta_c * charge + discharge / eta_d = e_bar
```

SciPy/HiGHS returns the equality-constraint marginal. For this LP, that dual marginal is a subgradient of the stage value with respect to incoming energy.

For price outcome `w`, let:

```text
Q_w(e_bar) = stage value
beta_w     = RHS marginal / subgradient
```

With probabilities `p_w`, the single expected-value cut is:

```text
beta = sum_w p_w beta_w
value = sum_w p_w Q_w(e_bar)
alpha = value - beta * e_bar

V_t(e) >= alpha + beta e
```

The implementation uses all discrete outcomes in the backward pass, not backward sampling. This keeps the code and verification logic simple.

## Lower and upper information

Because the stored cuts are lower approximations of convex cost-to-go functions, solving the initial-stage problem with those cuts produces an SDDP **lower bound** on the optimal expected cost.

A policy induced by the cuts is feasible. Therefore its true expected cost is an upper bound on the optimal expected cost. The repository evaluates that policy in two ways:

1. exact enumeration of the finite price tree, yielding the exact expectation of the induced policy for this small benchmark;
2. independent Monte Carlo simulation, yielding a sample mean, standard error, and Student-`t` 95% confidence interval.

The Monte Carlo sample mean may numerically fall below the true optimum because it is an estimator. It should not be interpreted as a deterministic upper bound by itself.

## Exact deterministic-equivalent benchmark

For verification only, `solve_extensive_form` constructs the complete scenario tree. Each tree node has its own charge, discharge, and outgoing-energy variables. Parent-child state equations enforce nonanticipativity implicitly through the tree structure.

For the default six-stage, three-outcome instance, the deterministic equivalent contains:

```text
scenario-tree nodes   1*3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 = 1092
continuous variables  3 * 1092 = 3276
```

SDDP does not build this tree during training. The full tree exists here only to provide an exact benchmark on a deliberately small instance.

## Reproducible development result

The following run uses 50 SDDP iterations and a separate 2,000-path Monte Carlo validation sample:

```bash
python -m sddp_storage \
  --iterations 50 \
  --validation-replications 2000 \
  --training-seed 2026 \
  --validation-seed 100000
```

Result:

```text
exact extensive-form optimum      -418.2644163713
final SDDP lower bound             -418.2644163713
lower-bound gap                       ~4e-13
exact expectation of SDDP policy   -418.2644163713
exact policy optimality gap           ~5e-13 in magnitude

Monte Carlo policy mean            -419.9398
Monte Carlo standard error            2.3384
95% confidence interval        [-424.5258, -415.3538]
```

The seeded lower-bound progression was:

```text
iteration    lower bound
1            -755.1594
5            -427.3100
10           -423.4408
20           -419.8382
30           -418.3501
40           -418.2920
50           -418.2644
```

On this small finite-support linear instance, the generated cuts recover the extensive-form optimum to numerical precision after 50 seeded iterations. This is a property of this benchmark run, **not a claim that SDDP always reaches exact equality after a fixed number of iterations**.

The independent Monte Carlo mean happens to lie below the exact optimum in this run. That is ordinary sampling error: the exact policy-tree evaluation confirms that the policy itself has the same expected value as the optimum to numerical precision.

## Verification strategy

The test suite checks more than code execution:

- storage-state balance and bounds;
- the HiGHS equality marginal against a local finite-difference derivative;
- a hand-checkable one-stage extensive-form optimum;
- exact scenario-tree node counts;
- monotonic nondecrease of the SDDP lower bound;
- lower bounds never exceeding the exact extensive-form optimum;
- generated cuts against exact continuation values on an energy grid;
- exact expectation of the induced policy as a valid primal upper bound;
- reproducible training under a fixed seed;
- benchmark confidence-interval accounting;
- CLI JSON output.

These tests are particularly important because a sign error in a dual marginal can create an invalid cut that looks numerically plausible while destroying the lower-bound interpretation.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.11+ is required.

## Run

```bash
python -m sddp_storage
```

The command prints JSON containing the extensive-form optimum, final lower bound, exact policy expectation, Monte Carlo confidence interval, cut count, and scenario-tree dimensions.

A shorter demonstration is available at:

```bash
python examples/run_demo.py
```

## Tests

```bash
pytest -q
```

GitHub Actions runs package installation, compilation, Ruff, the full regression suite, and an end-to-end SDDP smoke experiment on Python 3.11 and 3.12.

## Repository structure

```text
.
├── .github/workflows/ci.yml
├── examples/run_demo.py
├── src/sddp_storage/
│   ├── __init__.py
│   ├── __main__.py
│   ├── benchmark.py
│   ├── extensive.py
│   ├── model.py
│   └── sddp.py
├── tests/
│   ├── test_benchmark_cli.py
│   ├── test_extensive.py
│   ├── test_model.py
│   ├── test_sddp.py
│   └── test_stage.py
├── LICENSE
├── README.md
└── pyproject.toml
```

## Methodological boundaries

This repository intentionally does **not** claim to implement the full generality of modern SDDP packages. In particular, it does not include Markovian uncertainty, autoregressive processes, multicut variants, risk-averse measures, integer state/control variables, regularization, cut selection/deletion, parallel backward passes, backward sampling, nonlinear stage models, or infinite-horizon policy graphs.

The teaching model satisfies the important classical structure: finite horizon, convex linear stage problems, finite discrete uncertainty, stagewise independence, and relatively complete recourse. The last property is immediate here because doing nothing (`charge = discharge = 0`) is feasible from every storage state in `[0, capacity]`.

A larger research extension would compare single-cut and multicut SDDP, introduce Markovian prices, use risk measures such as CVaR, add cut-management strategies, and benchmark against SDDP.jl on the same data.

## Research grounding

- M. V. F. Pereira and L. M. V. G. Pinto, **Multi-stage stochastic optimization applied to energy planning**, *Mathematical Programming* 52, 359-375 (1991). DOI: `10.1007/BF01582895`.
- A. Shapiro, **Analysis of stochastic dual dynamic programming method**, *European Journal of Operational Research* 209(1), 63-72 (2011). DOI: `10.1016/j.ejor.2010.08.007`.
- SDDP.jl documentation, **An introduction to SDDP.jl** and **Introductory theory**, for modern policy-graph terminology and the forward/backward cutting-plane interpretation.

Pereira and Pinto introduced the energy-planning methodology based on piecewise-linear approximations to expected cost-to-go functions. Shapiro analyzes SDDP for multistage linear stochastic programs under stagewise-independent data. Modern SDDP.jl documentation emphasizes finite discrete uncertainty, convex stage problems, state transitions, and relatively complete recourse as core assumptions for the classical framework.

## License

This repository is licensed under the **JORS Academy Non-Commercial Source License 1.0**. Commercial use is prohibited without a separate prior written commercial license. See [`LICENSE`](LICENSE) for the complete terms.
