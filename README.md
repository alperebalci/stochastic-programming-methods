# Stochastic Programming Methods

<!-- portfolio-umbrella:start -->
## Portfolio role

This repository is the primary umbrella repository for this Jors Academy research area. Related projects have been consolidated under `projects/` so the methods, implementations, experiments, and case studies can be maintained and explored from one place.

### Included projects

- [`sddp-multistage-energy-storage`](projects/sddp-multistage-energy-storage/)
- [`stochastic-project-selection-optimization-pulp`](projects/stochastic-project-selection-optimization-pulp/)

Each consolidated project keeps its own files and a `SOURCE_REPOSITORY.md` provenance record. The snapshot preserves the source repository's default-branch files at consolidation time; repository-level history and metadata remain separate from the snapshot.
<!-- portfolio-umbrella:end -->

A reproducible three-stage stochastic production-planning example built with **Pyomo**, **mpi-sppy 0.14.0**, and the open-source **HiGHS** solver.

## Problem

The model plans capacity and production under two sequential demand uncertainties.

- **Stage 1:** choose capacity expansion before demand is known.
- **Stage 2:** after the first demand realization, choose production. Scenarios that share the same first-stage branch must share this decision.
- **Stage 3:** after the second demand realization, choose final recourse production.

The four equiprobable scenarios are `Scen_LL`, `Scen_LH`, `Scen_HL`, and `Scen_HH`.

## mpi-sppy structure

Each scenario is a Pyomo `ConcreteModel` with:

- `_mpisppy_probability = 0.25`
- a `ROOT` `ScenarioNode` for the stage-1 capacity decision
- a `ROOT_0` or `ROOT_1` node for the stage-2 production decision
- stage-specific cost expressions

The demonstration builds the deterministic equivalent with `mpisppy.utils.sputils.create_EF`. This path does not require MPI, which keeps local use and CI lightweight. The same scenario-tree metadata is suitable for later PH / hub-and-spoke experiments when MPI is available.

## Install

```bash
python -m pip install -e '.[dev]'
```

## Run

```bash
python -m stoch_planning.cli
# or
multistage-planning-demo
```

## Test

```bash
pytest
```

GitHub Actions runs the suite on Python 3.10, 3.11, 3.12, and 3.13 with a 90% coverage gate.
