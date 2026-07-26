# v0.18_EE Figures

## Canonical cross-version FLOP saving

![Canonical 3-exit FLOP-saving operating points through v0.18](cross_version_3exit_flops_v018.svg)

Discrete corrected-holdout operating points; not a learning curve or continuous Pareto frontier.

## Canonical cross-version quality

![Canonical 3-exit quality operating points through v0.18](cross_version_3exit_quality_v018.svg)

Macro-F1, Micro-F1, and Exact Match for the canonical 3-exit family.

## Fair architecture FLOP comparison

![v0.18 fair architecture FLOP comparison](v018_fair_architecture_flops.svg)

Each policy is compared with its architecture-specific final exit.

## Full-strict exit distribution

![v0.18 full-strict exit distribution](v018_exit_distribution.svg)

The selected fair five-exit holdout policy uses Exits 1, 3, and 5; Exit 2 and Exit 4 have zero coverage.

## Holdout constraint utilisation

![v0.18 quality-constraint utilisation](v018_constraint_utilisation.svg)

100% marks the maximum allowed degradation. Values above 100% fail the corresponding constraint.

## Existing summary figure

![Original v0.18 quality–compute summary](v018_quality_compute_summary.svg)

## Figure cautions

- Lines connect discrete operating points and do not represent training trajectories.
- FLOP estimates are architecture-based, not measured latency.
- Speedups are CPU-, batch-, threading-, and implementation-specific.
- v0.17 non-fair and v0.18 fair five-exit figures answer different questions.
