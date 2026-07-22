# Nature 2026-07 No-I/O Benchmark

This directory contains the official no-I/O timing data for all six frameworks.
The full Nature grid was run once with spike probing, retrieval, and parquet
output disabled via `--disable-spike-io`.

- Run label: `nature_2026_07_noio`
- Successful rows: 120 / 120
- Coverage: 20 parameter combinations for each of 6 frameworks
- Spike paths: 0, by design
- Baseline: median of the five matching `nature_2026_07` normal rounds
- Sign convention: a positive reduction means no-I/O was faster

| Framework | Median simulation reduction | Median total reduction | Simulation faster points | Total faster points |
|---|---:|---:|---:|---:|
| Brian2 CPU | 4.1% | 7.1% | 13/20 | 16/20 |
| Brian2CUDA | 11.2% | 13.5% | 19/20 | 20/20 |
| PyTorch CUDA | 9.6% | 7.9% | 20/20 | 20/20 |
| NEST GPU | -12.3% | 4.4% | 7/20 | 14/20 |
| GeNN | 9.3% | 14.2% | 16/20 | 19/20 |
| Brian2GeNN | 37.4% | 2.4% | 20/20 | 13/20 |

NEST GPU did not show a uniform simulation-only gain, although total wall time
improved for most points and by 12.6-15.6% at `n_run = 32`. PyTorch improved at
all points, but its benefit decreased with batching and was below 1% for the
largest `n_run = 32`, `t_run = 100` case.

Files:

- `timings.csv`: the 120 raw successful benchmark rows
- `timing_summary.csv`: per-grid comparison with normal medians and ranges
- `framework_summary.csv`: six-framework aggregate statistics
- `logs/`: retained valid run logs where useful for provenance
