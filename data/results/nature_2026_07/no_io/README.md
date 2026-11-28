# Nature 2026-07 No-I/O Benchmark

This directory contains the official no-I/O timing data for all six frameworks.
The full Nature grid was run five times with spike probing, retrieval, and parquet
output disabled via `--disable-spike-io`.

- Run label: `nature_2026_07_noio`
- Successful rows: 600 / 600
- Coverage: 20 parameter combinations per framework per round, across 6 frameworks
- Spike paths: 0, by design
- Baseline: median of the five matching `nature_2026_07` normal rounds
- No-I/O comparison: median of the five matching no-I/O rounds at each grid point
- Sign convention: a positive reduction means no-I/O was faster

| Framework | Median simulation reduction | Median total reduction | Simulation faster points | Total faster points |
|---|---:|---:|---:|---:|
| Brian2 CPU | 6.3% | 7.9% | 17/20 | 17/20 |
| Brian2CUDA | 15.6% | 13.3% | 12/20 | 14/20 |
| PyTorch CUDA | 14.1% | 12.8% | 20/20 | 20/20 |
| NEST GPU | 14.1% | 20.9% | 20/20 | 20/20 |
| GeNN | 8.0% | 11.1% | 14/20 | 18/20 |
| Brian2GeNN | 37.8% | 9.8% | 20/20 | 17/20 |

The recording-enabled rounds were collected from May through July 2026. No-I/O
round 1 was collected in June and July, and rounds 2 through 5 were collected
on 23 to 27 November on the same workstation. Differences between sessions can
therefore contribute to the observed timing differences.

For GeNN at `t_run = 0.1` s and `n_run = 1`, median simulation time increased
from 0.052 s with recording to 0.892 s without it. Median total time for that
case increased from 15.545 s to 16.753 s. All five measurements per condition
are included.

Files:

- `timings.csv`: the 600 raw successful benchmark rows
- `timing_summary.csv`: per-grid comparison with normal medians and ranges
- `framework_summary.csv`: six-framework aggregate statistics
- `logs/`: retained valid run logs where useful for provenance
