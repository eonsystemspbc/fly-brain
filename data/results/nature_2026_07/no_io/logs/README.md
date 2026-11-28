# No-I/O Logs

These logs are retained for provenance. The authoritative complete timing data
for all six frameworks are the CSV files in the parent directory.

`round_02.log` through `round_05.log` cover all six frameworks. In round 4,
a Brian2 CPU worker was killed during the 100-second, 32-trial case. The
runner repeated the full case with the same settings. The round log records
both attempts, and the timing CSV contains the successful retry.
