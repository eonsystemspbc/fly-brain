# Nature 2026-07 Publication Dataset

This is the consolidated official dataset for the Nature paper. It replaces the
previous split May and July result folders.

## Spike Outputs

- Run label: `nature_2026_07`
- Frameworks: Brian2 CPU, Brian2CUDA, PyTorch CUDA, NEST GPU, GeNN, and Brian2GeNN
- Grid per round: `t_run = [0.1, 1, 10, 100]` seconds and `n_run = [1, 4, 8, 16, 32]`
- Rounds: 5
- Files: 600 parquet files, with 120 files in each `round_XX` directory
- Manifest: `manifest.csv`
- Integrity hashes: `checksums.sha256`

The PyTorch and NEST GPU files are the corrected post-parity-fix runs. No
pre-fix PyTorch or NEST outputs are included.

## No-I/O Timings

The `no_io/` directory contains the complete one-round no-I/O timing grid for
all six frameworks under label `nature_2026_07_noio`. Spike probing, retrieval,
and parquet output were disabled, so no spike files or spike manifest exist for
that condition.

## Other Material

- `analysis/profiling/` contains the retained PyTorch Nsight Compute profile.
- `logs/` contains valid post-fix PyTorch and NEST benchmark logs retained for provenance.
- `../archive/` contains validation evidence and is not part of the publication bundle upload.

Verify the spike bundle from this directory with:

```bash
sha256sum -c checksums.sha256
```
