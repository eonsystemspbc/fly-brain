"""
Benchmark orchestrator for the Drosophila brain model.

Manages shared configuration, logging, CSV result persistence, and dispatches
to framework-specific runners:
  - run_brian2_cuda.py  (Brian2 C++ standalone / Brian2CUDA)
  - run_pytorch.py      (PyTorch)
  - run_nestgpu.py      (NEST GPU)
  - run_genn.py         (GeNN)
  - run_brian2_genn.py  (Brian2GeNN)

Entrypoint is in main.py at the project root.
"""

import os
import csv
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

os.environ['PYTHONUNBUFFERED'] = '1'

from pathlib import Path
from datetime import datetime

# ============================================================================
# Benchmark Configuration
# ============================================================================

T_RUN_VALUES_SEC = [0.1, 1, 10, 100, 1000]
PAPER_T_RUN_VALUES_SEC = [0.1, 1.0, 10.0, 100.0]
N_RUN_VALUES = [1, 4, 8, 16, 32]
PAPER_ROUNDS = 5

# ============================================================================
# Paths and Constants
# ============================================================================
current_dir = Path(__file__).resolve().parent
output_dir = current_dir / 'output'
path_comp = (current_dir / '../data/2025_Completeness_783.csv').resolve()
path_con = (current_dir / '../data/2025_Connectivity_783.parquet').resolve()
path_res = (current_dir / '../data/results').resolve()
path_wt = (current_dir / '../data').resolve()
csv_path = (current_dir / '../data/benchmark-results.csv').resolve()
repo_dir = current_dir.parent.resolve()

# ============================================================================
# Experiment Definitions
# ============================================================================

EXPERIMENTS = {
    'sugar': {
        'key': 'sugar',
        'name': 'Sugar GRNs (200 Hz)',
        'neu_exc': [
            720575940624963786,
            720575940630233916,
            720575940637568838,
            720575940638202345,
            720575940617000768,
            720575940630797113,
            720575940632889389,
            720575940621754367,
            720575940621502051,
            720575940640649691,
            720575940639332736,
            720575940616885538,
            720575940639198653,
            720575940639259967,
            720575940617937543,
            720575940632425919,
            720575940633143833,
            720575940612670570,
            720575940628853239,
            720575940629176663,
            720575940611875570,
        ],
        'neu_exc2': [],
        'neu_slnc': [],
        'stim_rate': 200.0,
    },
    'p9': {
        'key': 'p9',
        'name': 'P9s forward walking (100 Hz)',
        'neu_exc': [
            720575940627652358,  # P9 left
            720575940635872101,  # P9 right
        ],
        'neu_exc2': [],
        'neu_slnc': [],
        'stim_rate': 100.0,
    },
}

DEFAULT_EXPERIMENT = 'sugar'


def get_experiment(name=None):
    """Return experiment config dict by name (default: sugar)."""
    name = name or DEFAULT_EXPERIMENT
    if name not in EXPERIMENTS:
        raise ValueError(
            f"Unknown experiment '{name}'. "
            f"Available: {list(EXPERIMENTS.keys())}"
        )
    return EXPERIMENTS[name]

# ============================================================================
# Logging Utilities
# ============================================================================

class BenchmarkLogger:
    """Logger that writes to both console and file."""

    def __init__(self, log_file=None):
        self.log_file = log_file
        self.file_handle = None
        if log_file:
            self.file_handle = open(log_file, 'a')

    def log(self, message, end='\n'):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted = f"[{timestamp}] {message}"
        print(formatted, end=end, flush=True)
        if self.file_handle:
            self.file_handle.write(formatted + end)
            self.file_handle.flush()

    def log_raw(self, message, end='\n'):
        """Log without timestamp."""
        print(message, end=end, flush=True)
        if self.file_handle:
            self.file_handle.write(message + end)
            self.file_handle.flush()

    def close(self):
        if self.file_handle:
            self.file_handle.close()

# ============================================================================
# CSV Result Persistence
# ============================================================================

CSV_COLUMNS = [
    'framework', 'n_run', 't_run',
    'setup_time', 'build_time', 'sim_time', 'total_time',
    'realtime_ratio', 'spikes', 'active_neurons', 'status', 'timestamp',
    'backend_key', 'experiment', 'experiment_key', 'run_label', 'round',
    'spike_path',
]

MANIFEST_COLUMNS = [
    'run_label', 'round', 'framework', 'backend_key',
    'experiment', 'experiment_key', 'n_run', 't_run',
    'spike_path', 'spikes', 'active_neurons', 'status', 'timestamp',
    'spike_schema_version', 'spike_time_column', 'spike_time_unit',
]


def sanitize_run_label(run_label):
    """Return a filesystem-safe run label while keeping it human-readable."""
    if not run_label:
        return None
    safe = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in run_label)
    safe = safe.strip('_')
    return safe or None


def default_run_label():
    """Create a stable label for a multi-round paper benchmark invocation."""
    return datetime.now().strftime('paper_%Y%m%d_%H%M%S')


def get_spike_output_dir(run_label=None, round_idx=None):
    """Directory for spike parquet outputs for a benchmark invocation."""
    output = Path(path_res)
    run_label = sanitize_run_label(run_label)
    if run_label:
        output = output / run_label
    if round_idx is not None:
        output = output / f'round_{int(round_idx):02d}'
    return output


def get_spike_output_path(exp_name, run_label=None, round_idx=None):
    """Path for a benchmark's per-spike parquet file."""
    return get_spike_output_dir(run_label, round_idx) / f'{exp_name}.parquet'


def relative_to_repo(path):
    """Render paths relative to the repo when possible for portable CSVs."""
    if not path:
        return ''
    p = Path(path)
    try:
        return str(p.resolve().relative_to(repo_dir)).replace('\\', '/')
    except ValueError:
        return str(p)


def _csv_key(row):
    """Composite key for CSV updates.

    Legacy rows did not have run metadata, so unlabeled one-off benchmarks keep
    the old overwrite behavior. Labeled paper rounds are keyed by their run
    label and round number so repeated parameter sets are preserved.
    """
    base = (
        row.get('framework', ''),
        str(row.get('n_run', '')),
        str(row.get('t_run', '')),
    )
    if row.get('run_label') or row.get('round'):
        return base + (
            row.get('experiment_key', ''),
            row.get('run_label', ''),
            str(row.get('round', '')),
        )
    return base


def _save_manifest_csv(result, row):
    """Write/update a compact manifest for round-specific spike outputs."""
    run_label = row.get('run_label', '')
    if not run_label or not row.get('spike_path'):
        return

    manifest_path = path_res / run_label / 'manifest.csv'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_row = {
        'run_label': row.get('run_label', ''),
        'round': row.get('round', ''),
        'framework': row.get('framework', ''),
        'backend_key': row.get('backend_key', ''),
        'experiment': row.get('experiment', ''),
        'experiment_key': row.get('experiment_key', ''),
        'n_run': row.get('n_run', ''),
        't_run': row.get('t_run', ''),
        'spike_path': row.get('spike_path', ''),
        'spikes': row.get('spikes', ''),
        'active_neurons': row.get('active_neurons', ''),
        'status': row.get('status', ''),
        'timestamp': row.get('timestamp', ''),
        'spike_schema_version': '1',
        'spike_time_column': 'time_ms',
        'spike_time_unit': 'ms',
    }

    existing_rows = []
    if manifest_path.exists():
        with open(manifest_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for existing in reader:
                existing_rows.append(existing)

    def manifest_key(values):
        return (
            str(values.get('run_label', '')),
            str(values.get('framework', '')),
            str(values.get('backend_key', '')),
            str(values.get('experiment_key', '')),
            str(values.get('n_run', '')),
            str(values.get('t_run', '')),
            str(values.get('round', '')),
        )

    key = manifest_key(manifest_row)
    updated = False
    for i, existing in enumerate(existing_rows):
        existing_key = manifest_key(existing)
        if existing_key == key:
            existing_rows[i] = manifest_row
            updated = True
            break

    if not updated:
        existing_rows.append(manifest_row)

    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)


def save_result_csv(backend_name, result):
    """Append or update a benchmark result row in the CSV file.

    Unlabeled one-off runs keep the legacy (framework, n_run, t_run) key.
    Labeled repeated runs also key by experiment, run_label, and round so
    paper benchmark replicates are preserved instead of overwriting each other.
    """
    path_res.mkdir(parents=True, exist_ok=True)

    t = result.get('timings', {})

    row = {
        'framework': backend_name,
        'n_run': result['n_run'],
        't_run': result['t_run_sec'],
        'setup_time': round(t.get('network_creation_total',
                                  t.get('model_setup_total', 0)), 3),
        'build_time': round(t.get('device_build', 0), 3),
        'sim_time': round(t.get('simulation_total', 0), 3),
        'total_time': round(t.get('total_elapsed', 0), 3),
        'realtime_ratio': round(t.get('realtime_ratio', 0), 4),
        'spikes': result.get('n_spikes', 0),
        'active_neurons': result.get('n_active_neurons', 0),
        'status': result.get('status', 'unknown'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'backend_key': result.get('backend_key', ''),
        'experiment': result.get('experiment_name', ''),
        'experiment_key': result.get('experiment_key', ''),
        'run_label': result.get('run_label', ''),
        'round': result.get('round', ''),
        'spike_path': relative_to_repo(result.get('spike_path')),
    }

    key = _csv_key(row)

    existing_rows = []
    if csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append({col: r.get(col, '') for col in CSV_COLUMNS})

    updated = False
    for i, r in enumerate(existing_rows):
        existing_key = _csv_key(r)
        if existing_key == key:
            existing_rows[i] = {k: str(v) for k, v in row.items()}
            updated = True
            break

    if not updated:
        existing_rows.append({k: str(v) for k, v in row.items()})

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)

    _save_manifest_csv(result, row)


# ============================================================================
# Summary Printing
# ============================================================================

def print_summary_table(all_results, backend_name, logger):
    """Print a formatted summary table for benchmark results."""
    logger.log_raw("")
    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"SUMMARY: {backend_name}")
    logger.log_raw("=" * 80)
    logger.log_raw("")
    logger.log_raw(
        f"{'t_run':>8} | {'n_run':>6} | {'Setup':>10} | "
        f"{'Build':>10} | {'Simulation':>12} | {'Total':>10} | "
        f"{'RT Ratio':>10} | {'Spikes':>10} | Status"
    )
    logger.log_raw("-" * 110)

    for result in all_results:
        t = result.get('timings', {})
        status_icon = "\u2713" if result['status'] == 'success' else "\u2717"

        setup_time = t.get(
            'network_creation_total', t.get('model_setup_total', 0)
        )
        build_time = t.get('device_build', 0)
        sim_time = t.get('simulation_total', 0)
        total_time = t.get('total_elapsed', 0)
        realtime_ratio = t.get('realtime_ratio', 0)

        logger.log_raw(
            f"{result['t_run_sec']:>7.1f}s | "
            f"{result['n_run']:>6d} | "
            f"{setup_time:>9.2f}s | "
            f"{build_time:>9.2f}s | "
            f"{sim_time:>11.2f}s | "
            f"{total_time:>9.2f}s | "
            f"{realtime_ratio:>9.3f}x | "
            f"{result['n_spikes']:>10d} | "
            f"{status_icon} {result['status']}"
        )

    logger.log_raw("-" * 110)
    logger.log_raw("")
    logger.log("Benchmark suite complete!")

# ============================================================================
# Backend Dispatcher
# ============================================================================

BACKEND_NAMES = {
    'cpu': 'Brian2 (CPU)',
    'gpu': 'Brian2CUDA (GPU)',
    'pytorch': 'PyTorch',
    'nestgpu': 'NEST GPU',
    'genn': 'GeNN (GPU)',
    'brian2genn': 'Brian2GeNN (GPU)',
}


def run_benchmarks(backends, t_run_values=None, n_run_values=None,
                   experiment=None, logger=None, rounds=1, run_label=None,
                   round_start=1):
    """
    Run benchmarks for the specified backends.

    Args:
        backends: list of backend keys
            ('cpu', 'gpu', 'pytorch', 'nestgpu', 'genn', 'brian2genn')
        t_run_values: list of t_run durations in seconds, or None for all
        n_run_values: list of n_run values, or None for N_RUN_VALUES
        experiment: experiment config dict from get_experiment()
        logger: BenchmarkLogger instance
        rounds: number of repeated benchmark rounds to run
        run_label: optional label used to group round-specific spike outputs
        round_start: first round number to write, for resuming interrupted runs

    Returns:
        dict mapping backend key to list of result dicts
    """
    if experiment is None:
        experiment = get_experiment()

    all_results = {}
    total_backends = len(backends)
    rounds = int(rounds or 1)
    if rounds < 1:
        raise ValueError('rounds must be >= 1')
    round_start = int(round_start or 1)
    if round_start < 1:
        raise ValueError('round_start must be >= 1')
    run_label = sanitize_run_label(run_label)
    if rounds > 1 and not run_label:
        run_label = default_run_label()

    logger.log(f"Experiment: {experiment['name']}")
    logger.log(f"Stimulated neurons: {len(experiment['neu_exc'])} "
               f"at {experiment['stim_rate']} Hz")
    logger.log(f"Benchmark rounds: {rounds}")
    if run_label:
        logger.log(f"Run label: {run_label}")

    round_end = round_start + rounds - 1
    for round_idx in range(round_start, round_end + 1):
        round_label = round_idx if rounds > 1 or run_label else None
        if rounds > 1:
            logger.log_raw("")
            logger.log_raw("#" * 80)
            logger.log(f"STARTING BENCHMARK ROUND {round_idx}/{round_end}")
            logger.log_raw("#" * 80)

        for bi, backend in enumerate(backends, 1):
            logger.log_raw("")
            logger.log(
                f">>> Starting backend {bi}/{total_backends}: "
                f"{BACKEND_NAMES[backend]}"
            )

            if backend in ('cpu', 'gpu'):
                from run_brian2_cuda import run_all_benchmarks as run_brian2
                results = run_brian2(
                    use_cuda=(backend == 'gpu'),
                    t_run_values=t_run_values,
                    n_run_values=n_run_values,
                    experiment=experiment,
                    logger=logger,
                    run_label=run_label,
                    round_idx=round_label,
                )
                all_results.setdefault(backend, []).extend(results)

            elif backend == 'pytorch':
                from run_pytorch import run_all_benchmarks as run_torch
                results = run_torch(
                    t_run_values=t_run_values,
                    n_run_values=n_run_values,
                    experiment=experiment,
                    logger=logger,
                    run_label=run_label,
                    round_idx=round_label,
                )
                all_results.setdefault(backend, []).extend(results)

            elif backend == 'nestgpu':
                from run_nestgpu import run_all_benchmarks as run_nest
                results = run_nest(
                    t_run_values=t_run_values,
                    n_run_values=n_run_values,
                    experiment=experiment,
                    logger=logger,
                    run_label=run_label,
                    round_idx=round_label,
                )
                all_results.setdefault(backend, []).extend(results)

            elif backend == 'genn':
                from run_genn import run_all_benchmarks as run_genn
                results = run_genn(
                    t_run_values=t_run_values,
                    n_run_values=n_run_values,
                    experiment=experiment,
                    logger=logger,
                    run_label=run_label,
                    round_idx=round_label,
                )
                all_results.setdefault(backend, []).extend(results)

            elif backend == 'brian2genn':
                from run_brian2_genn import run_all_benchmarks as run_brian2genn
                results = run_brian2genn(
                    t_run_values=t_run_values,
                    n_run_values=n_run_values,
                    experiment=experiment,
                    logger=logger,
                    run_label=run_label,
                    round_idx=round_label,
                )
                all_results.setdefault(backend, []).extend(results)

            logger.log(
                f"<<< Finished backend {bi}/{total_backends}: "
                f"{BACKEND_NAMES[backend]}"
            )

    logger.log_raw("")
    logger.log(f"All {total_backends} backend(s) complete.")
    if csv_path.exists():
        logger.log(f"Results CSV: {csv_path}")
    if run_label:
        logger.log(f"Spike manifest: {path_res / run_label / 'manifest.csv'}")

    return all_results
