"""
Benchmark orchestrator for the Drosophila brain model.

Manages shared configuration, logging, CSV result persistence, and dispatches
to framework-specific runners:
  - run_brian2_cuda.py  (Brian2 C++ standalone / Brian2CUDA)
  - run_pytorch.py      (PyTorch)
  - run_nestgpu.py      (NEST GPU)

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
N_RUN_VALUES = [1, 30]

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
    'rts_ratio', 'spikes', 'active_neurons', 'status', 'timestamp', 
]


def save_result_csv(backend_name, result):
    """Append or update a benchmark result row in the CSV file.

    Uses (framework, n_run, t_run) as the composite key.  If a row with the
    same key already exists it is replaced; otherwise a new row is appended.
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
    }

    key = (row['framework'], str(row['n_run']), str(row['t_run']))

    existing_rows = []
    if csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)

    updated = False
    for i, r in enumerate(existing_rows):
        existing_key = (r.get('framework', ''),
                        str(r.get('n_run', '')),
                        str(r.get('t_run', '')))
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
}


def run_benchmarks(backends, t_run_values=None, n_run_values=None,
                   experiment=None, logger=None):
    """
    Run benchmarks for the specified backends.

    Args:
        backends: list of backend keys ('cpu', 'gpu', 'pytorch', 'nestgpu')
        t_run_values: list of t_run durations in seconds, or None for all
        n_run_values: list of n_run values, or None for N_RUN_VALUES
        experiment: experiment config dict from get_experiment()
        logger: BenchmarkLogger instance

    Returns:
        dict mapping backend key to list of result dicts
    """
    if experiment is None:
        experiment = get_experiment()

    all_results = {}
    total_backends = len(backends)

    logger.log(f"Experiment: {experiment['name']}")
    logger.log(f"Stimulated neurons: {len(experiment['neu_exc'])} "
               f"at {experiment['stim_rate']} Hz")

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
            )
            all_results[backend] = results

        elif backend == 'pytorch':
            from run_pytorch import run_all_benchmarks as run_torch
            results = run_torch(
                t_run_values=t_run_values,
                n_run_values=n_run_values,
                experiment=experiment,
                logger=logger,
            )
            all_results[backend] = results

        elif backend == 'nestgpu':
            from run_nestgpu import run_all_benchmarks as run_nest
            results = run_nest(
                t_run_values=t_run_values,
                n_run_values=n_run_values,
                experiment=experiment,
                logger=logger,
            )
            all_results[backend] = results

        logger.log(
            f"<<< Finished backend {bi}/{total_backends}: "
            f"{BACKEND_NAMES[backend]}"
        )

    logger.log_raw("")
    logger.log(f"All {total_backends} backend(s) complete.")
    if csv_path.exists():
        logger.log(f"Results CSV: {csv_path}")

    return all_results
