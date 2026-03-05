"""
Drosophila brain model benchmark runner.

Usage:
    # All backends, default durations and trials
    python main.py

    # Specific durations and trial count
    python main.py --t_run 0.1 1 10 --n_run 1

    # Specific backends (combinable)
    python main.py --brian2-cpu                          # Brian2 CPU only
    python main.py --brian2cuda-gpu                      # Brian2CUDA GPU only
    python main.py --pytorch                             # PyTorch only
    python main.py --nestgpu                             # NEST GPU only
    python main.py --brian2-cpu --pytorch --nestgpu      # Brian2 CPU + PyTorch + NEST GPU

    # Background with log
    nohup python main.py > data/results/benchmarks.log 2>&1 &
"""

import os
os.environ['PYTHONUNBUFFERED'] = '1'

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))

from benchmark import (
    BenchmarkLogger, T_RUN_VALUES_SEC, BACKEND_NAMES, run_benchmarks,
)


def main():
    parser = argparse.ArgumentParser(description='Drosophila brain model benchmark')
    parser.add_argument('--t_run', type=float, nargs='+', default=None,
                        help='Simulation duration(s) in seconds. '
                             f'Allowed: {T_RUN_VALUES_SEC}. Default: all')
    parser.add_argument('--n_run', type=int, nargs='+', default=None,
                        help='Number of trials. Default: [1, 30]')
    parser.add_argument('--log_file', type=str, default='data/results/benchmarks.log',
                        help='Log file path. Default: data/results/benchmarks.log')
    parser.add_argument('--no_log_file', action='store_true',
                        help='Disable file logging (console only)')

    parser.add_argument('--brian2-cpu', action='store_true',
                        help='Run Brian2 C++ standalone (CPU)')
    parser.add_argument('--brian2cuda-gpu', action='store_true',
                        help='Run Brian2CUDA (GPU)')
    parser.add_argument('--pytorch', action='store_true',
                        help='Run PyTorch benchmark')
    parser.add_argument('--nestgpu', action='store_true',
                        help='Run NEST GPU benchmark')

    args = parser.parse_args()

    # If no backend flags specified, run all
    if not (args.brian2_cpu or args.brian2cuda_gpu or args.pytorch or args.nestgpu):
        backends = ['cpu', 'gpu', 'pytorch', 'nestgpu']
    else:
        backends = []
        if args.brian2_cpu:
            backends.append('cpu')
        if args.brian2cuda_gpu:
            backends.append('gpu')
        if args.pytorch:
            backends.append('pytorch')
        if args.nestgpu:
            backends.append('nestgpu')

    # Import brian2cuda once to register the cuda_standalone device
    if 'gpu' in backends:
        import brian2cuda  # noqa: F401

    # Validate t_run values
    t_run_values = args.t_run
    if t_run_values:
        for val in t_run_values:
            if val not in T_RUN_VALUES_SEC:
                print(
                    f"Error: --t_run {val} is not in allowed values: "
                    f"{T_RUN_VALUES_SEC}"
                )
                return

    log_file = None if args.no_log_file else args.log_file
    logger = BenchmarkLogger(log_file=log_file)

    try:
        logger.log_raw("")
        logger.log("Starting benchmark suite")
        logger.log(
            f"Backends: {', '.join(BACKEND_NAMES[b] for b in backends)}"
        )
        t_run_display = t_run_values if t_run_values else T_RUN_VALUES_SEC
        logger.log(f"t_run values: {t_run_display}s")
        logger.log(f"n_run values: {args.n_run or [1, 30]}")
        logger.log(f"Log file: {log_file if log_file else 'disabled'}")

        run_benchmarks(
            backends=backends,
            t_run_values=t_run_values,
            n_run_values=args.n_run,
            logger=logger,
        )

    finally:
        logger.close()


if __name__ == '__main__':
    main()
