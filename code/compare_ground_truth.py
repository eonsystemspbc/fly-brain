"""
Compare backend spike outputs against the Brian2 (CPU) ground truth.

Brian2 CPU implements the canonical LIF model from Shiu et al. (Nature 2024)
using the FlyWire connectome.  This script loads per-neuron spike trains saved
by each backend and computes:

  - Active neuron overlap (Jaccard, precision, recall)
  - Per-neuron firing-rate Pearson correlation
  - Spike-count ratio relative to ground truth
  - Top neurons with the largest firing-rate deviations

Usage:
    python code/compare_ground_truth.py                     # defaults
    python code/compare_ground_truth.py --t_run 10          # different duration
    python code/compare_ground_truth.py --n_run 4           # multi-trial avg
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import get_spike_output_path

RESULTS_DIR = Path(__file__).resolve().parent / '../data/results'
OUTPUT_PATH = Path(__file__).resolve().parent / '../data/ground-truth-comparison.json'

BACKENDS = {
    'brian2cpp':  'Brian2 (CPU)',
    'brian2cuda': 'Brian2CUDA (GPU)',
    'pytorch':    'PyTorch (CUDA)',
    'nestgpu':    'NEST GPU',
    'genn':       'GeNN (GPU)',
    'brian2genn': 'Brian2GeNN (GPU)',
}

GROUND_TRUTH_KEY = 'brian2cpp'

# PyTorch saves spike times in ms; all others in seconds
MS_BACKENDS = {'pytorch'}


def load_spike_data(backend_key, t_run, n_run, run_label=None, round_idx=None):
    """Load a backend's parquet and return a DataFrame with times in seconds."""
    if run_label or round_idx is not None:
        path = get_spike_output_path(
            f'{backend_key}_t{t_run}s_n{n_run}', run_label, round_idx,
        )
    else:
        path = RESULTS_DIR / f'{backend_key}_t{t_run}s_n{n_run}.parquet'
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if 'time_ms' in df.columns:
        df['t'] = df['time_ms'] / 1000.0
    elif 'time_s' in df.columns:
        df['t'] = df['time_s']
    elif backend_key in MS_BACKENDS:
        df['t'] = df['t'] / 1000.0
    return df


def firing_rates(df, t_run, n_run):
    """Per-neuron average firing rate (Hz) across all trials."""
    counts = df.groupby('flywire_id').size()
    total_time = t_run * n_run
    return counts / total_time


def compare(gt_df, other_df, t_run, n_run):
    """Compute comparison metrics between ground truth and another backend."""
    gt_rates = firing_rates(gt_df, t_run, n_run)
    other_rates = firing_rates(other_df, t_run, n_run)

    gt_neurons = set(gt_rates.index)
    other_neurons = set(other_rates.index)
    all_neurons = gt_neurons | other_neurons
    shared = gt_neurons & other_neurons
    gt_only = gt_neurons - other_neurons
    other_only = other_neurons - gt_neurons

    jaccard = len(shared) / len(all_neurons) if all_neurons else 0.0
    precision = len(shared) / len(other_neurons) if other_neurons else 0.0
    recall = len(shared) / len(gt_neurons) if gt_neurons else 0.0

    # Firing-rate correlation over shared neurons
    if len(shared) >= 2:
        shared_list = sorted(shared)
        gt_vec = np.array([gt_rates[n] for n in shared_list])
        other_vec = np.array([other_rates[n] for n in shared_list])
        correlation = float(np.corrcoef(gt_vec, other_vec)[0, 1])

        pct_diffs = np.abs(other_vec - gt_vec) / np.where(
            gt_vec > 0, gt_vec, 1.0
        ) * 100
        median_pct_diff = float(np.median(pct_diffs))
        mean_pct_diff = float(np.mean(pct_diffs))

        # Top deviations
        deviation_idx = np.argsort(pct_diffs)[::-1][:10]
        top_deviations = []
        for i in deviation_idx:
            nid = shared_list[i]
            top_deviations.append({
                'flywire_id': int(nid),
                'gt_rate_hz': round(float(gt_vec[i]), 2),
                'other_rate_hz': round(float(other_vec[i]), 2),
                'pct_diff': round(float(pct_diffs[i]), 1),
            })
    else:
        correlation = None
        median_pct_diff = None
        mean_pct_diff = None
        top_deviations = []

    gt_total = int(gt_df.groupby('flywire_id').size().sum())
    other_total = int(other_df.groupby('flywire_id').size().sum())

    return {
        'spike_count': {
            'ground_truth': gt_total,
            'backend': other_total,
            'ratio': round(other_total / gt_total, 4) if gt_total else None,
        },
        'active_neurons': {
            'ground_truth': len(gt_neurons),
            'backend': len(other_neurons),
            'shared': len(shared),
            'gt_only': len(gt_only),
            'backend_only': len(other_only),
            'jaccard': round(jaccard, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
        },
        'firing_rate': {
            'pearson_correlation': (
                round(correlation, 6) if correlation is not None else None
            ),
            'median_pct_diff': (
                round(median_pct_diff, 2)
                if median_pct_diff is not None else None
            ),
            'mean_pct_diff': (
                round(mean_pct_diff, 2)
                if mean_pct_diff is not None else None
            ),
        },
        'top_deviations': top_deviations,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare backend spike outputs against Brian2 (CPU) ground truth',
    )
    parser.add_argument(
        '--t_run', type=float, default=1.0,
        help='Simulation duration in seconds (default: 1.0)',
    )
    parser.add_argument(
        '--n_run', type=int, default=1,
        help='Number of trials (default: 1)',
    )
    parser.add_argument(
        '-o', '--output', type=str, default=str(OUTPUT_PATH),
        help='Output JSON path',
    )
    parser.add_argument(
        '--run-label', '--run_label', dest='run_label', default=None,
        help='Read round-scoped outputs from data/results/<label>/',
    )
    parser.add_argument(
        '--round', dest='round_idx', type=int, default=None,
        help='Round number to compare when using --run-label',
    )
    args = parser.parse_args()

    t_run = args.t_run
    n_run = args.n_run

    print(f"Ground truth comparison: t_run={t_run}s, n_run={n_run}")
    print(f"Ground truth: {BACKENDS[GROUND_TRUTH_KEY]}")
    if args.run_label:
        print(f"Run label: {args.run_label}")
    if args.round_idx is not None:
        print(f"Round: {args.round_idx}")
    print()

    gt_df = load_spike_data(
        GROUND_TRUTH_KEY, t_run, n_run, args.run_label, args.round_idx,
    )
    if gt_df is None:
        expected_path = get_spike_output_path(
            f'{GROUND_TRUTH_KEY}_t{t_run}s_n{n_run}',
            args.run_label,
            args.round_idx,
        ) if args.run_label or args.round_idx is not None else (
            RESULTS_DIR / f'{GROUND_TRUTH_KEY}_t{t_run}s_n{n_run}.parquet'
        )
        print(
            f"ERROR: ground truth parquet not found: "
            f"{expected_path}"
        )
        sys.exit(1)

    gt_rates = firing_rates(gt_df, t_run, n_run)
    print(
        f"  {BACKENDS[GROUND_TRUTH_KEY]}: "
        f"{len(gt_df):,} spikes, "
        f"{gt_rates.index.nunique()} active neurons"
    )
    print()

    output = {
        'ground_truth': BACKENDS[GROUND_TRUTH_KEY],
        'experiment': 'sugar GRNs @ 200 Hz (21 neurons)',
        't_run_sec': t_run,
        'n_run': n_run,
        'run_label': args.run_label,
        'round': args.round_idx,
        'ground_truth_spikes': int(len(gt_df)),
        'ground_truth_active_neurons': int(gt_rates.index.nunique()),
        'comparisons': {},
    }

    for key, name in BACKENDS.items():
        if key == GROUND_TRUTH_KEY:
            continue

        other_df = load_spike_data(
            key, t_run, n_run, args.run_label, args.round_idx,
        )
        if other_df is None:
            print(f"  {name}: parquet not found — skipped")
            output['comparisons'][name] = {'status': 'no data'}
            continue

        metrics = compare(gt_df, other_df, t_run, n_run)
        output['comparisons'][name] = metrics

        sc = metrics['spike_count']
        an = metrics['active_neurons']
        fr = metrics['firing_rate']

        status = 'MATCH' if (
            fr['pearson_correlation'] is not None
            and fr['pearson_correlation'] > 0.99
            and an['jaccard'] > 0.90
        ) else 'CLOSE' if (
            fr['pearson_correlation'] is not None
            and fr['pearson_correlation'] > 0.95
            and an['jaccard'] > 0.80
        ) else 'DEVIATED'

        output['comparisons'][name]['verdict'] = status

        print(f"  {name}:")
        print(f"    Spikes:      {sc['backend']:>10,}  (ratio: {sc['ratio']})")
        print(f"    Active:      {an['backend']:>10}   (Jaccard: {an['jaccard']}, "
              f"shared: {an['shared']}, "
              f"gt_only: {an['gt_only']}, "
              f"backend_only: {an['backend_only']})")
        print(f"    Rate corr:   {fr['pearson_correlation']}")
        print(f"    Median diff: {fr['median_pct_diff']}%")
        print(f"    Verdict:     {status}")
        print()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
