"""
Pairwise spike-output comparisons across benchmark frameworks.

For each requested t_run/n_run combination, this script loads the per-spike
parquet files written by the benchmark runners and computes:

  - active-neuron overlap
  - per-neuron firing-rate parity/correlation
  - spike-time matches within a tolerance window

Outputs are intended as paper/team-facing comparison inputs:
  - pairwise_summary.csv
  - pairwise_summary.json
  - parity_rates.csv
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from benchmark import get_spike_output_path


FRAMEWORKS = {
    'brian2cpp': 'Brian2 (CPU)',
    'brian2cuda': 'Brian2CUDA (GPU)',
    'pytorch': 'PyTorch (CUDA)',
    'nestgpu': 'NEST GPU',
    'genn': 'GeNN (GPU)',
    'brian2genn': 'Brian2GeNN (GPU)',
}

DEFAULT_T_RUN = [0.1, 1.0, 10.0, 100.0]
DEFAULT_N_RUN = [1, 4, 8, 16, 32]


def _spike_path(backend_key, t_run, n_run, run_label=None, round_idx=None):
    exp_name = f'{backend_key}_t{t_run}s_n{n_run}'
    return get_spike_output_path(exp_name, run_label, round_idx)


def load_spikes(backend_key, t_run, n_run, run_label=None, round_idx=None):
    """Load one backend parquet and normalize time to seconds."""
    path = _spike_path(backend_key, t_run, n_run, run_label, round_idx)
    if not path.exists():
        return None, path

    schema_columns = set(pq.read_schema(path).names)
    if 'time_ms' in schema_columns:
        time_column = 'time_ms'
    elif 'time_s' in schema_columns:
        time_column = 'time_s'
    else:
        time_column = 't'

    columns = ['trial', 'flywire_id', 'neuron_index', time_column]
    df = pd.read_parquet(path, columns=columns)
    if len(df) == 0:
        df = pd.DataFrame(
            columns=['trial', 'flywire_id', 'neuron_index', 'time_s'],
        )
        return df, path

    if time_column == 'time_ms':
        time_s = df['time_ms'].to_numpy(dtype=np.float64) / 1000.0
    elif time_column == 'time_s':
        time_s = df['time_s'].to_numpy(dtype=np.float64)
    else:
        time_values = df['t'].to_numpy(dtype=np.float64)
        time_s = time_values / 1000.0 if np.nanmax(time_values) > t_run * 2 else time_values

    out = pd.DataFrame({
        'trial': df['trial'].to_numpy(dtype=np.int16, copy=False),
        'flywire_id': df['flywire_id'].to_numpy(copy=False),
        'neuron_index': df['neuron_index'].to_numpy(copy=False),
        'time_s': time_s,
    })
    return out, path


def firing_rates(df, t_run, n_run):
    counts = df.groupby('flywire_id').size()
    return counts / (t_run * n_run)


def pearson_or_none(a, b):
    if len(a) < 2:
        return None
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def grouped_times(df):
    groups = {}
    for key, group in df.groupby(['trial', 'flywire_id'], sort=False):
        groups[key] = np.sort(group['time_s'].to_numpy(dtype=np.float64))
    return groups


def prepare_metrics(df, t_run, n_run):
    """Precompute reusable per-file metrics for all pairwise comparisons."""
    rates = firing_rates(df, t_run, n_run)
    return {
        'rates': rates,
        'active': set(rates.index),
        'groups': grouped_times(df),
        'spikes': len(df),
    }


def count_timing_matches(times_a, times_b, tolerance_s):
    """Greedy one-to-one spike matching within tolerance."""
    i = 0
    j = 0
    matches = 0
    diffs = []
    while i < len(times_a) and j < len(times_b):
        diff = times_b[j] - times_a[i]
        if abs(diff) <= tolerance_s:
            matches += 1
            diffs.append(abs(diff))
            i += 1
            j += 1
        elif times_a[i] < times_b[j]:
            i += 1
        else:
            j += 1
    return matches, diffs


def compare_pair(key_a, data_a, key_b, data_b, t_run, n_run, tolerance_ms):
    rates_a = data_a['rates']
    rates_b = data_b['rates']

    active_a = data_a['active']
    active_b = data_b['active']
    active_union = active_a | active_b
    active_shared = active_a & active_b

    jaccard = len(active_shared) / len(active_union) if active_union else 0.0
    precision = len(active_shared) / len(active_b) if active_b else 0.0
    recall = len(active_shared) / len(active_a) if active_a else 0.0

    if active_shared:
        shared_ids = sorted(active_shared)
        rate_a = np.array([rates_a[n] for n in shared_ids], dtype=np.float64)
        rate_b = np.array([rates_b[n] for n in shared_ids], dtype=np.float64)
        rate_corr = pearson_or_none(rate_a, rate_b)
        rate_rmse = float(np.sqrt(np.mean((rate_a - rate_b) ** 2)))
        rate_mae = float(np.mean(np.abs(rate_a - rate_b)))
    else:
        rate_corr = None
        rate_rmse = None
        rate_mae = None

    tolerance_s = tolerance_ms / 1000.0
    groups_a = data_a['groups']
    groups_b = data_b['groups']

    timing_matches = 0
    timing_diffs = []
    for group_key in set(groups_a) & set(groups_b):
        matches, diffs = count_timing_matches(
            groups_a[group_key], groups_b[group_key], tolerance_s,
        )
        timing_matches += matches
        timing_diffs.extend(diffs)

    spikes_a = data_a['spikes']
    spikes_b = data_b['spikes']
    timing_f1 = (
        2.0 * timing_matches / (spikes_a + spikes_b)
        if (spikes_a + spikes_b) else 0.0
    )
    timing_precision = timing_matches / spikes_b if spikes_b else 0.0
    timing_recall = timing_matches / spikes_a if spikes_a else 0.0

    return {
        'framework_a': FRAMEWORKS[key_a],
        'framework_b': FRAMEWORKS[key_b],
        'backend_a': key_a,
        'backend_b': key_b,
        't_run': t_run,
        'n_run': n_run,
        'tolerance_ms': tolerance_ms,
        'spikes_a': int(spikes_a),
        'spikes_b': int(spikes_b),
        'spike_count_ratio_b_over_a': (
            round(spikes_b / spikes_a, 6) if spikes_a else None
        ),
        'active_a': int(len(active_a)),
        'active_b': int(len(active_b)),
        'active_shared': int(len(active_shared)),
        'active_jaccard': round(float(jaccard), 6),
        'active_precision_b_over_a': round(float(precision), 6),
        'active_recall_b_over_a': round(float(recall), 6),
        'rate_pearson_shared': (
            round(rate_corr, 8) if rate_corr is not None else None
        ),
        'rate_rmse_hz_shared': (
            round(rate_rmse, 6) if rate_rmse is not None else None
        ),
        'rate_mae_hz_shared': (
            round(rate_mae, 6) if rate_mae is not None else None
        ),
        'timing_matches': int(timing_matches),
        'timing_f1': round(float(timing_f1), 8),
        'timing_precision_b_over_a': round(float(timing_precision), 8),
        'timing_recall_b_over_a': round(float(timing_recall), 8),
        'timing_mean_abs_dt_ms': (
            round(float(np.mean(timing_diffs) * 1000.0), 6)
            if timing_diffs else None
        ),
        'timing_median_abs_dt_ms': (
            round(float(np.median(timing_diffs) * 1000.0), 6)
            if timing_diffs else None
        ),
    }


def parity_rows(key_a, data_a, key_b, data_b, t_run, n_run):
    rates_a = data_a['rates']
    rates_b = data_b['rates']
    active_union = sorted(set(rates_a.index) | set(rates_b.index))
    if not active_union:
        return pd.DataFrame()
    return pd.DataFrame({
        'backend_a': key_a,
        'framework_a': FRAMEWORKS[key_a],
        'backend_b': key_b,
        'framework_b': FRAMEWORKS[key_b],
        't_run': t_run,
        'n_run': n_run,
        'flywire_id': active_union,
        'rate_a_hz': [float(rates_a.get(n, 0.0)) for n in active_union],
        'rate_b_hz': [float(rates_b.get(n, 0.0)) for n in active_union],
    })


def main():
    parser = argparse.ArgumentParser(
        description='Pairwise framework spike-output comparisons',
    )
    parser.add_argument('--run-label', '--run_label', dest='run_label', required=True)
    parser.add_argument('--round', dest='round_idx', type=int, default=1)
    parser.add_argument('--t_run', type=float, nargs='+', default=DEFAULT_T_RUN)
    parser.add_argument('--n_run', type=int, nargs='+', default=DEFAULT_N_RUN)
    parser.add_argument('--tolerance-ms', type=float, default=1.0)
    parser.add_argument(
        '-o', '--output-dir', default=None,
        help='Default: data/results/<run-label>/comparisons',
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path('data/results') / args.run_label / 'comparisons'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    parity_frames = []
    missing = []

    for n_run in args.n_run:
        for t_run in args.t_run:
            loaded = {}
            for key in FRAMEWORKS:
                df, path = load_spikes(key, t_run, n_run, args.run_label, args.round_idx)
                if df is None:
                    missing.append({
                        'backend': key,
                        'framework': FRAMEWORKS[key],
                        't_run': t_run,
                        'n_run': n_run,
                        'path': str(path),
                    })
                else:
                    loaded[key] = prepare_metrics(df, t_run, n_run)

            for key_a, key_b in combinations(loaded.keys(), 2):
                row = compare_pair(
                    key_a, loaded[key_a], key_b, loaded[key_b],
                    t_run, n_run, args.tolerance_ms,
                )
                summary.append(row)
                parity_frames.append(
                    parity_rows(key_a, loaded[key_a], key_b, loaded[key_b], t_run, n_run)
                )

    summary_df = pd.DataFrame(summary)
    summary_csv = output_dir / 'pairwise_summary.csv'
    summary_json = output_dir / 'pairwise_summary.json'
    parity_csv = output_dir / 'parity_rates.csv'
    missing_json = output_dir / 'missing_inputs.json'

    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    if parity_frames:
        pd.concat(parity_frames, ignore_index=True).to_csv(parity_csv, index=False)
    else:
        pd.DataFrame().to_csv(parity_csv, index=False)

    with open(missing_json, 'w') as f:
        json.dump(missing, f, indent=2)

    print(f'Pairwise rows: {len(summary_df)}')
    print(f'Missing inputs: {len(missing)}')
    print(f'Summary CSV: {summary_csv}')
    print(f'Summary JSON: {summary_json}')
    print(f'Parity CSV: {parity_csv}')
    if missing:
        print(f'Missing JSON: {missing_json}')


if __name__ == '__main__':
    main()
