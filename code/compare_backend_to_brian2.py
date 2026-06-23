"""
Compare one backend against Brian2 CPU across labeled benchmark rounds.

This produces team-facing parity inputs matching the existing
``genn_vs_brian2_*`` support-file shape, generalized to any backend with saved
spike parquet files.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from compare_spike_outputs import (
    DEFAULT_N_RUN,
    DEFAULT_T_RUN,
    FRAMEWORKS,
    compare_pair,
    load_spikes,
    parity_rows,
    prepare_metrics,
)
from benchmark import get_spike_output_path


REFERENCE_BACKEND = 'brian2cpp'


def load_rate_data(backend_key, t_run, n_run, run_label, round_idx):
    """Load only per-neuron rates for large all-round parity comparisons."""
    path = get_spike_output_path(
        f'{backend_key}_t{t_run}s_n{n_run}', run_label, round_idx,
    )
    if not path.exists():
        return None, path

    df = pd.read_parquet(path, columns=['flywire_id'])
    rates = df.groupby('flywire_id').size() / (t_run * n_run)
    return {
        'rates': rates,
        'active': set(rates.index),
        'spikes': len(df),
    }, path


def rate_compare(ref_key, ref_data, comp_key, comp_data, t_run, n_run,
                 tolerance_ms):
    """Compute active-neuron and rate-parity metrics without spike-time matching."""
    rates_a = ref_data['rates']
    rates_b = comp_data['rates']
    active_a = ref_data['active']
    active_b = comp_data['active']
    active_shared = active_a & active_b
    active_union = active_a | active_b

    if active_shared:
        shared_ids = sorted(active_shared)
        rate_a = pd.Series([rates_a[n] for n in shared_ids], dtype='float64')
        rate_b = pd.Series([rates_b[n] for n in shared_ids], dtype='float64')
        rate_corr = float(rate_a.corr(rate_b))
        rate_rmse = float(((rate_a - rate_b) ** 2).mean() ** 0.5)
        rate_mae = float((rate_a - rate_b).abs().mean())
    else:
        rate_corr = None
        rate_rmse = None
        rate_mae = None

    spikes_a = ref_data['spikes']
    spikes_b = comp_data['spikes']
    return {
        'framework_a': FRAMEWORKS[ref_key],
        'framework_b': FRAMEWORKS[comp_key],
        'backend_a': ref_key,
        'backend_b': comp_key,
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
        'active_jaccard': (
            round(len(active_shared) / len(active_union), 6)
            if active_union else 0.0
        ),
        'active_precision_b_over_a': (
            round(len(active_shared) / len(active_b), 6) if active_b else 0.0
        ),
        'active_recall_b_over_a': (
            round(len(active_shared) / len(active_a), 6) if active_a else 0.0
        ),
        'rate_pearson_shared': (
            round(rate_corr, 8) if rate_corr is not None else None
        ),
        'rate_rmse_hz_shared': (
            round(rate_rmse, 6) if rate_rmse is not None else None
        ),
        'rate_mae_hz_shared': (
            round(rate_mae, 6) if rate_mae is not None else None
        ),
        'timing_matches': None,
        'timing_f1': None,
        'timing_precision_b_over_a': None,
        'timing_recall_b_over_a': None,
        'timing_mean_abs_dt_ms': None,
        'timing_median_abs_dt_ms': None,
    }


def rate_parity_rows(ref_key, ref_data, comp_key, comp_data, t_run, n_run):
    """Build parity rows from precomputed rate series."""
    rates_a = ref_data['rates']
    rates_b = comp_data['rates']
    active_union = sorted(set(rates_a.index) | set(rates_b.index))
    if not active_union:
        return pd.DataFrame()
    return pd.DataFrame({
        'reference_backend': ref_key,
        'reference_framework': FRAMEWORKS[ref_key],
        'comparison_backend': comp_key,
        'comparison_framework': FRAMEWORKS[comp_key],
        't_run': t_run,
        'n_run': n_run,
        'flywire_id': active_union,
        'reference_rate_hz': [float(rates_a.get(n, 0.0)) for n in active_union],
        'comparison_rate_hz': [float(rates_b.get(n, 0.0)) for n in active_union],
    })


def summary_row(raw, run_label, round_idx, ref_path, comp_path):
    """Map generic pairwise metrics to Brian2-vs-backend summary columns."""
    return {
        'run_label': run_label,
        'round': round_idx,
        'reference_framework': raw['framework_a'],
        'comparison_framework': raw['framework_b'],
        'reference_backend': raw['backend_a'],
        'comparison_backend': raw['backend_b'],
        'n_run': raw['n_run'],
        't_run': raw['t_run'],
        'tolerance_ms': raw['tolerance_ms'],
        'spikes_reference': raw['spikes_a'],
        'spikes_comparison': raw['spikes_b'],
        'spike_count_ratio_comparison_over_reference': (
            raw['spike_count_ratio_b_over_a']
        ),
        'active_reference': raw['active_a'],
        'active_comparison': raw['active_b'],
        'active_shared': raw['active_shared'],
        'active_union': raw['active_a'] + raw['active_b'] - raw['active_shared'],
        'active_jaccard': raw['active_jaccard'],
        'active_precision_comparison_over_reference': (
            raw['active_precision_b_over_a']
        ),
        'active_recall_comparison_over_reference': raw['active_recall_b_over_a'],
        'rate_pearson_shared': raw['rate_pearson_shared'],
        'rate_rmse_hz_shared': raw['rate_rmse_hz_shared'],
        'rate_mae_hz_shared': raw['rate_mae_hz_shared'],
        'timing_matches': raw['timing_matches'],
        'timing_f1': raw['timing_f1'],
        'timing_precision_comparison_over_reference': (
            raw['timing_precision_b_over_a']
        ),
        'timing_recall_comparison_over_reference': (
            raw['timing_recall_b_over_a']
        ),
        'timing_mean_abs_dt_ms': raw['timing_mean_abs_dt_ms'],
        'timing_median_abs_dt_ms': raw['timing_median_abs_dt_ms'],
        'reference_spike_path': str(ref_path),
        'comparison_spike_path': str(comp_path),
    }


def renamed_parity_rows(rows, run_label, round_idx):
    """Return parity rows with Brian2-vs-backend naming."""
    rows = rows.rename(columns={
        'backend_a': 'reference_backend',
        'framework_a': 'reference_framework',
        'backend_b': 'comparison_backend',
        'framework_b': 'comparison_framework',
        'rate_a_hz': 'reference_rate_hz',
        'rate_b_hz': 'comparison_rate_hz',
    })
    rows.insert(0, 'round', round_idx)
    rows.insert(0, 'run_label', run_label)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description='Compare one backend against Brian2 CPU across rounds',
    )
    parser.add_argument('--run-label', '--run_label', dest='run_label',
                        required=True)
    parser.add_argument('--backend', required=True, choices=FRAMEWORKS.keys())
    parser.add_argument('--rounds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--t_run', type=float, nargs='+', default=DEFAULT_T_RUN)
    parser.add_argument('--n_run', type=int, nargs='+', default=DEFAULT_N_RUN)
    parser.add_argument('--tolerance-ms', type=float, default=1.0)
    parser.add_argument(
        '--include-timing', action='store_true',
        help='Also compute greedy spike-time matches. This is expensive for '
             'large t_run/n_run parquets and is off by default.',
    )
    parser.add_argument(
        '-o', '--output-dir', default=None,
        help='Default: data/results/<run-label>/comparisons',
    )
    args = parser.parse_args()

    if args.backend == REFERENCE_BACKEND:
        raise ValueError('Comparison backend must differ from Brian2 CPU')

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path('data/results') / args.run_label / 'comparisons'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    parity_frames = []
    missing = []

    for round_idx in args.rounds:
        for n_run in args.n_run:
            for t_run in args.t_run:
                if args.include_timing:
                    ref_df, ref_path = load_spikes(
                        REFERENCE_BACKEND, t_run, n_run,
                        args.run_label, round_idx,
                    )
                    comp_df, comp_path = load_spikes(
                        args.backend, t_run, n_run, args.run_label, round_idx,
                    )
                    ref_missing = ref_df is None
                    comp_missing = comp_df is None
                else:
                    ref_data, ref_path = load_rate_data(
                        REFERENCE_BACKEND, t_run, n_run,
                        args.run_label, round_idx,
                    )
                    comp_data, comp_path = load_rate_data(
                        args.backend, t_run, n_run, args.run_label, round_idx,
                    )
                    ref_missing = ref_data is None
                    comp_missing = comp_data is None

                if ref_missing or comp_missing:
                    if ref_missing:
                        missing.append({
                            'backend': REFERENCE_BACKEND,
                            'framework': FRAMEWORKS[REFERENCE_BACKEND],
                            'round': round_idx,
                            't_run': t_run,
                            'n_run': n_run,
                            'path': str(ref_path),
                        })
                    if comp_missing:
                        missing.append({
                            'backend': args.backend,
                            'framework': FRAMEWORKS[args.backend],
                            'round': round_idx,
                            't_run': t_run,
                            'n_run': n_run,
                            'path': str(comp_path),
                        })
                    continue

                if args.include_timing:
                    ref_data = prepare_metrics(ref_df, t_run, n_run)
                    comp_data = prepare_metrics(comp_df, t_run, n_run)
                    raw = compare_pair(
                        REFERENCE_BACKEND, ref_data,
                        args.backend, comp_data,
                        t_run, n_run, args.tolerance_ms,
                    )
                    parity = parity_rows(
                        REFERENCE_BACKEND, ref_data,
                        args.backend, comp_data,
                        t_run, n_run,
                    )
                    if not parity.empty:
                        parity = renamed_parity_rows(
                            parity, args.run_label, round_idx,
                        )
                else:
                    raw = rate_compare(
                        REFERENCE_BACKEND, ref_data,
                        args.backend, comp_data,
                        t_run, n_run, args.tolerance_ms,
                    )
                    parity = rate_parity_rows(
                        REFERENCE_BACKEND, ref_data,
                        args.backend, comp_data,
                        t_run, n_run,
                    )
                    if not parity.empty:
                        parity.insert(0, 'round', round_idx)
                        parity.insert(0, 'run_label', args.run_label)

                summary.append(
                    summary_row(raw, args.run_label, round_idx, ref_path, comp_path)
                )
                if not parity.empty:
                    parity_frames.append(parity)

    prefix = f'{args.backend}_vs_brian2'
    summary_csv = output_dir / f'{prefix}_rate_summary.csv'
    summary_json = output_dir / f'{prefix}_rate_summary.json'
    parity_csv = output_dir / f'{prefix}_rate_parity.csv'
    missing_json = output_dir / f'{prefix}_missing_inputs.json'

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)

    if parity_frames:
        pd.concat(parity_frames, ignore_index=True).to_csv(parity_csv, index=False)
    else:
        pd.DataFrame().to_csv(parity_csv, index=False)

    with open(missing_json, 'w') as f:
        json.dump(missing, f, indent=2)

    print(f'Summary rows: {len(summary_df)}')
    print(f'Missing inputs: {len(missing)}')
    print(f'Summary CSV: {summary_csv}')
    print(f'Summary JSON: {summary_json}')
    print(f'Parity CSV: {parity_csv}')
    print(f'Missing JSON: {missing_json}')


if __name__ == '__main__':
    main()
