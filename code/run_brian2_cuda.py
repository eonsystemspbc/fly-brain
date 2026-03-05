"""
Brian2 / Brian2CUDA benchmark runner.

CPU path: joblib-parallelized trials in runtime mode (matching Phil's original).
GPU path: CUDA standalone, build once, run N times sequentially.
Called by benchmark.py orchestrator.
"""

import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import pandas as pd
from pathlib import Path
from textwrap import dedent
from time import time
import traceback
import shutil

from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, Network, mV, ms, Hz
from brian2 import set_device, device, prefs

import logging
from brian2.utils.logger import BrianLogger
BrianLogger.console_handler.setLevel(logging.WARNING)

from joblib import Parallel, delayed, parallel_backend

from benchmark import (
    T_RUN_VALUES_SEC, N_RUN_VALUES,
    output_dir, path_comp, path_con, path_res,
    neu_exc, neu_exc2, neu_slnc,
    print_summary_table, save_result_csv,
)

# ============================================================================
# Brian2 Network Parameters
# ============================================================================

default_params = {
    'v_0': -52 * mV,
    'v_rst': -52 * mV,
    'v_th': -45 * mV,
    't_mbr': 20 * ms,
    'tau': 5 * ms,
    't_rfc': 2.2 * ms,
    't_dly': 1.8 * ms,
    'w_syn': 0.275 * mV,
    'r_poi': 100 * Hz,
    'r_poi2': 0 * Hz,
    'f_poi': 250,
    'eqs': dedent('''
        dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
        dg/dt = -g / tau               : volt (unless refractory) 
        rfc                           : second
    '''),
    'eq_th': 'v > v_th',
    'eq_rst': 'v = v_rst; w = 0; g = 0 * mV',
}

# ============================================================================
# Runtime trial (for joblib parallelization on CPU)
# ============================================================================

def _run_trial_cpu(exc, exc2, slnc, path_comp_str, path_con_str, params):
    """Run a single trial in Brian2 runtime mode.

    Each joblib worker calls this with a fresh Python process (loky backend),
    so every trial creates its own independent network.  This mirrors Phil's
    original ``run_trial`` function.
    """
    import warnings
    warnings.filterwarnings('ignore')

    df_comp = pd.read_csv(path_comp_str, index_col=0)
    df_con = pd.read_parquet(path_con_str)

    neu = NeuronGroup(
        N=len(df_comp), model=params['eqs'], method='linear',
        threshold=params['eq_th'], reset=params['eq_rst'],
        refractory='rfc', name='default_neurons', namespace=params,
    )
    neu.v = params['v_0']
    neu.g = 0
    neu.rfc = params['t_rfc']

    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w',
                   delay=params['t_dly'], name='default_synapses')
    syn.connect(i=df_con['Presynaptic_Index'].values,
                j=df_con['Postsynaptic_Index'].values)
    syn.w = df_con['Excitatory x Connectivity'].values * params['w_syn']

    spk_mon = SpikeMonitor(neu)

    pois = []
    for i in exc:
        p = PoissonInput(target=neu[i], target_var='v', N=1,
                         rate=params['r_poi'],
                         weight=params['w_syn'] * params['f_poi'])
        neu[i].rfc = 0 * ms
        pois.append(p)
    for i in exc2:
        p = PoissonInput(target=neu[i], target_var='v', N=1,
                         rate=params['r_poi2'],
                         weight=params['w_syn'] * params['f_poi'])
        neu[i].rfc = 0 * ms
        pois.append(p)

    for i in slnc:
        syn.w[' {} == i'.format(i)] = 0 * mV

    net = Network(neu, syn, spk_mon, *pois)
    net.run(duration=params['t_run'])

    spk_trn = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
    return spk_trn

# ============================================================================
# Standalone helpers (for GPU and CPU n_run=1)
# ============================================================================

def create_network(path_comp, path_con, params, logger=None):
    """Create Brian2 network from connectivity data."""
    t_start = time()

    t_load_start = time()
    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)
    t_load = time() - t_load_start

    t_neurons_start = time()
    neu = NeuronGroup(
        N=len(df_comp), model=params['eqs'], method='linear',
        threshold=params['eq_th'], reset=params['eq_rst'],
        refractory='rfc', name='default_neurons', namespace=params,
    )
    neu.v = params['v_0']
    neu.g = 0
    neu.rfc = params['t_rfc']
    t_neurons = time() - t_neurons_start

    t_synapses_start = time()
    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w',
                   delay=params['t_dly'], name='default_synapses')
    syn.connect(i=df_con['Presynaptic_Index'].values,
                j=df_con['Postsynaptic_Index'].values)
    syn.w = df_con['Excitatory x Connectivity'].values * params['w_syn']
    t_synapses = time() - t_synapses_start

    spk_mon = SpikeMonitor(neu)

    timings = {
        'data_load': t_load,
        'neuron_creation': t_neurons,
        'synapse_creation': t_synapses,
        'network_creation_total': time() - t_start,
    }

    return neu, syn, spk_mon, df_comp, timings


def add_poisson_inputs(neu, exc, exc2, params):
    """Add Poisson inputs to specified neurons."""
    pois = []
    for i in exc:
        p = PoissonInput(target=neu[i], target_var='v', N=1,
                         rate=params['r_poi'],
                         weight=params['w_syn'] * params['f_poi'])
        neu[i].rfc = 0 * ms
        pois.append(p)
    for i in exc2:
        p = PoissonInput(target=neu[i], target_var='v', N=1,
                         rate=params['r_poi2'],
                         weight=params['w_syn'] * params['f_poi'])
        neu[i].rfc = 0 * ms
        pois.append(p)
    return pois


def silence_neurons(syn, slnc):
    """Set synapse weights to 0 for silenced neurons."""
    for i in slnc:
        syn.w[' {} == i'.format(i)] = 0 * mV

# ============================================================================
# Benchmark: standalone path (GPU, or CPU n_run=1)
# ============================================================================

def _run_standalone_benchmark(t_run_sec, n_run, use_cuda, exc, exc2, slnc,
                              i2flyid, logger, exp_name, timings):
    """Run benchmark using Brian2 standalone mode (C++ or CUDA)."""
    from brian2 import device as brian_device

    brian_device.reinit()
    brian_device.activate()

    if use_cuda:
        set_device('cuda_standalone', build_on_run=False)
    else:
        set_device('cpp_standalone', build_on_run=False)

    device_location = 'GPU' if use_cuda else 'CPU'
    t_run = t_run_sec * 1000 * ms

    # Network creation
    logger.log("Creating network...")
    t_network_start = time()

    neu, syn, spk_mon, _, network_timings = create_network(
        path_comp, path_con, default_params, logger
    )
    timings.update(network_timings)

    t_poisson_start = time()
    poi_inp = add_poisson_inputs(neu, exc, exc2, default_params)
    timings['poisson_inputs'] = time() - t_poisson_start

    if slnc:
        silence_neurons(syn, slnc)

    net = Network(neu, syn, spk_mon, *poi_inp)
    timings['network_creation_total'] = time() - t_network_start

    logger.log(f"  Data loading:     {timings['data_load']:.3f}s")
    logger.log(f"  Neuron creation:  {timings['neuron_creation']:.3f}s")
    logger.log(f"  Synapse creation: {timings['synapse_creation']:.3f}s")
    logger.log(f"  Poisson inputs:   {timings['poisson_inputs']:.3f}s")
    logger.log(f"  Total network:    {timings['network_creation_total']:.3f}s")

    # Code generation + build
    net.run(duration=t_run)

    device_type = 'CUDA' if use_cuda else 'C++'
    logger.log(f"Building {device_type} standalone code...")

    if output_dir.exists():
        shutil.rmtree(output_dir)

    t_build_start = time()
    brian_device.build(directory=output_dir, run=False, with_output=False)
    timings['device_build'] = time() - t_build_start
    logger.log(f"  Build time:       {timings['device_build']:.3f}s")

    # Run simulations
    logger.log(f"Running {n_run} trial(s) on {device_location}...")

    simulation_results = []
    trial_times = []

    t_simulation_start = time()
    for trial_idx in range(n_run):
        t_trial_start = time()
        brian_device.run(with_output=False)

        t_extract_start = time()
        spk_trn = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
        t_extract = time() - t_extract_start

        trial_time = time() - t_trial_start
        trial_times.append(trial_time)
        simulation_results.append(spk_trn)

        if n_run <= 5 or (trial_idx + 1) % 5 == 0 or trial_idx == 0:
            logger.log(
                f"  Trial {trial_idx + 1}/{n_run}: "
                f"{trial_time:.3f}s (extract: {t_extract:.3f}s)"
            )

    timings['simulation_total'] = time() - t_simulation_start
    timings['simulation_avg_per_trial'] = timings['simulation_total'] / n_run

    logger.log(f"  Total simulation: {timings['simulation_total']:.3f}s")
    logger.log(f"  Avg per trial:    {timings['simulation_avg_per_trial']:.3f}s")

    return simulation_results, timings

# ============================================================================
# Benchmark: joblib-parallel path (CPU, n_run > 1)
# ============================================================================

def _run_parallel_benchmark(t_run_sec, n_run, exc, exc2, slnc, logger,
                            exp_name, timings):
    """Run benchmark using joblib parallelization across CPU cores."""
    total_cores = os.cpu_count() or 1
    n_cores = max(1, total_cores - 4)
    t_run = t_run_sec * 1000 * ms

    trial_params = dict(default_params)
    trial_params['t_run'] = t_run

    timings['network_creation_total'] = 0.0
    timings['device_build'] = 0.0

    logger.log(
        f"Running {n_run} trial(s) in parallel "
        f"(joblib, {n_cores}/{total_cores} cores)..."
    )

    t_simulation_start = time()
    with parallel_backend('loky', n_jobs=n_cores):
        simulation_results = Parallel()(
            delayed(_run_trial_cpu)(
                exc, exc2, slnc, str(path_comp), str(path_con), trial_params
            )
            for _ in range(n_run)
        )
    timings['simulation_total'] = time() - t_simulation_start
    timings['simulation_avg_per_trial'] = timings['simulation_total'] / n_run

    logger.log(f"  Total simulation: {timings['simulation_total']:.3f}s")
    logger.log(f"  Avg per trial:    {timings['simulation_avg_per_trial']:.3f}s")

    return simulation_results, timings

# ============================================================================
# Main benchmark entry point
# ============================================================================

def run_single_benchmark(t_run_sec, n_run, use_cuda, logger,
                         run_idx=None, total_runs=None):
    """Run a single Brian2 benchmark with specified t_run and n_run.

    CPU n_run>1  → joblib parallel (runtime mode, all cores)
    CPU n_run=1  → C++ standalone (fast single trial)
    GPU any      → CUDA standalone (build once, run N times)
    """
    device_location = 'GPU' if use_cuda else 'CPU'
    exp_name = f'brian2{"cuda" if use_cuda else "cpp"}_t{t_run_sec}s_n{n_run}'

    run_info = f"[{run_idx}/{total_runs}] " if run_idx else ""
    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"{run_info}BENCHMARK: t_run={t_run_sec}s, n_run={n_run}")
    logger.log_raw("=" * 80)
    logger.log(f"Device: {device_location} "
               f"({'CUDA' if use_cuda else 'C++'})")
    logger.log(f"Experiment: {exp_name}")

    timings = {}
    results = {}

    try:
        # ===== ID mapping =====
        t_mapping_start = time()
        df_comp = pd.read_csv(path_comp, index_col=0)
        flyid2i = {j: i for i, j in enumerate(df_comp.index)}
        i2flyid = {j: i for i, j in flyid2i.items()}
        exc = [flyid2i[n] for n in neu_exc]
        exc2 = [flyid2i[n] for n in neu_exc2]
        slnc = [flyid2i[n] for n in neu_slnc]
        timings['id_mapping'] = time() - t_mapping_start
        logger.log(f"ID mapping:         {timings['id_mapping']:.3f}s")

        # ===== Run trials =====
        use_parallel = (not use_cuda) and (n_run > 1)

        if use_parallel:
            simulation_results, timings = _run_parallel_benchmark(
                t_run_sec, n_run, exc, exc2, slnc, logger, exp_name, timings
            )
        else:
            simulation_results, timings = _run_standalone_benchmark(
                t_run_sec, n_run, use_cuda, exc, exc2, slnc, i2flyid,
                logger, exp_name, timings
            )

        # ===== Collect and save results =====
        logger.log("Collecting results...")
        t_collect_start = time()

        ids, ts, trials = [], [], []
        for trial_idx, spk_dict in enumerate(simulation_results):
            for neuron_id, spike_times in spk_dict.items():
                ids.extend([neuron_id] * len(spike_times))
                trials.extend([trial_idx] * len(spike_times))
                ts.extend([float(t) for t in spike_times])

        df = pd.DataFrame({
            't': ts,
            'trial': trials,
            'flywire_id': [i2flyid[i] for i in ids],
            'exp_name': exp_name,
        })

        timings['result_collection'] = time() - t_collect_start

        t_save_start = time()
        Path(path_res).mkdir(parents=True, exist_ok=True)
        path_save = Path(path_res) / f'{exp_name}.parquet'
        df.to_parquet(path_save, compression='brotli')
        timings['result_save'] = time() - t_save_start

        logger.log(f"  Collection:       {timings['result_collection']:.3f}s")
        logger.log(f"  Save to file:     {timings['result_save']:.3f}s")
        logger.log(f"  Output file:      {path_save}")

        # ===== Totals =====
        timings['total_elapsed'] = (
            timings['id_mapping'] +
            timings['network_creation_total'] +
            timings.get('device_build', 0) +
            timings['simulation_total'] +
            timings['result_collection'] +
            timings['result_save']
        )

        total_simulated_time = t_run_sec * n_run
        timings['realtime_ratio'] = (
            total_simulated_time / timings['simulation_total']
            if timings['simulation_total'] > 0 else float('inf')
        )
        timings['realtime_ratio_total'] = (
            total_simulated_time / timings['total_elapsed']
            if timings['total_elapsed'] > 0 else float('inf')
        )

        n_active = len(set(ids))
        n_spikes = len(df)

        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': n_active,
            'n_spikes': n_spikes,
            'status': 'success',
            'timings': timings,
        }

        # ===== Summary =====
        logger.log_raw("")
        logger.log_raw("-" * 60)
        logger.log("TIMING SUMMARY")
        logger.log_raw("-" * 60)
        logger.log(f"  Network creation:   {timings['network_creation_total']:>10.3f}s")
        logger.log(f"  Device build:       {timings.get('device_build', 0):>10.3f}s")
        logger.log(f"  Simulation:         {timings['simulation_total']:>10.3f}s")
        logger.log(f"  Result processing:  "
                   f"{timings['result_collection'] + timings['result_save']:>10.3f}s")
        logger.log(f"  -----------------------------------------")
        logger.log(f"  TOTAL ELAPSED:      {timings['total_elapsed']:>10.3f}s")
        logger.log_raw("")
        logger.log(f"  Simulated time:     "
                   f"{total_simulated_time:>10.1f}s ({n_run} x {t_run_sec}s)")
        logger.log(f"  Realtime ratio (sim only): "
                   f"{timings['realtime_ratio']:>6.3f}x")
        logger.log(f"  Realtime ratio (total):    "
                   f"{timings['realtime_ratio_total']:>6.3f}x")
        logger.log_raw("")
        logger.log(f"  Active neurons:     {n_active:>10d}")
        logger.log(f"  Total spikes:       {n_spikes:>10d}")
        logger.log_raw("-" * 60)

    except Exception as e:
        logger.log(f"ERROR: {str(e)}")
        logger.log_raw(traceback.format_exc())
        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': 0,
            'n_spikes': 0,
            'status': f'error: {str(e)}',
            'timings': timings,
        }

    return results


def run_all_benchmarks(use_cuda, t_run_values=None, n_run_values=None,
                       logger=None):
    """Run all Brian2/Brian2CUDA benchmark combinations."""
    if t_run_values is None:
        t_run_values = T_RUN_VALUES_SEC
    if n_run_values is None:
        n_run_values = N_RUN_VALUES

    backend_name = 'Brian2CUDA (GPU)' if use_cuda else 'Brian2 (CPU)'
    device_label = 'GPU (CUDA)' if use_cuda else 'CPU (C++)'

    benchmarks = []
    for n_run in n_run_values:
        for t_run_sec in t_run_values:
            benchmarks.append((t_run_sec, n_run))

    total_runs = len(benchmarks)

    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"BENCHMARK SUITE: {backend_name}")
    logger.log_raw("=" * 80)
    logger.log(f"Device: {device_label}")
    logger.log(f"t_run values: {t_run_values} seconds")
    logger.log(f"n_run values: {n_run_values}")
    logger.log(f"Total benchmarks: {total_runs}")
    logger.log_raw("=" * 80)

    all_results = []

    for run_idx, (t_run_sec, n_run) in enumerate(benchmarks, 1):
        result = run_single_benchmark(
            t_run_sec=t_run_sec,
            n_run=n_run,
            use_cuda=use_cuda,
            logger=logger,
            run_idx=run_idx,
            total_runs=total_runs,
        )
        all_results.append(result)
        save_result_csv(backend_name, result)

    print_summary_table(all_results, backend_name, logger)

    return all_results
