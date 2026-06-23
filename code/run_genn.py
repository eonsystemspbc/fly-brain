"""
GeNN benchmark runner for the Drosophila brain model.

Implements the FlyWire LIF benchmark with PyGeNN's CUDA backend.  The model
uses batched GeNN simulations for n_run trials, Brian2-style Poisson activation
into membrane voltage, delayed sparse recurrent synapses, and GeNN spike
recording for per-neuron spike timing exports.

Called by benchmark.py orchestrator.
"""

import os
import shutil
import traceback
from math import gcd
from pathlib import Path
from time import perf_counter as time

import numpy as np
import pandas as pd
import pyarrow  # noqa: F401  - import before optional GPU libs

from benchmark import (
    T_RUN_VALUES_SEC, N_RUN_VALUES,
    output_dir, path_comp, path_con,
    get_experiment, get_spike_output_path, print_summary_table, save_result_csv,
)

# ============================================================================
# GeNN Model Parameters
# ============================================================================

MODEL_PARAMS = {
    'tauSyn': 5.0,        # ms
    'tDelay': 1.8,        # ms
    'v0': -52.0,          # mV
    'vReset': -52.0,      # mV
    'vRest': -52.0,       # mV
    'vThreshold': -45.0,  # mV
    'tauMem': 20.0,       # ms
    'tRefrac': 2.2,       # ms
    'scalePoisson': 250.0,
    'wScale': 0.275,
}

DT = 0.1  # ms


# ============================================================================
# Import / Environment Helpers
# ============================================================================

def _configure_cuda_environment():
    """Make the WSL CUDA toolkit discoverable for GeNN-generated builds."""
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if not cuda_path:
        default_cuda = Path('/usr/local/cuda-12.5')
        if default_cuda.exists():
            cuda_path = str(default_cuda)
            os.environ['CUDA_PATH'] = cuda_path
            os.environ['CUDA_HOME'] = cuda_path

    if cuda_path:
        cuda_bin = str(Path(cuda_path) / 'bin')
        cuda_lib = str(Path(cuda_path) / 'lib64')
        path_parts = os.environ.get('PATH', '').split(os.pathsep)
        if cuda_bin not in path_parts:
            os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')
        ld_parts = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
        if cuda_lib not in ld_parts:
            os.environ['LD_LIBRARY_PATH'] = (
                cuda_lib + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
            )


def _import_pygenn():
    _configure_cuda_environment()
    try:
        from pygenn import (  # noqa: WPS433 - runtime dependency check
            GeNNModel, create_neuron_model, init_weight_update,
            init_postsynaptic,
        )
    except ImportError as exc:
        raise ImportError(
            "PyGeNN is not installed. Install GeNN/PyGeNN 5.x, e.g. "
            "`pip install https://github.com/genn-team/genn/archive/refs/tags/5.4.0.zip`."
        ) from exc

    return GeNNModel, create_neuron_model, init_weight_update, init_postsynaptic


def _synchronize_cuda():
    """Synchronize CUDA kernels without pulling spike data from GeNN."""
    try:
        import torch  # noqa: WPS433 - optional sync helper
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _recording_window_steps(num_steps, n_run):
    """Choose an exact GeNN spike-recording window for the run length."""
    requested = int(os.environ.get('GENN_RECORDING_WINDOW_STEPS', '100000'))
    requested = max(1, requested)
    max_slots = int(os.environ.get('GENN_RECORDING_WINDOW_MAX_SLOTS', '800000'))
    max_slots = max(1, max_slots)
    batch_limited = max(1, max_slots // max(1, int(n_run)))
    window = min(num_steps, requested, batch_limited)
    if num_steps % window != 0:
        window = gcd(num_steps, window)
    return max(1, window)


# ============================================================================
# Data Utilities
# ============================================================================

def _load_model_data(experiment):
    """Load FlyWire IDs, stimulation indices, and recurrent sparse edges."""
    df_comp = pd.read_csv(path_comp, index_col=0)
    flywire_ids = df_comp.index.to_numpy()
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}

    exc = [flyid2i[n] for n in experiment['neu_exc']]
    exc2 = [flyid2i[n] for n in experiment['neu_exc2']]
    slnc = [flyid2i[n] for n in experiment['neu_slnc']]

    df_con = pd.read_parquet(path_con)
    pre = df_con['Presynaptic_Index'].to_numpy(dtype=np.uint32, copy=True)
    post = df_con['Postsynaptic_Index'].to_numpy(dtype=np.uint32, copy=True)
    weights = (
        df_con['Excitatory x Connectivity'].to_numpy(dtype=np.float32, copy=True)
        * np.float32(MODEL_PARAMS['wScale'])
    )

    if slnc:
        slnc_arr = np.asarray(slnc, dtype=np.uint32)
        keep = ~(
            np.isin(pre, slnc_arr, assume_unique=False)
            | np.isin(post, slnc_arr, assume_unique=False)
        )
        pre = pre[keep]
        post = post[keep]
        weights = weights[keep]

    return {
        'df_comp': df_comp,
        'flywire_ids': flywire_ids,
        'exc': exc,
        'exc2': exc2,
        'pre': pre,
        'post': post,
        'weights': weights,
    }


# ============================================================================
# GeNN Model Construction
# ============================================================================

def _create_genn_models(create_neuron_model):
    """Create custom GeNN source and LIF neuron models."""
    source_model = create_neuron_model(
        'FlyBrainBernoulliSource',
        params=[('Rate', 'scalar'), ('DT', 'scalar')],
        vars=[('Spike', 'scalar')],
        sim_code="""
        Spike = (gennrand_uniform() < (Rate * DT / 1000.0));
        """,
        threshold_condition_code='Spike > 0.0',
        reset_code='',
    )

    lif_model = create_neuron_model(
        'FlyBrainVoltageStimLIF',
        params=[
            ('SynDecay', 'scalar'),
            ('MemFactor', 'scalar'),
            ('Vrest', 'scalar'),
            ('Vreset', 'scalar'),
            ('Vthresh', 'scalar'),
        ],
        vars=[
            ('V', 'scalar'),
            ('G', 'scalar'),
            ('RefracTime', 'scalar'),
            ('RefracReset', 'scalar'),
        ],
        additional_input_vars=[('Vstim', 'scalar', 0.0)],
        sim_code="""
        if(RefracTime <= 0.0) {
            V += Vstim;
            V += MemFactor * (G - (V - Vrest));
            G = (G * SynDecay) + Isyn;
        }
        else {
            RefracTime -= dt;
        }
        """,
        threshold_condition_code='RefracTime <= 0.0 && V > Vthresh',
        reset_code="""
        V = Vreset;
        G = 0.0;
        RefracTime = RefracReset;
        """,
    )

    return source_model, lif_model


def _build_model(t_run_sec, n_run, experiment, model_data, exp_name, logger):
    """Construct, build, and load a PyGeNN model."""
    t_construct_start = time()
    GeNNModel, create_neuron_model, init_weight_update, init_postsynaptic = (
        _import_pygenn()
    )
    source_model, lif_model = _create_genn_models(create_neuron_model)

    num_neurons = len(model_data['df_comp'])
    params = MODEL_PARAMS
    delay_steps = max(1, int(round(params['tDelay'] / DT)))

    model = GeNNModel('float', exp_name.replace('.', '_'), backend='cuda')
    model.dt = DT
    model.batch_size = n_run
    model.timing_enabled = True

    seed = int(os.environ.get('GENN_SEED', '12345'))
    model.seed = seed

    refrac_reset = np.full(num_neurons, params['tRefrac'], dtype=np.float32)
    for idx in model_data['exc'] + model_data['exc2']:
        refrac_reset[idx] = 0.0

    neurons = model.add_neuron_population(
        'neurons',
        num_neurons,
        lif_model,
        {
            'SynDecay': 1.0 - (DT / params['tauSyn']),
            'MemFactor': DT / params['tauMem'],
            'Vrest': params['vRest'],
            'Vreset': params['vReset'],
            'Vthresh': params['vThreshold'],
        },
        {
            'V': np.full(num_neurons, params['v0'], dtype=np.float32),
            'G': np.zeros(num_neurons, dtype=np.float32),
            'RefracTime': np.zeros(num_neurons, dtype=np.float32),
            'RefracReset': refrac_reset,
        },
    )
    neurons.spike_recording_enabled = True

    recurrent = model.add_synapse_population(
        'recurrent',
        'SPARSE',
        neurons,
        neurons,
        init_weight_update(
            'StaticPulse',
            {},
            {'g': model_data['weights'].astype(np.float32, copy=False)},
        ),
        init_postsynaptic('DeltaCurr'),
    )
    recurrent.set_sparse_connections(model_data['pre'], model_data['post'])
    recurrent.axonal_delay_steps = delay_steps

    stim_indices = list(model_data['exc'])

    if stim_indices:
        source = model.add_neuron_population(
            'poisson_source',
            len(stim_indices),
            source_model,
            {
                'Rate': float(experiment['stim_rate']),
                'DT': DT,
            },
            {'Spike': np.zeros(len(stim_indices), dtype=np.float32)},
        )
        stim_weight = np.full(
            len(stim_indices),
            params['wScale'] * params['scalePoisson'],
            dtype=np.float32,
        )
        stim_syn = model.add_synapse_population(
            'stimulation',
            'SPARSE',
            source,
            neurons,
            init_weight_update('StaticPulse', {}, {'g': stim_weight}),
            init_postsynaptic('DeltaCurr'),
        )
        stim_syn.set_sparse_connections(
            np.arange(len(stim_indices), dtype=np.uint32),
            np.asarray(stim_indices, dtype=np.uint32),
        )
        stim_syn.post_target_var = 'Vstim'

    construct_time = time() - t_construct_start

    build_dir = output_dir / 'genn' / exp_name.replace('.', '_')
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    logger.log("Building GeNN CUDA model...")
    t_build_start = time()
    model.build(str(build_dir), always_rebuild=True)
    build_time = time() - t_build_start

    num_steps = int(t_run_sec * 1000.0 / DT)
    recording_window = _recording_window_steps(num_steps, n_run)
    logger.log("Loading GeNN model...")
    t_load_start = time()
    model.load(num_recording_timesteps=recording_window)
    load_time = time() - t_load_start

    return (
        model, neurons, construct_time, build_time, load_time,
        delay_steps, recording_window,
    )


# ============================================================================
# Benchmark Functions
# ============================================================================

def run_single_benchmark(t_run_sec, n_run, experiment, logger,
                         run_idx=None, total_runs=None,
                         run_label=None, round_idx=None):
    """Run one GeNN benchmark combination."""
    t_sim_ms = t_run_sec * 1000.0
    num_steps = int(t_sim_ms / DT)
    exp_name = f'genn_t{t_run_sec}s_n{n_run}'

    run_info = f"[{run_idx}/{total_runs}] " if run_idx else ""
    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"{run_info}BENCHMARK: t_run={t_run_sec}s, n_run={n_run}")
    logger.log_raw("=" * 80)
    logger.log("Device: GPU (GeNN CUDA)")
    logger.log(f"Steps: {num_steps} (dt={DT}ms)")
    logger.log(f"Experiment: {exp_name}")
    if run_label:
        logger.log(f"Run label: {run_label}")
    if round_idx is not None:
        logger.log(f"Round: {round_idx}")

    timings = {}

    try:
        t_setup_start = time()
        t_data_start = time()
        model_data = _load_model_data(experiment)
        timings['data_load'] = time() - t_data_start
        timings['model_setup_total'] = time() - t_setup_start

        logger.log(f"  Data loading:     {timings['data_load']:.3f}s")
        logger.log(
            f"  Neurons: {len(model_data['df_comp'])}, "
            f"Synapses: {len(model_data['weights'])}, Batch: {n_run}"
        )

        (
            model, neurons, construct_time, build_time, load_time,
            delay_steps, recording_window,
        ) = (
            _build_model(t_run_sec, n_run, experiment, model_data, exp_name, logger)
        )
        timings['model_creation'] = construct_time
        timings['device_build'] = build_time
        timings['model_load'] = load_time
        timings['network_creation_total'] = (
            timings['model_setup_total']
            + timings['model_creation']
            + timings['model_load']
        )

        logger.log(f"  Model creation:   {construct_time:.3f}s")
        logger.log(f"  Device build:     {build_time:.3f}s")
        logger.log(f"  Model load:       {load_time:.3f}s")
        logger.log(f"  Synaptic delay:   {delay_steps} timestep(s)")
        logger.log(f"  Recording window: {recording_window} timestep(s)")

        logger.log(f"Running simulation ({num_steps} steps, {n_run} trial(s) batched)...")

        spike_chunks = []
        timings['simulation_total'] = 0.0
        timings['result_collection'] = 0.0

        for chunk_start in range(0, num_steps, recording_window):
            chunk_end = chunk_start + recording_window
            t_simulation_start = time()
            for _ in range(recording_window):
                model.step_time()
            _synchronize_cuda()
            timings['simulation_total'] += time() - t_simulation_start

            t_collect_start = time()
            model.pull_recording_buffers_from_device()
            for batch_idx, (spike_times, spike_ids) in enumerate(
                neurons.spike_recording_data
            ):
                if len(spike_times) > 0:
                    spike_chunks.append(
                        (batch_idx, spike_times.copy(), spike_ids.copy())
                    )
            timings['result_collection'] += time() - t_collect_start

            if num_steps >= 10000:
                pct = chunk_end / num_steps * 100
                logger.log(
                    f"  Progress: {pct:.0f}% ({chunk_end}/{num_steps})"
                    f" - sim {timings['simulation_total']:.1f}s elapsed"
                )

        timings['simulation_avg_per_trial'] = timings['simulation_total'] / n_run
        logger.log(f"  Simulation time:  {timings['simulation_total']:.3f}s")

        genn_kernel_time = (
            model.neuron_update_time
            + model.presynaptic_update_time
            + model.postsynaptic_update_time
            + model.synapse_dynamics_time
        )
        timings['genn_kernel_time'] = genn_kernel_time
        logger.log(f"  GeNN kernel time: {genn_kernel_time:.3f}s")

        logger.log("Collecting results...")
        t_collect_start = time()

        frames = []
        for batch_idx, spike_times, spike_ids in spike_chunks:
            if len(spike_times) == 0:
                continue
            neuron_idx = spike_ids.astype(np.int64, copy=False)
            frames.append(
                pd.DataFrame(
                    {
                        't': spike_times.astype(np.float64, copy=False),
                        'time_ms': spike_times.astype(np.float64, copy=False),
                        'trial': np.full(len(spike_times), batch_idx, dtype=np.int16),
                        'neuron_index': neuron_idx,
                        'flywire_id': model_data['flywire_ids'][neuron_idx],
                        'exp_name': exp_name,
                    }
                )
            )

        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.DataFrame(
                {
                    't': [], 'time_ms': [], 'trial': [],
                    'neuron_index': [], 'flywire_id': [], 'exp_name': [],
                }
            )

        timings['result_collection'] += time() - t_collect_start

        t_save_start = time()
        path_save = get_spike_output_path(exp_name, run_label, round_idx)
        path_save.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path_save, compression='brotli')
        timings['result_save'] = time() - t_save_start
        model.unload()

        logger.log(f"  Collection:       {timings['result_collection']:.3f}s")
        logger.log(f"  Save to file:     {timings['result_save']:.3f}s")
        logger.log(f"  Output file:      {path_save}")

        timings['total_elapsed'] = (
            timings['network_creation_total']
            + timings['device_build']
            + timings['simulation_total']
            + timings['result_collection']
            + timings['result_save']
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

        n_active = df['flywire_id'].nunique() if len(df) > 0 else 0
        n_spikes = len(df)

        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': n_active,
            'n_spikes': n_spikes,
            'status': 'success',
            'timings': timings,
            'backend_key': 'genn',
            'experiment_name': experiment['name'],
            'experiment_key': experiment['key'],
            'run_label': run_label or '',
            'round': round_idx or '',
            'spike_path': str(path_save),
        }

        logger.log_raw("")
        logger.log_raw("-" * 60)
        logger.log("TIMING SUMMARY")
        logger.log_raw("-" * 60)
        logger.log(f"  Model setup/load:   {timings['network_creation_total']:>10.3f}s")
        logger.log(f"  Device build:       {timings['device_build']:>10.3f}s")
        logger.log(f"  Simulation:         {timings['simulation_total']:>10.3f}s")
        logger.log(
            f"  Result processing:  "
            f"{timings['result_collection'] + timings['result_save']:>10.3f}s"
        )
        logger.log(f"  -----------------------------------------")
        logger.log(f"  TOTAL ELAPSED:      {timings['total_elapsed']:>10.3f}s")
        logger.log_raw("")
        logger.log(
            f"  Simulated time:     {total_simulated_time:>10.1f}s "
            f"({n_run} x {t_run_sec}s)"
        )
        logger.log(
            f"  Realtime ratio (sim only): {timings['realtime_ratio']:>6.3f}x"
        )
        logger.log(
            f"  Realtime ratio (total):    "
            f"{timings['realtime_ratio_total']:>6.3f}x"
        )
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
            'backend_key': 'genn',
            'experiment_name': experiment['name'],
            'experiment_key': experiment['key'],
            'run_label': run_label or '',
            'round': round_idx or '',
        }

    return results


def run_all_benchmarks(t_run_values=None, n_run_values=None,
                       experiment=None, logger=None,
                       run_label=None, round_idx=None):
    """Run all GeNN benchmark combinations."""
    if t_run_values is None:
        t_run_values = T_RUN_VALUES_SEC
    if n_run_values is None:
        n_run_values = N_RUN_VALUES
    if experiment is None:
        experiment = get_experiment()

    backend_name = 'GeNN (GPU)'

    benchmarks = []
    for n_run in n_run_values:
        for t_run_sec in t_run_values:
            benchmarks.append((t_run_sec, n_run))

    total_runs = len(benchmarks)

    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"BENCHMARK SUITE: {backend_name}")
    logger.log_raw("=" * 80)
    logger.log("Device: GPU (GeNN CUDA)")
    logger.log(f"t_run values: {t_run_values} seconds")
    logger.log(f"n_run values: {n_run_values}")
    if run_label:
        logger.log(f"Run label: {run_label}")
    if round_idx is not None:
        logger.log(f"Round: {round_idx}")
    logger.log(f"Total benchmarks: {total_runs}")
    logger.log_raw("=" * 80)

    all_results = []

    for run_idx, (t_run_sec, n_run) in enumerate(benchmarks, 1):
        result = run_single_benchmark(
            t_run_sec=t_run_sec,
            n_run=n_run,
            experiment=experiment,
            logger=logger,
            run_idx=run_idx,
            total_runs=total_runs,
            run_label=run_label,
            round_idx=round_idx,
        )
        all_results.append(result)
        save_result_csv(backend_name, result)

    print_summary_table(all_results, backend_name, logger)

    return all_results
