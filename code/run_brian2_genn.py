"""
Brian2GeNN benchmark runner for the Drosophila brain model.

Brian2GeNN is a Brian2 standalone device targeting GeNN.  It currently pins an
older Brian2 release than Brian2CUDA, so this backend is intended to be run from
the separate ``brain-fly-brian2genn`` environment documented in the README.
"""

import os
import shutil
import traceback
from pathlib import Path
from time import perf_counter as time

import pandas as pd

from brian2 import (
    Hz, Network, NeuronGroup, PoissonInput, SpikeMonitor, Synapses,
    device, mV, ms, prefs, set_device,
)

from benchmark import (
    T_RUN_VALUES_SEC, N_RUN_VALUES,
    output_dir, path_comp, path_con,
    get_experiment, get_spike_output_path, print_summary_table, save_result_csv,
)
from run_brian2_cuda import default_params


def _configure_environment(set_brian_prefs=False):
    """Make CUDA and GeNN 4.x discoverable for Brian2GeNN."""
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
    if not cuda_path:
        for candidate in (Path('/usr/local/cuda-12.5'), Path('/usr/local/cuda')):
            if candidate.exists():
                cuda_path = str(candidate)
                break

    if cuda_path:
        os.environ['CUDA_PATH'] = cuda_path
        os.environ['CUDA_HOME'] = cuda_path
        cuda_bin = str(Path(cuda_path) / 'bin')
        cuda_lib = str(Path(cuda_path) / 'lib64')
        if cuda_bin not in os.environ.get('PATH', '').split(os.pathsep):
            os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')
        if cuda_lib not in os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep):
            os.environ['LD_LIBRARY_PATH'] = (
                cuda_lib + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')
            )
        if set_brian_prefs and 'devices.genn.cuda_backend.cuda_path' in prefs:
            prefs.devices.genn.cuda_backend.cuda_path = cuda_path

    genn_path = (
        os.environ.get('BRIAN2GENN_GENN_PATH')
        or os.environ.get('GENN_PATH')
    )
    if not genn_path:
        candidate = Path.home() / '.local' / 'src' / 'genn-4.9.0'
        if candidate.exists():
            genn_path = str(candidate)

    if genn_path:
        os.environ['GENN_PATH'] = genn_path
        genn_bin = str(Path(genn_path) / 'bin')
        if genn_bin not in os.environ.get('PATH', '').split(os.pathsep):
            os.environ['PATH'] = genn_bin + os.pathsep + os.environ.get('PATH', '')
        if set_brian_prefs and 'devices.genn.path' in prefs:
            prefs.devices.genn.path = genn_path

    if set_brian_prefs and 'devices.genn.connectivity' in prefs:
        prefs.devices.genn.connectivity = 'SPARSE'


def _import_brian2genn():
    _configure_environment(set_brian_prefs=False)
    try:
        import brian2genn  # noqa: F401,WPS433 - registers the genn device
    except ImportError as exc:
        raise ImportError(
            "Brian2GeNN is not installed. Use the isolated "
            "`brain-fly-brian2genn` environment because Brian2GeNN 1.7.0 "
            "requires Brian2<2.6 while Brian2CUDA requires Brian2 2.8.0."
        ) from exc
    _configure_environment(set_brian_prefs=True)


def _cleanup_build_dir(build_dir, logger):
    """Remove generated Brian2GeNN code unless explicitly kept for debugging."""
    if os.environ.get('BRIAN2GENN_KEEP_BUILD', '0') == '1':
        logger.log(f"  Keeping build dir: {build_dir}")
        return
    shutil.rmtree(build_dir, ignore_errors=True)


def _create_network(params, exc, exc2, slnc, logger=None):
    """Create a Brian2GeNN-compatible Brian2 network."""
    t_start = time()

    t_load_start = time()
    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)
    t_load = time() - t_load_start

    brian2genn_eqs = params['eqs'] + """
        stim_weight  : volt
        stim_weight2 : volt
    """

    t_neurons_start = time()
    neu = NeuronGroup(
        N=len(df_comp), model=brian2genn_eqs, method='linear',
        threshold=params['eq_th'], reset=params['eq_rst'],
        refractory='rfc', name='default_neurons', namespace=params,
    )
    neu.v = params['v_0']
    neu.g = 0
    neu.rfc = params['t_rfc']
    neu.stim_weight = 0 * mV
    neu.stim_weight2 = 0 * mV
    t_neurons = time() - t_neurons_start

    t_synapses_start = time()
    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w',
                   delay=params['t_dly'], name='default_synapses')
    syn.connect(i=df_con['Presynaptic_Index'].values,
                j=df_con['Postsynaptic_Index'].values)
    syn.w = df_con['Excitatory x Connectivity'].values * params['w_syn']
    t_synapses = time() - t_synapses_start

    spk_mon = SpikeMonitor(neu)

    t_poisson_start = time()
    pois = []
    if exc:
        neu.stim_weight[exc] = params['w_syn'] * params['f_poi']
        neu.rfc[exc] = 0 * ms
        pois.append(
            PoissonInput(target=neu, target_var='v', N=1,
                         rate=params['r_poi'], weight='stim_weight')
        )
    if exc2:
        neu.stim_weight2[exc2] = params['w_syn'] * params['f_poi']
        neu.rfc[exc2] = 0 * ms
        pois.append(
            PoissonInput(target=neu, target_var='v', N=1,
                         rate=params['r_poi2'], weight='stim_weight2')
        )

    for i in slnc:
        syn.w[' {} == i'.format(i)] = 0 * mV
    t_poisson = time() - t_poisson_start

    timings = {
        'data_load': t_load,
        'neuron_creation': t_neurons,
        'synapse_creation': t_synapses,
        'poisson_inputs': t_poisson,
        'network_creation_total': time() - t_start,
    }

    return neu, syn, spk_mon, pois, df_comp, timings


def _build_network(t_run_sec, experiment, exp_name, logger, timings,
                   trial_idx=0, round_idx=None):
    """Create the Brian2 network and run Brian2GeNN's build-on-run path."""
    from brian2 import device as brian_device

    brian_device.reinit()
    brian_device.activate()

    _import_brian2genn()

    build_dir = output_dir / 'brian2genn' / exp_name.replace('.', '_')
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    set_device('genn', directory=str(build_dir), debug=False)

    params = dict(default_params)
    params['r_poi'] = experiment['stim_rate'] * Hz
    t_run = t_run_sec * 1000 * ms

    t_mapping_start = time()
    df_comp = pd.read_csv(path_comp, index_col=0)
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}
    i2flyid = {j: i for i, j in flyid2i.items()}
    exc = [flyid2i[n] for n in experiment['neu_exc']]
    exc2 = [flyid2i[n] for n in experiment['neu_exc2']]
    slnc = [flyid2i[n] for n in experiment['neu_slnc']]
    timings['id_mapping'] = time() - t_mapping_start
    logger.log(f"ID mapping:         {timings['id_mapping']:.3f}s")

    logger.log("Creating network...")
    t_network_start = time()
    neu, syn, spk_mon, poi_inp, _, network_timings = _create_network(
        params, exc, exc2, slnc, logger=logger,
    )
    timings.update(network_timings)

    net = Network(neu, syn, spk_mon, *poi_inp)
    timings['network_creation_total'] = time() - t_network_start

    logger.log(f"  Data loading:     {timings['data_load']:.3f}s")
    logger.log(f"  Neuron creation:  {timings['neuron_creation']:.3f}s")
    logger.log(f"  Synapse creation: {timings['synapse_creation']:.3f}s")
    logger.log(f"  Poisson inputs:   {timings['poisson_inputs']:.3f}s")
    logger.log(f"  Total network:    {timings['network_creation_total']:.3f}s")

    base_seed = int(os.environ.get('BRIAN2GENN_SEED', '12345'))
    seed_offset = int(trial_idx)
    if round_idx is not None:
        seed_offset += int(round_idx) * 100000
    trial_seed = base_seed + seed_offset
    device.insert_code('main', f'srand({trial_seed});')

    logger.log("Building and running Brian2GeNN/GeNN CUDA executable...")
    t_build_run_start = time()
    net.run(duration=t_run)
    build_run_wall = time() - t_build_run_start

    first_sim_time = getattr(device, '_last_run_time', None)
    if first_sim_time is None:
        first_sim_time = build_run_wall

    timings['device_build'] = max(0.0, build_run_wall - first_sim_time)
    timings['simulation_total'] = first_sim_time
    timings['simulation_wall_total'] = build_run_wall
    timings['spike_extraction_total'] = 0.0

    t_extract_start = time()
    first_spikes = {k: v for k, v in spk_mon.spike_trains().items() if len(v)}
    extract_time = time() - t_extract_start
    timings['spike_extraction_total'] += extract_time

    logger.log(f"  Build/codegen wall: {timings['device_build']:.3f}s")
    logger.log(f"  First run time:     {first_sim_time:.3f}s")
    logger.log(f"  First extraction:   {extract_time:.3f}s")

    _cleanup_build_dir(build_dir, logger)

    return i2flyid, timings, first_spikes


def run_single_benchmark(t_run_sec, n_run, experiment, logger,
                         run_idx=None, total_runs=None,
                         run_label=None, round_idx=None):
    """Run a single Brian2GeNN benchmark."""
    exp_name = f'brian2genn_t{t_run_sec}s_n{n_run}'

    run_info = f"[{run_idx}/{total_runs}] " if run_idx else ""
    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"{run_info}BENCHMARK: t_run={t_run_sec}s, n_run={n_run}")
    logger.log_raw("=" * 80)
    logger.log("Device: GPU (Brian2GeNN)")
    logger.log(f"Experiment: {exp_name}")
    if run_label:
        logger.log(f"Run label: {run_label}")
    if round_idx is not None:
        logger.log(f"Round: {round_idx}")

    timings = {
        'id_mapping': 0.0,
        'network_creation_total': 0.0,
        'device_build': 0.0,
        'simulation_total': 0.0,
        'simulation_wall_total': 0.0,
        'spike_extraction_total': 0.0,
    }

    try:
        simulation_results = []
        i2flyid = None
        logger.log(
            f"Running {n_run} independent Brian2GeNN trial(s); "
            "Brian2GeNN rebuilds for each trial."
        )
        for trial_idx in range(n_run):
            trial_label = f"{trial_idx + 1}/{n_run}"
            build_exp_name = (
                exp_name if n_run == 1
                else f"{exp_name}_trial{trial_idx + 1:02d}"
            )
            logger.log_raw("")
            logger.log(f"  Trial {trial_label}: build/run start")
            trial_timings = {}
            trial_i2flyid, trial_timings, trial_spikes = _build_network(
                t_run_sec, experiment, build_exp_name, logger, trial_timings,
                trial_idx=trial_idx, round_idx=round_idx,
            )
            if i2flyid is None:
                i2flyid = trial_i2flyid
            simulation_results.append(trial_spikes)

            for key in (
                'id_mapping', 'network_creation_total', 'device_build',
                'simulation_total', 'simulation_wall_total',
                'spike_extraction_total',
            ):
                timings[key] += trial_timings.get(key, 0.0)

            if n_run <= 5 or (trial_idx + 1) % 5 == 0:
                logger.log(
                    f"  Trial {trial_label}: "
                    f"sim {trial_timings.get('simulation_total', 0.0):.3f}s, "
                    f"build {trial_timings.get('device_build', 0.0):.3f}s, "
                    f"spikes {sum(len(v) for v in trial_spikes.values())}"
                )

        timings['simulation_avg_per_trial'] = timings['simulation_total'] / n_run
        logger.log(f"  Total simulation: {timings['simulation_total']:.3f}s")
        logger.log(f"  Spike extraction: {timings['spike_extraction_total']:.3f}s")
        logger.log(f"  Avg per trial:    {timings['simulation_avg_per_trial']:.3f}s")

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
            'neuron_index': ids,
            'flywire_id': [i2flyid[i] for i in ids],
            'exp_name': exp_name,
        })
        df.insert(1, 'time_ms', df['t'] * 1000.0)
        timings['result_collection'] = time() - t_collect_start

        t_save_start = time()
        path_save = get_spike_output_path(exp_name, run_label, round_idx)
        path_save.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path_save, compression='brotli')
        timings['result_save'] = time() - t_save_start
        logger.log(f"  Collection:       {timings['result_collection']:.3f}s")
        logger.log(f"  Save to file:     {timings['result_save']:.3f}s")
        logger.log(f"  Output file:      {path_save}")

        timings['total_elapsed'] = (
            timings['id_mapping']
            + timings['network_creation_total']
            + timings.get('device_build', 0)
            + timings['simulation_total']
            + timings.get('spike_extraction_total', 0)
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

        n_active = len(set(ids))
        n_spikes = len(df)

        results = {
            't_run_sec': t_run_sec,
            'n_run': n_run,
            'n_active_neurons': n_active,
            'n_spikes': n_spikes,
            'status': 'success',
            'timings': timings,
            'backend_key': 'brian2genn',
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
        logger.log(f"  Network creation:   {timings['network_creation_total']:>10.3f}s")
        logger.log(f"  Device build:       {timings.get('device_build', 0):>10.3f}s")
        logger.log(f"  Simulation:         {timings['simulation_total']:>10.3f}s")
        logger.log(f"  Spike extraction:   {timings['spike_extraction_total']:>10.3f}s")
        logger.log(
            f"  Result processing:  "
            f"{timings['result_collection'] + timings['result_save']:>10.3f}s"
        )
        logger.log("  -----------------------------------------")
        logger.log(f"  TOTAL ELAPSED:      {timings['total_elapsed']:>10.3f}s")
        logger.log_raw("")
        logger.log(
            f"  Simulated time:     {total_simulated_time:>10.1f}s "
            f"({n_run} x {t_run_sec}s)"
        )
        logger.log(f"  Realtime ratio (sim only): {timings['realtime_ratio']:>6.3f}x")
        logger.log(f"  Realtime ratio (total):    {timings['realtime_ratio_total']:>6.3f}x")
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
            'backend_key': 'brian2genn',
            'experiment_name': experiment['name'],
            'experiment_key': experiment['key'],
            'run_label': run_label or '',
            'round': round_idx or '',
        }

    return results


def run_all_benchmarks(t_run_values=None, n_run_values=None,
                       experiment=None, logger=None,
                       run_label=None, round_idx=None):
    """Run all Brian2GeNN benchmark combinations."""
    if t_run_values is None:
        t_run_values = T_RUN_VALUES_SEC
    if n_run_values is None:
        n_run_values = N_RUN_VALUES
    if experiment is None:
        experiment = get_experiment()

    backend_name = 'Brian2GeNN (GPU)'

    benchmarks = []
    for n_run in n_run_values:
        for t_run_sec in t_run_values:
            benchmarks.append((t_run_sec, n_run))

    total_runs = len(benchmarks)

    logger.log_raw("")
    logger.log_raw("=" * 80)
    logger.log(f"BENCHMARK SUITE: {backend_name}")
    logger.log_raw("=" * 80)
    logger.log("Device: GPU (Brian2GeNN)")
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
