"""Eon benchmark entry points for the optional Apple MLX backend.

GPL-2.0-or-later. See mlx_backend and the independent regression fixtures.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from benchmark import (
    BACKEND_NAMES,
    N_RUN_VALUES,
    T_RUN_VALUES_SEC,
    BenchmarkLogger,
    get_experiment,
    get_spike_output_path,
    path_comp,
    path_con,
    print_summary_table,
    save_result_csv,
    spike_io_enabled,
)


def spike_table(connectome, runs, exp_name):
    import numpy as np
    import pyarrow as pa

    trial = np.concatenate([run.spikes.trials for run in runs])
    neurons = np.concatenate([run.spikes.neurons for run in runs])
    steps = np.concatenate([run.spikes.steps for run in runs])
    order = np.lexsort((neurons, steps, trial))
    seconds = steps[order].astype(np.float64) * 0.0001
    schema = pa.schema(
        [
            ("t", pa.float64()),
            ("time_ms", pa.float64()),
            ("trial", pa.int64()),
            ("neuron_index", pa.int64()),
            ("flywire_id", pa.int64()),
            ("exp_name", pa.string()),
        ]
    )
    return pa.table(
        {
            "t": seconds,
            "time_ms": seconds * 1000,
            "trial": trial[order],
            "neuron_index": neurons[order],
            "flywire_id": connectome.neuron_ids[neurons[order]],
            "exp_name": [exp_name] * len(seconds),
        },
        schema=schema,
    )


def run_single_benchmark(
    t_run_sec,
    n_run,
    experiment,
    logger,
    run_idx=None,
    total_runs=None,
    run_label=None,
    round_idx=None,
    *,
    seed=20261004,
    stimulus_path=None,
):
    """Run fresh singletons sequentially; n_run never selects a new batch mode.

    Saved input is the retained stimulus.json/stimulus.npz directory. Shorter
    horizons and trial counts select verified prefixes of that archive.
    """
    import mlx_engine

    started = perf_counter()
    if not spike_io_enabled():
        raise ValueError(
            "MLX does not support --disable-spike-io; spike recording is required"
        )
    steps = mlx_engine.dimensions(t_run_sec, n_run)
    mlx_engine.configure()
    exp_name = f"mlx_t{t_run_sec}s_n{n_run}"
    path = get_spike_output_path(exp_name, run_label, round_idx)
    sidecar = path.with_suffix(".mlx.json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(
            f"MLX output already exists; choose a fresh run label: {path}"
        )

    import resource

    import numpy as np
    import pyarrow.parquet as pq

    spec = mlx_engine.experiment_spec(experiment)
    connectome = mlx_engine.load_inputs(path_comp, path_con)
    inputs, silenced = mlx_engine.stimuli(
        connectome, spec, steps, n_run, seed, stimulus_path
    )
    setup = perf_counter() - started
    simulation = 0.0
    postprocessing = 0.0
    runs = []
    for index, stimulus in enumerate(inputs):
        logger.log(f"MLX {exp_name}: fresh trial {index + 1}/{n_run}")
        run_started = perf_counter()
        result = mlx_engine.execute(connectome, stimulus, silenced)
        engine_elapsed = perf_counter() - run_started
        setup += result.timings["setup_s"]
        simulation += result.timings["simulation_wall_s"]
        postprocessing += max(
            0.0,
            engine_elapsed
            - result.timings["setup_s"]
            - result.timings["simulation_wall_s"],
        )
        runs.append(result)
    collecting = perf_counter()
    table = spike_table(connectome, runs, exp_name)
    collection = postprocessing + perf_counter() - collecting
    path.parent.mkdir(parents=True, exist_ok=True)
    saving = perf_counter()
    with path.open("xb") as destination:
        pq.write_table(table, destination, compression="brotli")
    save_time = perf_counter() - saving
    elapsed = perf_counter() - started
    timings = {
        "network_creation_total": setup,
        "device_build": 0.0,
        "simulation_total": simulation,
        "simulation_avg_per_trial": simulation / n_run,
        "result_collection": collection,
        "result_save": save_time,
        "total_elapsed": elapsed,
        "realtime_ratio": t_run_sec * n_run / simulation,
        "realtime_ratio_total": t_run_sec * n_run / elapsed,
    }
    result = {
        "t_run_sec": t_run_sec,
        "n_run": n_run,
        "n_spikes": table.num_rows,
        "n_active_neurons": int(np.unique(table["flywire_id"].to_numpy()).size),
        "status": "success",
        "timings": timings,
        "backend_key": "mlx",
        "experiment_name": experiment["name"],
        "experiment_key": experiment["key"],
        "run_label": run_label or "",
        "round": round_idx or "",
        "spike_path": str(path),
    }
    pin = mlx_engine.input_pin()
    metadata = {
        "result": result,
        "execution_mode": "sequential-fresh-singletons",
        "device": runs[0].device_name,
        "dt_ms": 0.1,
        "seed": inputs[0].seed,
        "generator": "PCG64",
        "seed_tuples": [[x.seed, x.generator_code, x.trial_indices[0]] for x in inputs],
        "canonical_event_sha256": [x.sha256 for x in inputs],
        "stimulus_source": str(Path(stimulus_path).resolve())
        if stimulus_path
        else "canonical-generation",
        "activated_ids": list(spec.activated_ids),
        "activation_classes": list(spec.activation_classes),
        "rates_hz": list(spec.rates_hz),
        "silenced_ids": list(spec.silenced_ids),
        "input_sha256": {
            "completeness": pin.completeness_sha256,
            "connectivity": pin.connectivity_sha256,
        },
        "spike_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mlx_peak_allocation_bytes": max(x.peak_device_bytes for x in runs),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "per_trial_timings": [x.timings for x in runs],
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-metal", "numpy", "pyarrow")
        },
        "MLX_ENABLE_TF32": "0",
        "compilation": "disabled",
        "timing_scope": "Whole-loop simulation includes first-step evaluation and block collection. Final sorting and Parquet export are outside; total ends after Parquet writing, before JSON sidecar/logging.",
        "memory_scope": "Process ru_maxrss and MLX allocator peaks are distinct, non-additive scopes; neither establishes physical unified-memory usage.",
        "validation_scope": "Experimental singleton backend. P9 count equivalence remains unestablished; current singleton throughput is slower than Brian2. No full-network M5 batch/ten-second acceptance claim.",
    }
    with sidecar.open("x") as destination:
        json.dump(metadata, destination, indent=2, allow_nan=False)
        destination.write("\n")
    logger.log(f"MLX wrote {table.num_rows} spikes to {path}")
    return result


def run_all_benchmarks(
    t_run_values=None,
    n_run_values=None,
    experiment=None,
    logger=None,
    run_label=None,
    round_idx=None,
):
    if logger is None:
        logger = BenchmarkLogger()
    if not spike_io_enabled():
        raise ValueError(
            "MLX does not support --disable-spike-io; spike recording is required"
        )
    if experiment is None:
        experiment = get_experiment()
    times = T_RUN_VALUES_SEC if t_run_values is None else t_run_values
    counts = N_RUN_VALUES if n_run_values is None else n_run_values
    parameters = [(duration, count) for count in counts for duration in times]
    results = []
    for index, (duration, count) in enumerate(parameters, 1):
        result = run_single_benchmark(
            duration,
            count,
            experiment,
            logger,
            index,
            len(parameters),
            run_label,
            round_idx,
        )
        save_result_csv(BACKEND_NAMES["mlx"], result)
        results.append(result)
    print_summary_table(results, BACKEND_NAMES["mlx"], logger)
    return results
