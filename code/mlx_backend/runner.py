from time import perf_counter

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from . import core
from .bucketed import prepare
from .core import boolean_input, evaluate
from .models import (
    Connectome,
    NeuronRows64,
    SimulationRun,
    SpikeEvents,
    Stimulus,
    TrialIds64,
)


def run(
    connectome: Connectome,
    stimulus: Stimulus,
    silenced: tuple[int, ...],
    *,
    precision: str,
) -> SimulationRun:
    mx.reset_peak_memory()
    started = perf_counter()
    execution = prepare(connectome, stimulus.targets, silenced, precision)
    state = core.initial_state(execution.network, len(stimulus.trial_indices))
    with mx.stream(mx.gpu):
        evaluate(
            state.voltage_offset_mv,
            state.synaptic_mv,
            state.last_spike_step,
            state.queue,
        )
    setup_s = perf_counter() - started
    steps = stimulus.events.shape[1]
    first_step_s = simulation_s = collection_s = 0.0
    collected: list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]] = []
    trial_ids = np.array(stimulus.trial_indices, dtype=np.int64)
    started_loop = perf_counter()
    for begin in range(0, steps, 256):
        block: list[mx.array] = []
        for step in range(begin, min(begin + 256, steps)):
            started_step = perf_counter()
            with mx.stream(mx.gpu):
                events = boolean_input(stimulus.events[:, step, :].astype(np.bool_))
                state, trace = execution.advance(state, events)
                evaluate(
                    state.voltage_offset_mv,
                    state.synaptic_mv,
                    state.last_spike_step,
                    state.queue,
                    trace.spikes,
                )
            elapsed = perf_counter() - started_step
            if step == 0:
                first_step_s = elapsed
            else:
                simulation_s += elapsed
            block.append(trace.spikes)
        started_collection = perf_counter()
        with mx.stream(mx.gpu):
            raster = np.asarray(mx.stack(block), dtype=np.bool_)
        local_steps, local_trials, neurons = np.nonzero(raster)
        collected.append(
            (
                trial_ids[local_trials],
                neurons.astype(np.int64),
                (local_steps + begin).astype(np.int64),
            )
        )
        collection_s += perf_counter() - started_collection
        print(f'Simulated {min(begin + 256, steps)}/{steps} steps', flush=True)
    simulation_wall_s = perf_counter() - started_loop
    started_collection = perf_counter()
    trials = np.concatenate([part[0] for part in collected])
    neurons = np.concatenate([part[1] for part in collected])
    spike_steps = np.concatenate([part[2] for part in collected])
    order = np.lexsort((neurons, spike_steps, trials))
    spikes = SpikeEvents(
        TrialIds64(trials[order]), NeuronRows64(neurons[order]), spike_steps[order]
    )
    collection_s += perf_counter() - started_collection
    return SimulationRun(
        spikes,
        {
            'setup_s': setup_s,
            'first_step_s': first_step_s,
            'warm_simulation_s': simulation_s,
            'simulation_wall_s': simulation_wall_s,
            'collection_s': collection_s,
            'compilation_s': 0.0,
        },
        mx.get_peak_memory(),
        str(mx.device_info()['device_name']),
    )
