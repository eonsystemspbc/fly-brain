from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import numpy as np

from . import core
from .accumulation import compensated_tree, scaled_count_sum
from .core import boolean_input
from .mapping import absolute_count_sums, bucket_destinations, silence_sources
from .models import Connectome

Advance = Callable[[core.State, mx.array], tuple[core.State, core.StepTrace]]


@dataclass(frozen=True)
class Execution:
    network: core.Network
    advance: Advance


@dataclass(frozen=True)
class DeviceBucket:
    targets: mx.array
    edge_ids: mx.array
    counts: mx.array
    occupied: mx.array


@dataclass(frozen=True)
class Layout:
    buckets: tuple[DeviceBucket, ...]
    inverse_targets: mx.array
    edges: int
    exact_counts: bool


def make_layout(connectome: Connectome, *, exact_counts: bool = False) -> Layout:
    buckets = bucket_destinations(connectome)
    target_order = np.concatenate([bucket.targets for bucket in buckets])
    inverse = np.argsort(target_order).astype(np.int32)
    with mx.stream(mx.gpu):
        return Layout(
            tuple(
                DeviceBucket(
                    mx.array(bucket.targets),
                    mx.array(bucket.edge_ids),
                    mx.array(bucket.counts),
                    boolean_input(bucket.occupied),
                )
                for bucket in buckets
            ),
            mx.array(inverse),
            connectome.sources.size,
            exact_counts and bool(np.all(absolute_count_sums(connectome) <= 2**24)),
        )


def accumulate(
    layout: Layout, accepted: mx.array, initial: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    if (
        accepted.dtype != mx.bool_
        or accepted.ndim != 2
        or accepted.shape[1] != layout.edges
        or initial.shape != (accepted.shape[0], layout.inverse_targets.size)
        or initial.dtype != mx.float32
    ):
        raise ValueError('Accumulation requires Boolean edges and native trial state')
    with mx.stream(mx.gpu):
        padded_events = mx.concatenate(
            [mx.zeros((accepted.shape[0], 1), dtype=mx.bool_), accepted], axis=1
        )
        high_counts: list[mx.array] = []
        low_counts: list[mx.array] = []
        for bucket in layout.buckets:
            active = padded_events[:, bucket.edge_ids + 1] & bucket.occupied
            counts = mx.where(active, bucket.counts, 0)
            if layout.exact_counts:
                high = mx.sum(counts, axis=-1)
                low = mx.zeros_like(high)
            else:
                high, low = compensated_tree(counts)
            high_counts.append(high)
            low_counts.append(low)
        return scaled_count_sum(
            mx.concatenate(high_counts, axis=1)[:, layout.inverse_targets],
            mx.concatenate(low_counts, axis=1)[:, layout.inverse_targets],
            initial,
        )


def advance(
    network: core.Network, layout: Layout, state: core.State, events: mx.array
) -> tuple[core.State, core.StepTrace]:
    trials = state.voltage_offset_mv.shape[0]
    if events.shape != (trials, network.input_targets.size) or events.dtype != mx.bool_:
        raise ValueError('Events must be Boolean with shape (trials, input channels)')
    with mx.stream(mx.gpu):
        available = state.step - state.last_spike_step >= network.refractory_steps
        voltage, synaptic = core.integrate(
            state.voltage_offset_mv, state.synaptic_mv, available
        )
        pre_voltage, pre_synaptic = voltage, synaptic
        pre_physical_voltage = pre_voltage - 45
        spikes = core.threshold_spikes(pre_voltage, available)
        last_spike = mx.where(spikes, state.step, state.last_spike_step)
        receiving = available & ~spikes
        due = state.queue[state.step % core.QUEUE_SLOTS][:, network.sources]
        accepted = due & receiving[:, network.destinations]
        discarded = due & ~receiving[:, network.destinations]
        synaptic, _, _ = accumulate(layout, accepted, synaptic)
        synaptic = mx.where(receiving, synaptic, pre_synaptic)
        accepted_inputs = events & receiving[:, network.input_targets]
        neurons = mx.arange(network.neurons)
        for channel in range(network.input_targets.size):
            target = neurons == network.input_targets[channel]
            voltage = mx.where(
                accepted_inputs[:, channel, None] & target, voltage + 68.75, voltage
            )
        slots = mx.arange(core.QUEUE_SLOTS)[:, None, None]
        queue = mx.where(slots == state.step % core.QUEUE_SLOTS, False, state.queue)
        queue = mx.where(
            slots == (state.step + core.DELAY_STEPS) % core.QUEUE_SLOTS,
            spikes[None, :, :],
            queue,
        )
        trace = core.StepTrace(
            pre_physical_voltage,
            pre_synaptic,
            available,
            spikes,
            receiving,
            due,
            accepted,
            discarded,
            accepted_inputs,
            voltage - 45,
            synaptic,
            pre_voltage,
            voltage,
        )
        updated = core.State(
            mx.where(spikes, -7, voltage),
            mx.where(spikes, 0, synaptic),
            last_spike,
            queue,
            state.step + 1,
        )
        return updated, trace


def _components(
    connectome: Connectome,
    input_targets: tuple[int, ...],
    silenced: tuple[int, ...],
    precision: str,
    *,
    exact_counts: bool = False,
) -> tuple[core.Network, Layout]:
    mx.disable_compile()
    connectome = silence_sources(connectome, silenced)
    network = core.make_network(
        connectome.neuron_ids.size,
        connectome.sources,
        connectome.destinations,
        connectome.weights_mv,
        input_targets,
        precision=precision,
    )
    layout = make_layout(connectome, exact_counts=exact_counts)
    return network, layout


def prepare(
    connectome: Connectome,
    input_targets: tuple[int, ...],
    silenced: tuple[int, ...],
    precision: str,
    *,
    exact_counts: bool = True,
) -> Execution:
    network, layout = _components(
        connectome, input_targets, silenced, precision, exact_counts=exact_counts
    )
    return Execution(network, partial(advance, network, layout))
