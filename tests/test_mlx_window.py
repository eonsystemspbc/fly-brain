"""Window boundary and event-semantics checks on real Metal hardware."""

import hashlib
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("Metal is required", allow_module_level=True)

from mlx_backend import core, runner, window
from mlx_backend.accumulation import scaled_count_sum
from mlx_backend.models import Connectome, InputPin, Stimulus

pytestmark = pytest.mark.metal


def toy(counts=(400, -100, 200)):
    values = np.asarray(counts, dtype=np.int32)
    return Connectome(
        np.arange(3, dtype=np.int64),
        np.array([0, 0, 1], np.int32),
        np.array([1, 2, 2], np.int32),
        values,
        values.astype(np.float64) * 0.275,
    )


def stimulus(events, targets):
    return Stimulus(
        events,
        targets,
        (200.0,) * len(targets),
        tuple(range(events.shape[0])),
        0,
        0,
        hashlib.sha256(events.tobytes()).hexdigest(),
    )


@pytest.mark.parametrize(
    "steps,targets,silenced,trials,block",
    [
        (80, (0,), (), 1, 1),
        (77, (0, 0), (), 2, 2),
        (55, (0,), (0,), 1, 2),
        (37, (), (), 1, 1),
    ],
)
def test_window_spikes_match_oracle_across_boundaries(
    steps, targets, silenced, trials, block
):
    events = np.zeros((trials, steps, len(targets)), np.uint8)
    if targets:
        events[:, [0, 3, 16, 17, 18, 34, 35], :] = 1
        if trials == 2:
            events[1, 3, :] = 0
    s = stimulus(events, targets)
    c = toy()
    expected = runner.run(c, s, silenced, precision="0")
    actual = window.run_prepared(
        window.prepare(c, targets, silenced), s, block_windows=block
    )
    for name in ("trials", "neurons", "steps"):
        np.testing.assert_array_equal(
            getattr(actual.spikes, name), getattr(expected.spikes, name)
        )


def test_delivery_preserves_each_step_and_signed_counts():
    p = window.prepare(toy(), ())
    state = window.initial_state(p)
    bits = mx.array([1 | (1 << 17), 1 << 3, 0], dtype=mx.uint32)
    (arrivals,) = window.delivery_kernel()(
        inputs=[p.offsets, p.destinations, p.counts, bits],
        template=[("N", 3), ("B", 1), ("W", 18)],
        grid=(3, 1, 1),
        threadgroup=(3, 1, 1),
        output_shapes=[(18, 3)],
        output_dtypes=[mx.int32],
        init_value=0,
    )
    expected = np.zeros((18, 3), np.int32)
    expected[[0, 17], 1] = 400
    expected[[0, 17], 2] = -100
    expected[3, 2] = 200
    np.testing.assert_array_equal(np.asarray(arrivals), expected)
    assert np.count_nonzero(np.asarray(state.arrivals)) == 0


def test_unrepresentable_arrival_bounds_are_rejected():
    with pytest.raises(ValueError, match="arrival counts"):
        window.prepare(toy((2**24 + 1, 0, 0)), ())


def test_oversized_kernel_indices_are_rejected_before_allocation():
    p = window.prepare(toy(), ())
    s = stimulus(np.zeros((1, 1, 0), np.uint8), ())
    with pytest.raises(ValueError, match="safe Metal indexing"):
        window.run_prepared(replace(p, neurons=2**27), s)
    empty = np.broadcast_to(np.zeros((1, 1, 0), np.uint8), (1, 2**31, 0))
    with pytest.raises(ValueError, match="safe Metal indexing"):
        window.run_prepared(p, stimulus(empty, ()))


def test_cached_layout_preserves_input_identity_targets_and_silencing(monkeypatch):
    monkeypatch.setattr(window, "_LAYOUT_CACHE", OrderedDict())
    c = replace(toy(), pin=InputPin("completeness", "connectivity", 3, 3))
    first = window.prepare(c, (0,))
    assert window.prepare(replace(c), (0,)) is first
    silenced = window.prepare(c, (0,), (0,))
    np.testing.assert_array_equal(np.asarray(silenced.counts), [0, 0, 200])
    targeted = window.prepare(c, (1,))
    np.testing.assert_array_equal(np.asarray(targeted.refractory), [22, 0, 22])
    changed = replace(
        toy((100, -100, 200)), pin=replace(c.pin, connectivity_sha256="new")
    )
    np.testing.assert_array_equal(
        np.asarray(window.prepare(changed, (0,)).counts), [100, -100, 200]
    )
    assert len(window._LAYOUT_CACHE) == 2
    # Unverified mutable toy inputs never enter the persistent cache.
    unverified = toy()
    assert window.prepare(unverified, (0,)) is not window.prepare(unverified, (0,))


def test_recording_compacts_active_words_and_detects_overflow():
    words = [
        mx.array([0, 3, 0, 8], dtype=mx.uint32),
        mx.array([4, 0, 0, 0], dtype=mx.uint32),
    ]
    windows, indices, masks = window.collect_words(words)
    actual = sorted(zip(windows.tolist(), indices.tolist(), masks.tolist()))
    assert actual == [(0, 1, 3), (0, 3, 8), (1, 0, 4)]
    with pytest.raises(OverflowError, match="capacity"):
        window.collect_words(words, capacity=2)


def test_partial_windows_preserve_pending_arrivals_when_resumed():
    p = window.prepare(toy(), (0,))
    events = np.zeros((1, 36, 1), np.uint8)
    events[:, [0, 3, 16, 17], :] = 1
    device_events = mx.array(events.reshape(-1))
    whole = window.initial_state(p)
    split = window.initial_state(p)
    for length in (18, 18):
        whole, _ = window.advance(p, whole, device_events, 36, length)
    for length in (7, 11, 13, 5):
        split, _ = window.advance(p, split, device_events, 36, length)
    for name in ("voltage", "synaptic", "last_spike", "arrivals"):
        a, b = np.asarray(getattr(whole, name)), np.asarray(getattr(split, name))
        if a.dtype == np.float32:
            a, b = a.view(np.uint32), b.view(np.uint32)
        np.testing.assert_array_equal(a, b)


def test_compensated_step_is_bitwise_equal_on_random_and_boundary_states():
    n = 1_000_000
    rng = np.random.default_rng(20261017)
    voltage = rng.uniform(-100, 100, n).astype(np.float32)
    synaptic = rng.uniform(-1000, 1000, n).astype(np.float32)
    voltage[8:12] = [-7, -7, -7, np.nextafter(np.float32(-7), np.float32(-6))]
    synaptic[8:12] = [0, -0.0, np.nextafter(np.float32(0), np.float32(1)), 0]
    voltage[:8] = [
        0,
        -0.0,
        -7,
        np.nextafter(np.float32(0), np.float32(1)),
        np.nextafter(np.float32(0), np.float32(-1)),
        -52,
        45,
        7,
    ]
    last = rng.integers(-30, 1, n, dtype=np.int32)
    last[:8] = [-22, -21, -23, -22, -22, -22, -1, 0]
    last[8:12] = -22
    counts = rng.integers(-69948, 69949, n, dtype=np.int32)
    counts[:8] = [0, 0, -1, 1, 69948, -69948, 0, 1]
    arrivals = np.zeros((18, n), dtype=np.int32)
    arrivals[0] = counts
    v, g, l = mx.array(voltage), mx.array(synaptic), mx.array(last)
    available = -l >= 22
    ev, eg = core.integrate(v, g, available)
    spikes = core.threshold_spikes(ev, available)
    receiving = available & ~spikes
    summed, _, _ = scaled_count_sum(
        mx.array(counts.astype(np.float32)), mx.zeros((n,)), eg
    )
    eg = mx.where(spikes, 0, mx.where(receiving, summed, eg))
    ev = mx.where(spikes, -7, ev)
    el = mx.where(spikes, 0, l)
    actual = window.neuron_kernel(True)(
        inputs=[
            v,
            g,
            l,
            mx.array(arrivals),
            mx.zeros((1,), dtype=mx.uint8),
            mx.zeros((n + 1,), dtype=mx.int32),
            mx.zeros((1,), dtype=mx.int32),
            mx.full((n,), 22, dtype=mx.int32),
            mx.array([0, 1, 1], dtype=mx.int32),
        ],
        template=[("N", n), ("B", 1), ("C", 0)],
        grid=(n, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(n,)] * 4,
        output_dtypes=[mx.float32, mx.float32, mx.int32, mx.uint32],
    )
    for reference, candidate in zip((ev, eg, el, spikes.astype(mx.uint32)), actual):
        a, b = np.asarray(reference), np.asarray(candidate)
        if a.dtype == np.float32:
            a, b = a.view(np.uint32), b.view(np.uint32)
        np.testing.assert_array_equal(a, b)


def test_real_network_midrun_window_preserves_state_and_pending_arrivals():
    import mlx_engine
    from mlx_backend.bucketed import prepare as prepare_oracle
    from mlx_backend.experiments import EXPERIMENTS

    root = Path(__file__).resolve().parents[1]
    c = mlx_engine.load_inputs(
        root / "data/2025_Completeness_783.csv",
        root / "data/2025_Connectivity_783.parquet",
    )
    ss, silenced = mlx_engine.stimuli(
        c,
        EXPERIMENTS["sugar"],
        10000,
        1,
        20261004,
        root / "tests/data/mlx-replay/sugar",
    )
    stimulus = ss[0]
    oracle = prepare_oracle(c, stimulus.targets, silenced, "0")
    state = core.initial_state(oracle.network)
    for k in range(100):
        state, _ = oracle.advance(
            state, mx.array(stimulus.events[:, k, :].astype(np.bool_))
        )
        core.evaluate(
            state.voltage_offset_mv,
            state.synaptic_mv,
            state.last_spike_step,
            state.queue,
        )
    prepared = window.prepare(c, stimulus.targets, silenced)
    offsets = np.asarray(prepared.offsets)
    destinations, counts = (
        np.asarray(prepared.destinations),
        np.asarray(prepared.counts),
    )

    def pending(current):
        queue = np.asarray(current.queue)
        result = np.zeros((18, c.neuron_ids.size), np.int32)
        for k in range(18):
            active = np.flatnonzero(queue[(current.step + k) % core.QUEUE_SLOTS, 0])
            for source in active:
                begin, end = offsets[source : source + 2]
                np.add.at(result[k], destinations[begin:end], counts[begin:end])
        return result

    candidate = window.State(
        state.voltage_offset_mv.reshape(-1),
        state.synaptic_mv.reshape(-1),
        state.last_spike_step.reshape(-1),
        mx.array(pending(state)),
        state.step,
    )
    updated, bits = window.advance(
        prepared, candidate, mx.array(stimulus.events.reshape(-1)), 10000
    )
    expected_bits = np.zeros(c.neuron_ids.size, np.uint32)
    for k in range(18):
        state, trace = oracle.advance(
            state, mx.array(stimulus.events[:, state.step, :].astype(np.bool_))
        )
        core.evaluate(
            state.voltage_offset_mv,
            state.synaptic_mv,
            state.last_spike_step,
            state.queue,
            trace.spikes,
        )
        expected_bits |= np.asarray(trace.spikes)[0].astype(np.uint32) << k
    for a, b in (
        (state.voltage_offset_mv, updated.voltage),
        (state.synaptic_mv, updated.synaptic),
        (state.last_spike_step, updated.last_spike),
    ):
        a, b = np.asarray(a).reshape(-1), np.asarray(b)
        if a.dtype == np.float32:
            a, b = a.view(np.uint32), b.view(np.uint32)
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.asarray(updated.arrivals), pending(state))
    np.testing.assert_array_equal(np.asarray(bits), expected_bits)
