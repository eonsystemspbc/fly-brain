"""Source-driven Metal windows with exact arithmetic and bounded recording."""

from dataclasses import dataclass
from collections import OrderedDict
from functools import lru_cache
from time import perf_counter

import mlx.core as mx
import numpy as np

from . import core
from .accumulation import SCALE_HIGH, SCALE_LOW
from .models import (
    Connectome,
    NeuronRows64,
    SimulationRun,
    SpikeEvents,
    Stimulus,
    TrialIds64,
)

WINDOW = core.DELAY_STEPS
LAYOUT_VERSION = 1
_LAYOUT_CACHE = OrderedDict()


def _tree_source(width):
    """Expand the fixed reduction tree so Metal needs no per-thread arrays."""
    values = [(f"h[{i}]", "0.0f") for i in range(width)]
    lines = [f"template<> inline float fly_tree<{width}>(thread float* h) {{"]
    level = 0
    while len(values) > 1:
        reduced = []
        for k in range(0, len(values), 2):
            left, right = values[k : k + 2]
            name = f"p{level}_{k // 2}"
            lines.append(f"FlyPair {name} = fly_sum({left[0]}, {right[0]});")
            lines.append(
                f"FlyPair q{name} = fly_sum({name}.hi, "
                f"({left[1]} + {right[1]}) + {name}.lo);"
            )
            reduced.append((f"q{name}.hi", f"q{name}.lo"))
        values = reduced
        level += 1
    lines.append(f"return {values[0][0]} + {values[0][1]}; }}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Prepared:
    neurons: int
    targets: tuple[int, ...]
    offsets: mx.array
    destinations: mx.array
    counts: mx.array
    channel_offsets: mx.array
    channels: mx.array
    refractory: mx.array
    maximum_incoming_count: int


@dataclass(frozen=True)
class State:
    voltage: mx.array
    synaptic: mx.array
    last_spike: mx.array
    arrivals: mx.array
    step: int = 0


def prepare(connectome: Connectome, targets, silenced=()) -> Prepared:
    n = connectome.neuron_ids.size
    targets = tuple(int(i) for i in targets)
    silenced = tuple(sorted(set(int(i) for i in silenced)))
    key = (LAYOUT_VERSION, connectome.pin, targets, silenced)
    if connectome.pin is not None and key in _LAYOUT_CACHE:
        _LAYOUT_CACHE.move_to_end(key)
        return _LAYOUT_CACHE[key]
    if n <= 0 or any(i < 0 or i >= n for i in targets):
        raise ValueError("Invalid neuron count or input target")
    counts = connectome.counts.copy()
    if silenced:
        counts[np.isin(connectome.sources, silenced)] = 0
    bound = np.bincount(
        connectome.destinations, weights=np.abs(counts.astype(np.int64)), minlength=n
    )
    maximum = int(bound.max(initial=0))
    if maximum > 2**24:
        raise ValueError(
            "Window prototype requires exact float32 integer arrival counts"
        )
    if connectome.sources.size > np.iinfo(np.int32).max:
        raise ValueError("Connection indexing exceeds int32")
    order = np.argsort(connectome.sources, kind="stable")
    offsets = np.concatenate(
        ([0], np.cumsum(np.bincount(connectome.sources, minlength=n)))
    ).astype(np.int32)
    channels = np.argsort(targets, kind="stable").astype(np.int32)
    channel_offsets = np.concatenate(
        ([0], np.cumsum(np.bincount(targets, minlength=n)))
    ).astype(np.int32)
    refractory = np.full(n, 22, dtype=np.int32)
    refractory[list(targets)] = 0
    with mx.stream(mx.gpu):
        result = Prepared(
            n,
            targets,
            mx.array(offsets),
            mx.array(
                connectome.destinations[order] if order.size else np.zeros(1, np.int32)
            ),
            mx.array(counts[order] if order.size else np.zeros(1, np.int32)),
            mx.array(channel_offsets),
            mx.array(channels if channels.size else np.zeros(1, np.int32)),
            mx.array(refractory),
            maximum,
        )
        mx.eval(
            result.offsets,
            result.destinations,
            result.counts,
            result.channel_offsets,
            result.channels,
            result.refractory,
        )
    # Only the loader's hash-verified immutable inputs are reusable. Keep at
    # most two layouts, including target/refractory and silencing differences.
    if connectome.pin is not None:
        _LAYOUT_CACHE[key] = result
        while len(_LAYOUT_CACHE) > 2:
            _LAYOUT_CACHE.popitem(last=False)
    return result


def initial_state(prepared: Prepared, trials=1) -> State:
    size = prepared.neurons * trials
    with mx.stream(mx.gpu):
        state = State(
            mx.full((size,), -7, dtype=mx.float32),
            mx.zeros((size,), dtype=mx.float32),
            mx.full((size,), core.INITIAL_LAST_SPIKE, dtype=mx.int32),
            mx.zeros((WINDOW, size), dtype=mx.int32),
        )
        mx.eval(state.voltage, state.synaptic, state.last_spike, state.arrivals)
    return state


EXACT_HEADER = f"""
#pragma clang fp contract(off)
struct FlyPair {{ float hi; float lo; }};
inline FlyPair fly_sum(float a, float b) {{
    float s = a + b;
    float r = s - a;
    return {{s, (a - (s - r)) + (b - r)}};
}}
inline FlyPair fly_split(float x) {{
    float e = 4097.0f * x;
    float h = e - (e - x);
    return {{h, x - h}};
}}
inline FlyPair fly_product(float a, float b) {{
    float p = a * b;
    FlyPair x = fly_split(a), y = fly_split(b);
    float e = x.lo * y.lo - (((p - x.hi * y.hi) - x.lo * y.hi) - x.hi * y.lo);
    return {{p, e}};
}}
template<int L> inline float fly_tree(thread float* h);
{_tree_source(8)}
{_tree_source(16)}
inline float fly_voltage(float v, float g) {{
    FlyPair centered = fly_sum(v, 7.0f);
    FlyPair p = fly_product(centered.hi, {core.VOLTAGE_INCREMENT:.18g}f);
    FlyPair q = fly_product(centered.lo, {core.VOLTAGE_INCREMENT:.18g}f);
    FlyPair r = fly_product(g, {core.COEFFICIENTS[2]:.18g}f);
    float h[8] = {{v, p.hi, p.lo, q.hi, q.lo, r.hi, r.lo, 0.0f}};
    return fly_tree<8>(h);
}}
inline float fly_arrival(float g, int count) {{
    if(count == 0) return g;
    float values[16] = {{0}};
    values[0] = g;
    int i = 1;
    #pragma unroll
    for(int c = 0; c < 2; ++c) {{
        float value = c == 0 ? float(count) : 0.0f;
        FlyPair p = fly_product(value, {SCALE_HIGH:.18g}f);
        FlyPair q = fly_product(value, {SCALE_LOW:.18g}f);
        values[i++] = p.hi; values[i++] = p.lo;
        values[i++] = q.hi; values[i++] = q.lo;
    }}
    return fly_tree<16>(values);
}}
"""


@lru_cache(maxsize=2)
def neuron_kernel(exact=True):
    integration = (
        "v = fly_voltage(v, g);"
        if exact
        else f"v = (v + (v + 7.0f) * {core.VOLTAGE_INCREMENT:.18g}f) + g * {core.COEFFICIENTS[2]:.18g}f;"
    )
    accumulation = (
        "g = fly_arrival(g, count);" if exact else "g = g + float(count) * 0.275f;"
    )
    source = f"""
    uint i = thread_position_in_grid.x;
    if(i >= N * B) return;
    uint row = i % N, trial = i / N;
    int start = params[0], length = params[1], steps = params[2];
    float v = voltage[i], g = synaptic[i];
    int last = last_spike[i];
    int rf = refractory[row];
    int cb = channel_offsets[row], ce = channel_offsets[row + 1];
    uint fired = 0;
    for(int k = 0; k < length; ++k) {{
        int step = start + k;
        bool available = step - last >= rf;
        if(available) {{
            // At rest with no synaptic current the voltage update is exactly -7.
            if(v != -7.0f || g != 0.0f) {{ {integration} }}
            g = g + g * {core.SYNAPTIC_INCREMENT:.18g}f;
        }}
        bool spike = available && v > 0.0f;
        if(available && !spike) {{
            int count = arrivals[k * N * B + i];
            {accumulation}
            for(int p = cb; p < ce; ++p) {{
                int ch = channels[p];
                if(events[(trial * steps + step) * C + ch]) v += 68.75f;
            }}
        }}
        if(spike) {{
            fired |= 1u << k;
            last = step;
            v = -7.0f;
            g = 0.0f;
        }}
    }}
    out_voltage[i] = v;
    out_synaptic[i] = g;
    out_last_spike[i] = last;
    spike_bits[i] = fired;
    """
    return mx.fast.metal_kernel(
        name="fly_window_neurons_exact" if exact else "fly_window_neurons_plain",
        input_names=[
            "voltage",
            "synaptic",
            "last_spike",
            "arrivals",
            "events",
            "channel_offsets",
            "channels",
            "refractory",
            "params",
        ],
        output_names=["out_voltage", "out_synaptic", "out_last_spike", "spike_bits"],
        source=source,
        header=EXACT_HEADER if exact else "#pragma clang fp contract(off)\n",
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def delivery_kernel():
    return mx.fast.metal_kernel(
        name="fly_window_delivery",
        input_names=["offsets", "destinations", "counts", "spike_bits"],
        output_names=["next_arrivals"],
        atomic_outputs=True,
        source="""
        uint i = thread_position_in_grid.x;
        if(i >= N * B) return;
        uint bits = spike_bits[i];
        if(bits == 0) return;
        uint row = i % N, trial = i / N;
        int begin = offsets[row], end = offsets[row + 1];
        for(uint k = 0; k < W; ++k) {
            if(!(bits & (1u << k))) continue;
            for(int edge = begin; edge < end; ++edge) {
                uint target = k * N * B + trial * N + destinations[edge];
                atomic_fetch_add_explicit(&next_arrivals[target], counts[edge], memory_order_relaxed);
            }
        }
        """,
        compile_options={"math_mode": "safe"},
    )


@lru_cache(maxsize=1)
def recording_kernel():
    return mx.fast.metal_kernel(
        name="fly_window_record",
        input_names=["words"],
        output_names=["counter", "entries"],
        atomic_outputs=True,
        source="""
        uint i = thread_position_in_grid.x;
        if(i >= WORDS) return;
        uint bits = words[i];
        if(bits == 0) return;
        uint pos = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        if(pos >= CAPACITY) return;
        atomic_store_explicit(&entries[2*pos], i, memory_order_relaxed);
        atomic_store_explicit(&entries[2*pos+1], bits, memory_order_relaxed);
        """,
        compile_options={"math_mode": "safe"},
    )


def collect_words(block, *, capacity=65536):
    """Transfer active words only; overflow is a hard error, never data loss."""
    words = mx.stack(block)
    capacity = min(words.size, capacity)
    if capacity < 1:
        raise ValueError("Recording capacity must be positive")
    counter, entries = recording_kernel()(
        inputs=[words],
        template=[("WORDS", words.size), ("CAPACITY", capacity)],
        grid=(words.size, 1, 1),
        threadgroup=(min(256, words.size), 1, 1),
        output_shapes=[(1,), (capacity, 2)],
        output_dtypes=[mx.uint32, mx.uint32],
        init_value=0,
    )
    count = int(counter.item())
    if count > capacity:
        raise OverflowError(
            f"Window recording capacity {capacity} exceeded by {count} active words"
        )
    active = np.asarray(entries[:count])
    windows, indices = np.divmod(active[:, 0].astype(np.int64), words.shape[1])
    return windows, indices, active[:, 1]


def advance(
    prepared: Prepared,
    state: State,
    events: mx.array,
    steps: int,
    length=WINDOW,
    *,
    exact=True,
):
    if not 1 <= length <= WINDOW or state.step + length > steps:
        raise ValueError("Invalid window length")
    n, b = prepared.neurons, state.voltage.size // prepared.neurons
    size = n * b
    with mx.stream(mx.gpu):
        v, g, last, bits = neuron_kernel(exact)(
            inputs=[
                state.voltage,
                state.synaptic,
                state.last_spike,
                state.arrivals,
                events,
                prepared.channel_offsets,
                prepared.channels,
                prepared.refractory,
                mx.array([state.step, length, steps], dtype=mx.int32),
            ],
            template=[("N", n), ("B", b), ("C", len(prepared.targets))],
            grid=(size, 1, 1),
            threadgroup=(min(256, size), 1, 1),
            output_shapes=[(size,)] * 4,
            output_dtypes=[mx.float32, mx.float32, mx.int32, mx.uint32],
        )
        (arrivals,) = delivery_kernel()(
            inputs=[prepared.offsets, prepared.destinations, prepared.counts, bits],
            template=[("N", n), ("B", b), ("W", WINDOW)],
            grid=(size, 1, 1),
            threadgroup=(min(256, size), 1, 1),
            output_shapes=[(WINDOW, size)],
            output_dtypes=[mx.int32],
            init_value=0,
        )
        if length < WINDOW:
            arrivals = mx.concatenate(
                [state.arrivals[length:], arrivals[:length]], axis=0
            )
    return State(v, g, last, arrivals, state.step + length), bits


def run_prepared(
    prepared: Prepared, stimulus: Stimulus, *, record=True, block_windows=64, exact=True
) -> SimulationRun:
    raw = stimulus.events
    if (
        raw.dtype != np.uint8
        or raw.ndim != 3
        or raw.shape[0] < 1
        or raw.shape[1] < 1
        or raw.shape[2] != len(prepared.targets)
        or np.any(raw > 1)
        or tuple(stimulus.targets) != prepared.targets
        or raw.shape[0] != len(stimulus.trial_indices)
        or block_windows < 1
    ):
        raise ValueError("Invalid canonical stimulus or recording block size")
    trials, steps, _ = raw.shape
    limit = np.iinfo(np.int32).max
    windows = (steps + WINDOW - 1) // WINDOW
    if (
        prepared.neurons * trials * max(WINDOW, min(block_windows, windows)) > limit
        or raw.size > limit
        or steps - core.INITIAL_LAST_SPIKE > limit
    ):
        raise ValueError("Window dimensions exceed safe Metal indexing")
    mx.reset_peak_memory()
    started = perf_counter()
    state = initial_state(prepared, trials)
    with mx.stream(mx.gpu):
        events = mx.array(raw.reshape(-1) if raw.size else np.zeros(1, np.uint8))
        mx.eval(events)
        setup_s = perf_counter() - started
        compiling = perf_counter()
        # Compile against disposable state; measured execution starts fresh below.
        probe, bits = advance(
            prepared, state, events, steps, min(WINDOW, steps), exact=exact
        )
        mx.eval(probe.voltage, probe.synaptic, probe.last_spike, probe.arrivals, bits)
        if record:
            sizes = {min(block_windows, windows)}
            if windows % block_windows:
                sizes.add(windows % block_windows)
            for size in sizes:
                collect_words([bits] * size, capacity=65536 * trials)
    jit_and_warmup_s = perf_counter() - compiling
    del probe, bits
    parts = []
    block, starts = [], []
    trial_ids = np.asarray(stimulus.trial_indices, dtype=np.int64)
    collection_s = 0.0
    started = perf_counter()
    for begin in range(0, steps, WINDOW):
        state, bits = advance(
            prepared, state, events, steps, min(WINDOW, steps - begin), exact=exact
        )
        mx.async_eval(
            state.voltage, state.synaptic, state.last_spike, state.arrivals, bits
        )
        if record:
            block.append(bits)
            starts.append(begin)
        if record and (len(block) == block_windows or state.step == steps):
            collecting = perf_counter()
            windows, indices, masks = collect_words(block, capacity=65536 * trials)
            for bit in range(WINDOW):
                select = (masks & (1 << bit)) != 0
                selected = indices[select]
                if not selected.size:
                    continue
                parts.append(
                    (
                        trial_ids[selected // prepared.neurons],
                        selected % prepared.neurons,
                        np.asarray(starts, dtype=np.int64)[windows[select]] + bit,
                    )
                )
            collection_s += perf_counter() - collecting
            block, starts = [], []
    mx.eval(state.voltage, state.synaptic, state.last_spike, state.arrivals)
    simulation_s = perf_counter() - started
    collecting = perf_counter()
    arrays = [
        np.concatenate([p[i] for p in parts]).astype(np.int64)
        if parts
        else np.array([], np.int64)
        for i in range(3)
    ]
    order = np.lexsort((arrays[1], arrays[2], arrays[0]))
    spikes = SpikeEvents(
        TrialIds64(arrays[0][order]), NeuronRows64(arrays[1][order]), arrays[2][order]
    )
    final_collection_s = perf_counter() - collecting
    return SimulationRun(
        spikes,
        dict(
            setup_s=setup_s,
            simulation_wall_s=simulation_s,
            collection_s=collection_s + final_collection_s,
            final_collection_s=final_collection_s,
            jit_and_warmup_s=jit_and_warmup_s,
        ),
        mx.get_peak_memory(),
        str(mx.device_info()["device_name"]),
    )


def run(connectome: Connectome, stimulus: Stimulus, silenced=(), *, precision="0"):
    if precision != "0":
        raise ValueError("Window engine requires the pinned precision configuration")
    started = perf_counter()
    prepared = prepare(connectome, stimulus.targets, silenced)
    layout_s = perf_counter() - started
    result = run_prepared(prepared, stimulus)
    result.timings["layout_s"] = layout_s
    result.timings["setup_s"] += layout_s
    return result
