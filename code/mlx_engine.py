"""Apple MLX adapter for the validated GPL-2.0-or-later runtime."""

import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path


def configure():
    """Check the supported platform before importing the optional Metal runtime."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX requires native Apple silicon macOS")
    if os.environ.setdefault("MLX_ENABLE_TF32", "0") != "0":
        raise ValueError("MLX requires MLX_ENABLE_TF32=0")
    for name, version in (
        ("mlx", "0.32.3"),
        ("mlx-metal", "0.32.3"),
        ("numpy", "1.26.4"),
        ("pyarrow", "25.0.1"),
    ):
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f"MLX requires {name}=={version}")


def dimensions(duration, trials):
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise TypeError("Duration must be a finite positive number")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Duration must be finite and positive")
    steps = round(duration / 0.0001)
    if steps < 1 or not math.isclose(
        steps * 0.0001, duration, abs_tol=1e-12, rel_tol=0
    ):
        raise ValueError("Duration must contain an integer number of 0.1 ms steps")
    if type(trials) is not int:
        raise TypeError("Trial count must be a positive integer")
    if trials < 1:
        raise ValueError("Trial count must be a positive integer")
    return steps


def input_pin():
    from mlx_backend.models import InputPin

    return InputPin(
        completeness_sha256="52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4",
        connectivity_sha256="efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
        neurons=138639,
        edges=15091983,
    )


def load_inputs(completeness, connectivity):
    from mlx_backend.inputs import load_connectome

    return load_connectome(Path(completeness), Path(connectivity), input_pin())


def experiment_spec(experiment):
    """Reject unsupported overrides instead of silently ignoring their semantics."""
    from mlx_backend.experiments import EXPERIMENTS

    key = experiment.get("key")
    if key not in EXPERIMENTS:
        raise ValueError(
            "MLX requires a pinned sugar, p9, sugar-silenced, two-class or silent configuration"
        )
    spec = EXPERIMENTS[key]
    first = tuple(experiment.get("neu_exc", ()))
    second = tuple(experiment.get("neu_exc2", ()))
    rates = (float(experiment.get("stim_rate", 0)),) * len(first) + (
        float(experiment.get("stim_rate2", 0)),
    ) * len(second)
    if (
        first + second != spec.activated_ids
        or rates != spec.rates_hz
        or tuple(experiment.get("neu_slnc", ())) != spec.silenced_ids
        or len(first)
        != (
            spec.first_class_channels
            if spec.first_class_channels is not None
            else len(spec.activated_ids)
        )
    ):
        raise ValueError(
            "MLX experiment targets, rates, classes or silencing differ from the pinned configuration"
        )
    return spec


def read_replay(directory, spec, targets, steps, trials):
    """Validate complete saved bytes before selecting a horizon/trial prefix."""
    import numpy as np

    directory = Path(directory)
    metadata = json.loads((directory / "stimulus.json").read_text())
    artifact = directory / "stimulus.npz"
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != metadata["artifact_sha256"]:
        raise ValueError("Saved MLX stimulus archive hash differs")
    pin = input_pin()
    expected = {
        "format_version": 1,
        "layout": ["trial", "step", "channel"],
        "dt_ms": 0.1,
        "generator": "PCG64",
        "numpy": "1.26.4",
        "dtype": "|u1",
        "targets": list(targets),
        "activated_ids": list(spec.activated_ids),
        "activation_classes": list(spec.activation_classes),
        "rates_hz": list(spec.rates_hz),
        "amplitudes_mv": [68.75] * len(targets),
        "experiment": spec.name,
        "experiment_code": spec.code,
        "generator_code": spec.generator_code,
        "silenced_ids": list(spec.silenced_ids),
        "input_sha256": {
            "completeness": pin.completeness_sha256,
            "connectivity": pin.connectivity_sha256,
        },
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Saved MLX stimulus setup differs from the pinned experiment")
    with np.load(artifact, allow_pickle=False) as saved:
        if saved.files != ["events"]:
            raise ValueError(
                "Saved MLX stimulus must contain only canonical event bits"
            )
        events = saved["events"]
    trial_ids = metadata["trial_indices"]
    seed = metadata["seed"]
    if (
        events.dtype != np.uint8
        or events.ndim != 3
        or list(events.shape) != metadata["shape"]
        or events.shape[2] != len(targets)
        or steps > events.shape[1]
        or trials > events.shape[0]
        or events.shape[0] != len(trial_ids)
        or type(seed) is not int
        or seed < 0
        or any(
            type(index) is not int or not 0 <= index <= 2**63 - 1 for index in trial_ids
        )
        or len(set(trial_ids)) != len(trial_ids)
        or np.any(events > 1)
        or hashlib.sha256(events.tobytes()).hexdigest()
        != metadata["canonical_event_sha256"]
        or [hashlib.sha256(row.tobytes()).hexdigest() for row in events]
        != metadata["per_trial_event_sha256"]
        or [[seed, spec.generator_code, index] for index in trial_ids]
        != metadata["seed_tuples"]
    ):
        raise ValueError(
            "Saved MLX stimulus shape, identity, horizon or event bytes differ"
        )
    selected = events[:trials, :steps]
    selected.setflags(write=False)
    return selected, tuple(trial_ids[:trials]), seed


def stimuli(connectome, spec, steps, trials, seed, replay=None):
    import numpy as np
    from mlx_backend.models import Stimulus, TrialId
    from mlx_backend.stimuli import neuron_indices, trial_events, uniforms

    if type(seed) is not int or seed < 0:
        raise ValueError("MLX seed must be a nonnegative integer")
    targets = neuron_indices(connectome, spec.activated_ids)
    if replay is not None:
        events, trial_ids, seed = read_replay(replay, spec, targets, steps, trials)
    else:
        events = None
        trial_ids = tuple(range(trials))
    result = []
    for row, trial in enumerate(trial_ids):
        bits = (
            events[row]
            if events is not None
            else trial_events(
                spec.rates_hz,
                uniforms((seed, spec.generator_code, trial), (steps, len(targets))),
            )
        )
        native = np.asarray(bits[None], dtype=np.uint8)
        native.setflags(write=False)
        result.append(
            Stimulus(
                native,
                targets,
                spec.rates_hz,
                (TrialId(trial),),
                seed,
                spec.generator_code,
                hashlib.sha256(native.tobytes()).hexdigest(),
            )
        )
    return result, neuron_indices(connectome, spec.silenced_ids)


def execute(connectome, stimulus, silenced):
    # The numerical implementation and its operation order are copied verbatim.
    from mlx_backend.runner import run

    return run(connectome, stimulus, silenced, precision="0")
