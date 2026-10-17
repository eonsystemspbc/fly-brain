import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import benchmark
import mlx_engine
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import run_mlx
from mlx_backend.experiments import EXPERIMENTS
from mlx_backend.models import Connectome, SimulationRun, SpikeEvents

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    spec = EXPERIMENTS[name]
    first = spec.first_class_channels
    if first is None:
        first = len(spec.activated_ids)
    return {
        "key": name,
        "name": name,
        "neu_exc": list(spec.activated_ids[:first]),
        "neu_exc2": list(spec.activated_ids[first:]),
        "neu_slnc": list(spec.silenced_ids),
        "stim_rate": spec.rates_hz[0] if first else 0,
        "stim_rate2": spec.rates_hz[first] if first < len(spec.rates_hz) else 0,
    }


def test_optional_imports_and_help_work_without_site_packages(tmp_path):
    script = f"""
import sys, runpy
sys.path.insert(0, {str(ROOT / "code")!r})
import benchmark, mlx_engine, run_mlx
assert not {{'numpy', 'pyarrow', 'mlx', 'torch', 'brian2'}} & sys.modules.keys()
sys.argv = ['main.py', '--help']
try:
    runpy.run_path({str(ROOT / "main.py")!r}, run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert not {{'numpy', 'pyarrow', 'mlx', 'torch', 'brian2'}} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--mlx" in result.stdout
    assert not list(tmp_path.iterdir())


def test_independent_reference_fixtures_have_exact_integer_spikes():
    for name, count in (("sugar", 16752), ("p9", 858)):
        fixture = ROOT / "tests/data/mlx-replay" / name
        with np.load(fixture / "spikes.npz", allow_pickle=False) as expected:
            assert set(expected.files) == {"neuron_rows", "steps", "flywire_ids"}
            assert all(expected[key].dtype == np.int64 for key in expected.files)
            assert all(expected[key].shape == (count,) for key in expected.files)
            assert np.all((expected["steps"] >= 0) & (expected["steps"] < 10000))


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], ["cpu", "gpu", "pytorch", "nestgpu", "genn"]),
        (["--mlx"], ["mlx"]),
        (["--brian2-cpu", "--mlx"], ["cpu", "mlx"]),
    ],
)
def test_cli_backend_selection_preserves_defaults(monkeypatch, flags, expected):
    module = importlib.util.spec_from_file_location("eon_main", ROOT / "main.py")
    main = importlib.util.module_from_spec(module)
    module.loader.exec_module(main)
    calls = []
    monkeypatch.setattr(main, "run_benchmarks", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setitem(sys.modules, "brian2cuda", types.ModuleType("brian2cuda"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--no_log_file", *flags])
    main.main()
    assert calls[0]["backends"] == expected
    assert calls[0]["t_run_values"] is None and calls[0]["n_run_values"] is None


def test_common_dispatch_forwards_round_and_experiment_without_loading_an_engine(
    monkeypatch,
):
    fake = types.ModuleType("run_mlx")
    calls = []
    fake.run_all_benchmarks = lambda **kwargs: (
        calls.append(kwargs) or [{"status": "success"}]
    )
    monkeypatch.setitem(sys.modules, "run_mlx", fake)
    result = benchmark.run_benchmarks(
        ["mlx"],
        [0.1],
        [1],
        benchmark.get_experiment("p9"),
        benchmark.BenchmarkLogger(),
        2,
        "label!",
        3,
    )
    assert len(result["mlx"]) == 2
    assert [call["round_idx"] for call in calls] == [3, 4]
    assert all(
        call["run_label"] == "label"
        and call["experiment"]["key"] == "p9"
        and call["t_run_values"] == [0.1]
        and call["n_run_values"] == [1]
        for call in calls
    )


@pytest.mark.parametrize("flag,value", [(["--disable-spike-io"], ""), ([], "1")])
def test_timing_only_mode_is_rejected_before_optional_imports(tmp_path, flag, value):
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "main.py"), "--mlx", "--no_log_file", *flag],
        cwd=tmp_path,
        env={**os.environ, benchmark.SPIKE_IO_ENV_VAR: value},
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 2
    assert "requires spike recording" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "duration,trials",
    [
        (0, 1),
        (float("nan"), 1),
        (float("inf"), 1),
        (True, 1),
        (0.00015, 1),
        (1, 0),
        (1, True),
        (1, 1.5),
    ],
)
def test_invalid_dimensions_are_rejected(duration, trials):
    with pytest.raises((TypeError, ValueError)):
        mlx_engine.dimensions(duration, trials)


@pytest.mark.parametrize(
    "name", ["sugar", "p9", "sugar-silenced", "two-class", "silent"]
)
def test_pinned_class_and_silencing_setup_is_preserved(name):
    assert mlx_engine.experiment_spec(config(name)) == EXPERIMENTS[name]


def test_changed_setup_and_unsupported_platform_precision_versions_fail(monkeypatch):
    changed = {**benchmark.get_experiment(), "stim_rate": 100}
    with pytest.raises(ValueError, match="differ"):
        mlx_engine.experiment_spec(changed)
    monkeypatch.setattr(mlx_engine.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="Apple silicon"):
        mlx_engine.configure()
    monkeypatch.setattr(mlx_engine.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(mlx_engine.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    with pytest.raises(ValueError, match="MLX_ENABLE_TF32"):
        mlx_engine.configure()
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(mlx_engine.importlib.metadata, "version", lambda name: "99")
    with pytest.raises(RuntimeError, match="mlx==0.32.3"):
        mlx_engine.configure()


def saved_input(directory):
    spec = EXPERIMENTS["sugar"]
    events = np.zeros((2, 32, 21), dtype=np.uint8)
    events[0, 1, 0] = events[1, 3, 2] = 1
    np.savez_compressed(directory / "stimulus.npz", events=events)
    pin = mlx_engine.input_pin()
    metadata = {
        "format_version": 1,
        "layout": ["trial", "step", "channel"],
        "shape": list(events.shape),
        "dtype": "|u1",
        "dt_ms": 0.1,
        "generator": "PCG64",
        "numpy": "1.26.4",
        "seed": 20261004,
        "trial_indices": [2, 4],
        "targets": list(range(21)),
        "activated_ids": list(spec.activated_ids),
        "activation_classes": list(spec.activation_classes),
        "rates_hz": list(spec.rates_hz),
        "amplitudes_mv": [68.75] * 21,
        "experiment": "sugar",
        "experiment_code": 0,
        "generator_code": 0,
        "silenced_ids": [],
        "canonical_event_sha256": hashlib.sha256(events.tobytes()).hexdigest(),
        "per_trial_event_sha256": [
            hashlib.sha256(row.tobytes()).hexdigest() for row in events
        ],
        "seed_tuples": [[20261004, 0, 2], [20261004, 0, 4]],
        "input_sha256": {
            "completeness": pin.completeness_sha256,
            "connectivity": pin.connectivity_sha256,
        },
        "artifact_sha256": hashlib.sha256(
            (directory / "stimulus.npz").read_bytes()
        ).hexdigest(),
    }
    (directory / "stimulus.json").write_text(json.dumps(metadata))
    return events, metadata


def test_saved_replay_retains_exact_prefix_trial_ids_and_seed(tmp_path):
    events, _ = saved_input(tmp_path)
    actual, ids, seed = mlx_engine.read_replay(
        tmp_path, EXPERIMENTS["sugar"], tuple(range(21)), 17, 1
    )
    np.testing.assert_array_equal(actual, events[:1, :17])
    assert ids == (2,) and seed == 20261004 and not actual.flags.writeable
    with pytest.raises(ValueError, match="horizon"):
        mlx_engine.read_replay(tmp_path, EXPERIMENTS["sugar"], tuple(range(21)), 33, 1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rates_hz", [100] * 21),
        ("activation_classes", [2] * 21),
        ("canonical_event_sha256", "bad"),
        ("trial_indices", [2, 2]),
        ("seed_tuples", [[20261004, 0, 0], [20261004, 0, 1]]),
        ("shape", [2, 31, 21]),
        ("per_trial_event_sha256", ["bad", "bad"]),
    ],
)
def test_saved_replay_rejects_setup_identity_and_byte_mismatches(
    tmp_path, field, value
):
    _, metadata = saved_input(tmp_path)
    metadata[field] = value
    (tmp_path / "stimulus.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        mlx_engine.read_replay(tmp_path, EXPERIMENTS["sugar"], tuple(range(21)), 17, 1)


def test_nonbinary_events_fail_even_with_consistent_hashes(tmp_path):
    events, metadata = saved_input(tmp_path)
    events[0, 0, 0] = 2
    np.savez_compressed(tmp_path / "stimulus.npz", events=events)
    metadata["artifact_sha256"] = hashlib.sha256(
        (tmp_path / "stimulus.npz").read_bytes()
    ).hexdigest()
    metadata["canonical_event_sha256"] = hashlib.sha256(events.tobytes()).hexdigest()
    metadata["per_trial_event_sha256"] = [
        hashlib.sha256(row.tobytes()).hexdigest() for row in events
    ]
    (tmp_path / "stimulus.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="event bytes"):
        mlx_engine.read_replay(tmp_path, EXPERIMENTS["sugar"], tuple(range(21)), 17, 1)


@pytest.mark.parametrize("empty", [False, True])
def test_export_and_both_comparison_loaders_preserve_units_ids_and_empty_schema(
    tmp_path, monkeypatch, empty
):
    import compare_ground_truth
    import compare_spike_outputs

    monkeypatch.setattr(benchmark, "path_res", tmp_path)
    monkeypatch.setattr(benchmark, "csv_path", tmp_path / "benchmark.csv")
    ids = np.array([720575940624963786, 720575940630233916], dtype=np.int64)
    c = Connectome(
        ids,
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.int32),
        np.array([], dtype=np.float64),
    )
    trial = np.array([] if empty else [0, 40000], dtype=np.int64)
    rows = np.array([] if empty else [0, 1], dtype=np.int64)
    steps = np.array([] if empty else [1, 999], dtype=np.int64)
    run = SimulationRun(SpikeEvents(trial, rows, steps), {}, 0, "host-export")
    name = "mlx_t0.1s_n2"
    table = run_mlx.spike_table(c, [run], name)
    assert table.schema.field("flywire_id").type == pa.int64()
    assert table.schema.field("trial").type == pa.int64()
    assert table.schema.field("t").type == pa.float64()
    path = benchmark.get_spike_output_path(name, "host", 1)
    path.parent.mkdir(parents=True)
    pq.write_table(table, path)
    gt = compare_ground_truth.load_spike_data("mlx", 0.1, 2, "host", 1)
    pair, _ = compare_spike_outputs.load_spikes("mlx", 0.1, 2, "host", 1)
    assert pair["trial"].dtype == np.int64
    assert pair["time_s"].dtype == np.float64
    np.testing.assert_array_equal(pair["trial"].to_numpy(), trial)
    np.testing.assert_array_equal(gt["flywire_id"].to_numpy(), ids[rows])
    np.testing.assert_array_equal(pair["flywire_id"].to_numpy(), ids[rows])
    np.testing.assert_allclose(gt["t"].to_numpy(), steps * 0.0001, rtol=0, atol=1e-17)
    np.testing.assert_allclose(
        pair["time_s"].to_numpy(), steps * 0.0001, rtol=0, atol=1e-17
    )
    result = {
        "t_run_sec": 0.1,
        "n_run": 2,
        "n_spikes": len(steps),
        "n_active_neurons": len(steps),
        "status": "success",
        "timings": {
            "network_creation_total": 1.23456,
            "device_build": 0,
            "simulation_total": 2.34567,
            "total_elapsed": 3.58023,
            "realtime_ratio": 0.0852634,
        },
        "backend_key": "mlx",
        "experiment_name": "host",
        "experiment_key": "host",
        "run_label": "host",
        "round": 1,
        "spike_path": str(path),
    }
    benchmark.save_result_csv("MLX (Apple GPU)", result)
    with (tmp_path / "benchmark.csv").open() as file:
        reader = csv.DictReader(file)
        csv_rows = list(reader)
        assert reader.fieldnames == benchmark.CSV_COLUMNS
    assert csv_rows[0]["setup_time"] == "1.235"
    manifest = list(csv.DictReader((tmp_path / "host/manifest.csv").open()))
    assert manifest[0]["spike_time_column"] == "time_ms"
    assert manifest[0]["spike_time_unit"] == "ms"


def test_existing_output_is_preserved_before_loading_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(mlx_engine, "configure", lambda: None)
    path = tmp_path / "existing.parquet"
    path.write_bytes(b"existing evidence")
    monkeypatch.setattr(run_mlx, "get_spike_output_path", lambda *args: path)
    with pytest.raises(FileExistsError, match="fresh run label"):
        run_mlx.run_single_benchmark(
            0.1, 1, benchmark.get_experiment(), benchmark.BenchmarkLogger()
        )
    assert path.read_bytes() == b"existing evidence"
