"""Local CPU/MLX acceptance grid. Writes only to a fresh execution directory.

CPU n>1 uses concurrent C++ standalone processes, not serial CPU trials.
The common scope is recorded simulation plus spike materialization; executable
initialization/output and end-to-end harness timings are reported separately.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def key(case, duration, trials):
    return f"{case}:t{duration:g}:n{trials}"


def cpu_model(case, duration, directory):
    import benchmark
    import run_brian2_cuda as reference
    from brian2 import Hz, Network, defaultclock, device, ms, prefs, set_device

    device.reinit()
    set_device("cpp_standalone", build_on_run=False)
    prefs.devices.cpp_standalone.extra_make_args_unix = ["-j2"]
    device.headers.extend(["<chrono>", "<iomanip>", "<fstream>"])
    defaultclock.dt = 0.1 * ms
    experiment = benchmark.get_experiment(case)
    params = dict(reference.default_params)
    params["r_poi"] = experiment["stim_rate"] * Hz
    started = perf_counter()
    neu, syn, monitor, df, _ = reference.create_network(
        benchmark.path_comp, benchmark.path_con, params
    )
    lookup = {int(v): i for i, v in enumerate(df.index)}
    pois = reference.add_poisson_inputs(
        neu, [lookup[i] for i in experiment["neu_exc"]], [], params
    )
    network = Network(neu, syn, monitor, *pois)
    device.insert_code(
        "main",
        """
    { unsigned long seed_value; std::ifstream f(brian::results_dir + "seed.txt");
      f >> seed_value; if(!f) throw std::runtime_error("Missing benchmark seed");
      for(unsigned i=0; i<brian::_random_generators.size(); ++i)
          brian::_random_generators[i].seed(seed_value + i); }
    const auto bench_start = std::chrono::steady_clock::now();
    """,
    )
    network.run(duration * 1000 * ms)
    device.insert_code(
        "main",
        """
    { const auto bench_end = std::chrono::steady_clock::now();
      std::ofstream f(brian::results_dir + "loop-wall.txt");
      f << std::setprecision(17)
        << std::chrono::duration<double>(bench_start.time_since_epoch()).count() << " "
        << std::chrono::duration<double>(bench_end.time_since_epoch()).count(); }
    """,
    )
    setup = perf_counter() - started
    started = perf_counter()
    device.build(directory=str(directory), run=False, with_output=False)
    return monitor, setup, perf_counter() - started


def cpu_sample(directory, monitor, trials, sample_dir):
    from brian2 import get_device

    device = get_device()

    paths = [sample_dir / f"trial-{i}" for i in range(trials)]
    for i, p in enumerate(paths):
        p.mkdir(parents=True)
        (p / "seed.txt").write_text(str(20261004 + i))

    def invoke(p):
        subprocess.run(
            [str(directory / "main"), "--results_dir", str(p) + "/"],
            cwd=directory,
            check=True,
            capture_output=True,
        )

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=trials) as pool:
        list(pool.map(invoke, paths))
    harness = perf_counter() - started
    stamps = [np.loadtxt(p / "loop-wall.txt") for p in paths]
    loop = max(s[1] for s in stamps) - min(s[0] for s in stamps)
    started = perf_counter()
    counts = []
    for p in paths:
        device.results_dir = str(p) + "/"
        device.has_been_run = True
        for name in ("i", "t", "count", "N"):
            device.array_cache[monitor.variables[name]] = None
        spikes = {i: t for i, t in monitor.spike_trains().items() if len(t)}
        counts.append(sum(map(len, spikes.values())))
    collection = perf_counter() - started
    return dict(
        common_s=float(loop + collection),
        recorded_loop_s=float(loop),
        collection_s=collection,
        harness_simulation_s=harness,
        warm_total_s=harness + collection,
        spikes=counts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "mlx"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--guard-reference", type=Path)
    parser.add_argument(
        "--cases", nargs="+", choices=["sugar", "p9"], default=["sugar", "p9"]
    )
    parser.add_argument("--durations", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--trials", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--strategy", choices=["batch", "sequential"], default="batch")
    args = parser.parse_args()
    if args.rounds < 1 or any(n < 1 for n in args.trials):
        parser.error("Rounds and trial counts must be positive")
    import mlx_engine

    for duration in args.durations:
        mlx_engine.dimensions(duration, 1)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    guard = (
        json.loads(args.guard_reference.read_text()) if args.guard_reference else None
    )
    report = dict(
        started_at=datetime.now(timezone.utc).isoformat(),
        complete=False,
        backend=args.backend,
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        tracked_changes=subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        ).strip(),
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                ROOT / "code/mlx_backend/window.py",
                ROOT / "code/run_mlx.py",
                ROOT / "code/mlx_engine.py",
                Path(__file__),
            ]
        },
        scope="recorded simulation wall time plus spike materialization; final file export excluded",
        cpu_strategy="concurrent C++ standalone processes with seed 20261004+trial",
        mlx_strategy=args.strategy,
        measured_rounds=args.rounds,
        acceptance_measurement=args.rounds >= 5,
        input_pin=vars(mlx_engine.input_pin()),
        seed=20261004,
        machine=dict(
            platform=platform.platform(),
            processor=platform.processor(),
            load_average=os.getloadavg(),
            power=subprocess.check_output(["pmset", "-g", "batt"], text=True).strip(),
        ),
        guard_reference=str(args.guard_reference) if guard else None,
        cases={},
    )
    versions = (
        ["numpy", "brian2"] if args.backend == "cpu" else ["numpy", "mlx", "mlx-metal"]
    )
    report["versions"] = {n: importlib.metadata.version(n) for n in versions}
    if args.backend == "mlx":
        import mlx.core as mx

        report["machine"]["device"] = dict(mx.device_info())
    for case in args.cases:
        for duration in args.durations:
            if args.backend == "cpu":
                directory = args.out / f"cpp-{case}-t{duration:g}"
                monitor, setup, build = cpu_model(case, duration, directory)
            for trials in args.trials:
                name = key(case, duration, trials)
                samples = []
                for repeat in range(args.rounds + 1):
                    if args.backend == "cpu":
                        row = cpu_sample(
                            directory,
                            monitor,
                            trials,
                            directory / f"n{trials}-r{repeat}",
                        )
                        row.update(
                            setup_s=setup,
                            build_s=build,
                            cold_total_s=setup + build + row["warm_total_s"],
                        )
                    else:
                        import benchmark
                        import run_mlx

                        benchmark.path_res = args.out / "mlx-results"
                        result = run_mlx.run_single_benchmark(
                            duration,
                            trials,
                            benchmark.get_experiment(case),
                            benchmark.BenchmarkLogger(),
                            run_label=f"{case}-n{trials}-t{duration:g}",
                            round_idx=repeat + 1,
                            strategy=args.strategy,
                        )
                        metadata = json.loads(
                            Path(result["spike_path"])
                            .with_suffix(".mlx.json")
                            .read_text()
                        )
                        engines = metadata["engine_timings"]
                        common = sum(
                            t["simulation_wall_s"] + t.get("final_collection_s", 0)
                            for t in engines
                        )
                        row = dict(
                            common_s=common,
                            setup_s=result["timings"]["network_creation_total"],
                            build_s=result["timings"]["device_build"],
                            harness_simulation_s=result["timings"]["simulation_total"],
                            harness_total_s=result["timings"]["total_elapsed"],
                            result_collection_s=result["timings"]["result_collection"],
                            spikes=result["n_spikes"],
                            peak_bytes=metadata["mlx_peak_allocation_bytes"],
                            spike_path=result["spike_path"],
                        )
                    row.update(repeat=repeat, warmup=repeat == 0)
                    samples.append(row)
                    data = dict(samples=samples)
                    data["complete"] = repeat == args.rounds
                    data["verdict"] = "PENDING"
                    if repeat:
                        median = float(np.median([s["common_s"] for s in samples[1:]]))
                        data["median_common_s"] = median
                        if baseline:
                            b = baseline["cases"][name]["median_common_s"]
                            speedup = b / median
                            data.update(
                                baseline_s=b,
                                speedup=speedup,
                                performance_band="PASS"
                                if speedup >= 5
                                else "FLAG"
                                if speedup >= 2
                                else "FAIL",
                            )
                            if data["complete"]:
                                data["verdict"] = (
                                    "" if args.rounds >= 5 else "SMOKE_"
                                ) + data["performance_band"]
                        if guard and name in guard["cases"]:
                            previous = guard["cases"][name]["median_common_s"]
                            data["regression_fraction"] = median / previous - 1
                            if data["complete"]:
                                data["guard"] = (
                                    "REPEAT_REQUIRED"
                                    if median > previous * 1.1
                                    else "OK"
                                )
                    report["cases"][name] = data
                    (args.out / "results.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    print(
                        name,
                        repeat,
                        round(row["common_s"], 6),
                        data.get("verdict", ""),
                        flush=True,
                    )
    report["complete"] = True
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["final_load_average"] = os.getloadavg()
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
