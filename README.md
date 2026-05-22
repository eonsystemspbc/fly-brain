# Emulation of the *Drosophila Fly* Brain

Whole-brain leaky integrate-and-fire model of the adult fruit fly, built from the
[FlyWire](https://flywire.ai/) connectome (~138k neurons, ~5M synapses).
Activate and silence arbitrary neurons; observe downstream spike propagation.

Based on the paper
[*A leaky integrate-and-fire computational model based on the connectome of the
entire adult Drosophila brain reveals insights into sensorimotor processing*](https://www.biorxiv.org/content/10.1101/2023.05.02.539144v1)
(Shiu et al.).

## Usage

With this computational model, one can manipulate the neural activity of a set of _Drosophila_ neurons.
The output of the model is the spike times and rates of all affected neurons.

Two types of manipulations are currently implemented:
- *Activation*:
Neurons can be activated at a fixed frequency to model optogenetic activation.
This triggers Poisson spiking in the target neurons. 
Two sets of neurons with distinct frequencies can be defined.
- *Silencing*:
In addition to activation, a different set of neurons can be silenced to model optogenetic silencing.
This sets all synaptic connections to and from those neurons to zero.

The entrypoint is [main.py](main.py), which parses CLI arguments and calls
[code/benchmark.py](code/benchmark.py) -- the central orchestrator that dispatches
to framework-specific runners:
[run_brian2_cuda.py](code/run_brian2_cuda.py),
[run_pytorch.py](code/run_pytorch.py), and
[run_nestgpu.py](code/run_nestgpu.py).

```bash
# Run all 4 frameworks with default durations (0.1s–1000s) and trials (1,4,8,16,32)
python main.py

# Specific durations and trial count
python main.py --t_run 0.1 1 10 --n_run 1

# Single framework
python main.py --nestgpu --t_run 1 --n_run 1

# Combine frameworks
python main.py --brian2-cpu --pytorch --t_run 0.1 1 --n_run 1 4 8 16 32

# Five-round Nature-paper benchmark suite
# Uses the March grid: t_run=(0.1,1,10,100), n_run=(1,4,8,16,32), all 4 backends
python main.py --paper --run-label nature_2026_05
```

Results are incrementally saved to `data/benchmark-results.csv` as each
benchmark completes, with separate columns for setup time (loading, compilation)
and simulation time (the always-on cost). For repeated paper runs, the CSV keeps
the original March rows and appends new rows keyed by `run_label` and `round`;
the corresponding spike parquet path is recorded in `spike_path`.

Spike timing exports are written after the benchmarked simulation section so
file I/O does not contaminate `sim_time`. A labeled paper run writes partitioned
outputs like:

```text
data/results/nature_2026_05/
├── manifest.csv
├── round_01/
│   ├── brian2cpp_t1.0s_n1.parquet
│   ├── brian2cuda_t1.0s_n1.parquet
│   ├── pytorch_t1.0s_n1.parquet
│   └── nestgpu_t1.0s_n1.parquet
└── round_02/
```

Each spike parquet has one row per spike. The canonical timing column for new
exports is `time_ms`, with `trial`, `neuron_index`, `flywire_id`, and `exp_name`.
The legacy `t` column is kept for existing analysis scripts.

### Ground truth comparison

Brian2 (CPU) serves as the ground truth for neural accuracy: it implements the
canonical LIF model from
[Shiu et al. (Nature 2024)](https://www.nature.com/articles/s41586-024-07763-9),
which achieved 91% prediction accuracy against experimental _Drosophila_ data.
Each backend also saves per-neuron spike trains to `data/results/`, and a
comparison script measures how closely the other backends reproduce Brian2's
output:

```bash
python code/compare_ground_truth.py                  # default: t_run=1s, n_run=1
python code/compare_ground_truth.py --t_run 10 --n_run 4   # longer / averaged
python code/compare_ground_truth.py --run-label nature_2026_05 --round 1
```

This computes active-neuron overlap (Jaccard), per-neuron firing-rate
correlation, and spike-count ratios, and writes structured results to
`data/ground-truth-comparison.json`.

## Installation

### Conda environment

The `brain-fly` conda environment provides everything needed to run the
**Brian2**, **Brian2CUDA**, and **PyTorch** backends (including CUDA-enabled
PyTorch):

```bash
conda env create -f environment.yml
conda activate brain-fly
```

### NEST GPU

NEST GPU requires a separate build from source with a custom neuron model
(`user_m1`). This is only needed if you want to use the `--nestgpu` backend.

**Prerequisites:**

- **NVIDIA CUDA Toolkit** (12.x) — follow the
  [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
- **CMake** — `sudo apt install cmake` (or see
  [cmake.org](https://cmake.org/download/)).

**Steps:**

1. Clone NEST GPU:

```bash
git clone https://github.com/nest/nest-gpu
```

2. Copy the custom source files into the NEST GPU tree. You must replace `/path/to/nest-gpu` with your own local path:

```bash
cp scripts/nestgpu_source_files/src/user_m1.{h,cu}    /path/to/nest-gpu/src/
cp scripts/nestgpu_source_files/pythonlib/nestgpu.py   /path/to/nest-gpu/pythonlib/
```

   The patched `nestgpu.py` fixes weight array initialization (lines 2225-2227).

3. Build and install (set `-DCMAKE_CUDA_ARCHITECTURES` to match your GPU, e.g.
   `89` for RTX 4070):

```bash
cmake -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_INSTALL_PREFIX=$HOME/.nest-gpu-build \
      /path/to/nest-gpu
make -j$(nproc) && make install
```

For a full setup from a fresh Windows machine (WSL2 + CUDA + Miniconda), see
[scripts/setup_WSL_CUDA.sh](scripts/setup_WSL_CUDA.sh).

----

## Frameworks

| Framework | Backend | Status |
|---|---|---|
| **Brian2** | C++ standalone (multi-core CPU) | ready |
| **Brian2CUDA** | CUDA standalone (GPU) | ready |
| **PyTorch** | CUDA (GPU) | ready |
| **NEST GPU** | CUDA (GPU, custom `user_m1` neuron) | ready |

All four frameworks share the same data, model parameters, and folder structure.
A single conda environment (`brain-fly`) plus a system-level NEST GPU install
runs everything.

## Quickstart

```bash
# Create the conda environment (includes CUDA-enabled PyTorch)
conda env create -f environment.yml
conda activate brain-fly

# Run a 1-second benchmark on all backends
python main.py --t_run 1 --n_run 1 --no_log_file

# Specific backends (combinable)
python main.py --brian2-cpu                    # Brian2 CPU only
python main.py --brian2cuda-gpu               # Brian2CUDA GPU only
python main.py --pytorch                      # PyTorch only
python main.py --nestgpu                      # NEST GPU only
python main.py --pytorch --nestgpu            # PyTorch + NEST GPU

# Full benchmark suite (all durations, n_run=1,4,8,16,32, all backends)
python main.py

# Nature-paper suite: all backends, March parameter grid, 5 rounds
python main.py --paper --run-label nature_2026_05
```

### `main.py` options

| Flag | Description |
|---|---|
| *(default)* | Run all: Brian2 (CPU) → Brian2CUDA (GPU) → PyTorch → NEST GPU |
| `--brian2-cpu` | Brian2 C++ standalone (CPU) only |
| `--brian2cuda-gpu` | Brian2CUDA (GPU) only |
| `--pytorch` | PyTorch (GPU/CPU) only |
| `--nestgpu` | NEST GPU only |
| `--t_run` | Simulation duration(s) in seconds, e.g. `--t_run 0.1 1 10` |
| `--n_run` | Number of independent trials, e.g. `--n_run 1 4 8 16 32` |
| `--paper` | Run the paper suite: `t_run=[0.1,1,10,100]`, `n_run=[1,4,8,16,32]`, 5 rounds |
| `--rounds` | Repeat the full selected backend/parameter suite N times |
| `--round-start` | First round number to write, useful for resuming a labeled run |
| `--run-label` | Group repeated spike outputs under `data/results/<label>/` and append labeled CSV rows |
| `--log_file FILE` | Write log to file (default: `data/results/benchmarks.log`) |
| `--no_log_file` | Console output only |

Backend flags are combinable: `--brian2-cpu --pytorch` runs Brian2 CPU then PyTorch.

## Project structure

```
fly-brain/
├── main.py                     # Entrypoint (benchmark runner CLI)
├── environment.yml             # Conda env definition (brain-fly)
├── code/
│   ├── benchmark.py            # Orchestrator: config, logging, dispatcher
│   ├── run_brian2_cuda.py      # Brian2 / Brian2CUDA benchmark runner
│   ├── run_pytorch.py          # PyTorch benchmark runner (model + utils)
│   ├── run_nestgpu.py          # NEST GPU benchmark runner (subprocess per trial)
│   ├── compare_ground_truth.py # Compare backends against Brian2 (CPU) ground truth
│   └── paper-brian2/           # Original paper code (not used by benchmarks)
│       ├── model.py            # Core LIF network model (Brian2)
│       ├── utils.py            # Analysis helpers (load_exps, get_rate)
│       ├── example.ipynb       # Tutorial: activation, silencing, rate analysis
│       └── figures.ipynb       # Reproduce paper figures (uses archive 630 data)
├── data/
│   ├── 2025_Completeness_783.csv       # Neuron list (FlyWire v783)
│   ├── 2025_Connectivity_783.parquet   # Synapse connectivity (FlyWire v783)
│   ├── benchmark-results.csv           # Accumulated benchmark timings
│   ├── ground-truth-comparison.json   # Backend accuracy vs Brian2 (CPU)
│   ├── sez_neurons.pickle              # SEZ neuron subset (for figures)
│   ├── weight_coo.pkl                  # Cached sparse weights COO (gitignored)
│   ├── weight_csr.pkl                  # Cached sparse weights CSR (gitignored)
│   ├── archive/
│   │   ├── 2023_Completeness_630.csv   # Legacy v630 data
│   │   └── 2023_Connectivity_630.parquet
└── scripts/
    └── setup_WSL_CUDA.sh       # WSL2 + CUDA + Miniconda setup
```

## Data

The model uses FlyWire connectome data version **783** (public release).
Legacy version 630 data is kept in `data/archive/` for paper figure reproduction.

| File | Description | Size |
|---|---|---|
| `2025_Completeness_783.csv` | Neuron IDs and metadata | 3.2 MB |
| `2025_Connectivity_783.parquet` | Pre/post-synaptic indices + weights | 97 MB |
| `weight_coo.pkl` | Sparse weight matrix (COO), auto-generated by PyTorch | ~288 MB |
| `weight_csr.pkl` | Sparse weight matrix (CSR), auto-generated by PyTorch | ~289 MB |

## Architecture per framework

| | Brian2 / Brian2CUDA | PyTorch | NEST GPU |
|---|---|---|---|
| Build step | C++ / CUDA codegen + compile | None (eager mode) | None |
| Trial parallelism | Sequential (`device.run`) | Batched (`batch_size=n_run`) | Subprocess per trial (cannot reset in-process) |
| Weight format | Brian2 `Synapses` object | Sparse CSR tensor | Array-based `Connect` |
| Neuron model | Brian2 equations | Custom `nn.Module` classes | Custom CUDA kernel (`user_m1`) |
| Timestep | 0.1 ms | 0.1 ms | 0.1 ms |

## System requirements

- Linux (tested on Ubuntu 22.04 under WSL2 on Windows 11)
- NVIDIA GPU with CUDA 12.x (tested on RTX 4070)
- Miniconda / Anaconda
- NEST GPU compiled from source (for `--nestgpu` backend)
- `scripts/setup_WSL_CUDA.sh` documents the full setup from a fresh Windows machine
