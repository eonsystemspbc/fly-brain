import numpy as np
from numpy.typing import NDArray

from .models import Connectome, FlyWireNeuronId, NeuronRow


def neuron_indices(
    connectome: Connectome, ids: tuple[FlyWireNeuronId, ...]
) -> tuple[NeuronRow, ...]:
    lookup = {
        FlyWireNeuronId(int(identifier)): NeuronRow(index)
        for index, identifier in enumerate(connectome.neuron_ids)
    }
    missing = set(ids) - lookup.keys()
    if missing:
        raise ValueError(
            f'Experiment neurons absent from pinned data: {sorted(missing)}'
        )
    return tuple(lookup[identifier] for identifier in ids)


def trial_events(
    rates_hz: tuple[float, ...], uniforms: NDArray[np.float64]
) -> NDArray[np.uint8]:
    if (uniforms.dtype, uniforms.ndim) != (np.float64, 2):
        raise ValueError('Stimulus uniforms require a native float64 matrix')
    probabilities = np.asarray(rates_hz, dtype=np.float64) * 0.0001
    return (uniforms < probabilities).astype(np.uint8)


def uniforms(seed: tuple[int, ...], size: int | tuple[int, int]) -> NDArray[np.float64]:
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(seed))))
    return generator.random(size)
