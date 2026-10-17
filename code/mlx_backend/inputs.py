import csv
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray

from .mapping import absolute_count_sums
from .models import Connectome, FlyWireIds64, InputPin, NeuronRows32


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_connectome(
    completeness: Path, connectivity: Path, pin: InputPin
) -> Connectome:
    for path, expected in (
        (completeness, pin.completeness_sha256),
        (connectivity, pin.connectivity_sha256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f'Pinned input hash differs: {path}')
    with completeness.open(newline='') as source:
        rows = csv.reader(source)
        if next(rows) != ['', 'Completed']:
            raise ValueError('Completeness columns differ from the pinned contract')
        neuron_ids = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    if neuron_ids.size != pin.neurons or np.unique(neuron_ids).size != pin.neurons:
        raise ValueError('Neuron count or unique identifier order is invalid')
    columns = [
        'Presynaptic_ID',
        'Postsynaptic_ID',
        'Presynaptic_Index',
        'Postsynaptic_Index',
        'Connectivity',
        'Excitatory',
        'Excitatory x Connectivity',
    ]
    if not set(columns) <= set(pq.read_schema(connectivity).names):
        raise ValueError('Connectivity columns differ from the pinned contract')
    table = pq.read_table(connectivity, columns=columns)

    def integer_column(name: str) -> NDArray[np.int64]:
        column = table[name]
        values = column.to_numpy()
        if column.null_count or values.dtype != np.dtype(np.int64):
            raise ValueError(f'Connectivity column must contain non-null int64: {name}')
        if values.size != pin.edges:
            raise ValueError('Connection count differs from the pinned input')
        return np.asarray(values, dtype=np.int64)

    source_indices = integer_column('Presynaptic_Index')
    destination_indices = integer_column('Postsynaptic_Index')
    for name, indices in (
        ('Presynaptic_ID', source_indices),
        ('Postsynaptic_ID', destination_indices),
    ):
        if np.any(indices < 0) or np.any(indices >= pin.neurons):
            raise ValueError('Connection index is outside the neuron ordering')
        if not np.array_equal(neuron_ids[indices], integer_column(name)):
            raise ValueError(f'Identifier/index orientation differs: {name}')
    unsigned = integer_column('Connectivity')
    sign = integer_column('Excitatory')
    counts64 = integer_column('Excitatory x Connectivity')
    if np.any(unsigned < 0) or np.any(unsigned > 2**24):
        raise ValueError('Connectivity counts must be nonnegative and within 2^24')
    if not np.all(np.isin(sign, [-1, 1])):
        raise ValueError('Connectivity signs must be -1 or 1')
    if not np.array_equal(counts64, sign * unsigned):
        raise ValueError('Signed counts differ from sign times connectivity')
    if pin.neurons <= 0 or max(pin.neurons, pin.edges) > np.iinfo(np.int32).max:
        raise ValueError('Connectome identity arrays must fit int32 indexing')
    connectome = Connectome(
        FlyWireIds64(neuron_ids),
        NeuronRows32(source_indices.astype(np.int32)),
        NeuronRows32(destination_indices.astype(np.int32)),
        counts64.astype(np.int32),
        counts64.astype(np.float64) * pin.scale_mv,
    )
    if np.any(absolute_count_sums(connectome) > 2**40):
        raise ValueError(
            'Incoming absolute count sum exceeds the 2^40 arithmetic guard'
        )
    for array in (
        connectome.neuron_ids,
        connectome.sources,
        connectome.destinations,
        connectome.counts,
        connectome.weights_mv,
    ):
        array.setflags(write=False)
    return connectome
