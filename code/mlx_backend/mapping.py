import numpy as np
from numpy.typing import NDArray

from .models import (
    Connectome,
    DestinationBucket,
    DestinationGrouping,
    EdgeRows32,
    NeuronRows32,
    PaddedEdgeRows32,
)


def absolute_count_sums(connectome: Connectome) -> NDArray[np.int64]:
    sums = np.zeros(connectome.neuron_ids.size, dtype=np.int64)
    np.add.at(sums, connectome.destinations, np.abs(connectome.counts.astype(np.int64)))
    return sums


def group_destinations(connectome: Connectome) -> DestinationGrouping:
    order = np.argsort(connectome.destinations, kind='stable').astype(np.int32)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size, dtype=np.int32)
    degree = np.bincount(connectome.destinations, minlength=connectome.neuron_ids.size)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(degree)))
    return DestinationGrouping(EdgeRows32(order), inverse, offsets)


def bucket_destinations(connectome: Connectome) -> tuple[DestinationBucket, ...]:
    grouping = group_destinations(connectome)
    degree = np.diff(grouping.offsets)
    widths = np.fromiter(
        (1 << (max(1, int(value)) - 1).bit_length() for value in degree),
        dtype=np.int32,
        count=degree.size,
    )
    buckets: list[DestinationBucket] = []
    for width in np.unique(widths):
        targets = np.flatnonzero(widths == width).astype(np.int32)
        edge_ids = np.full((targets.size, width), -1, dtype=np.int32)
        counts = np.zeros((targets.size, width), dtype=np.float32)
        occupied = np.zeros((targets.size, width), dtype=np.bool_)
        for row, target in enumerate(targets):
            edges = grouping.edge_ids[
                grouping.offsets[target] : grouping.offsets[target + 1]
            ]
            edge_ids[row, : edges.size] = edges
            counts[row, : edges.size] = connectome.counts[edges]
            occupied[row, : edges.size] = True
        for values in (targets, edge_ids, counts, occupied):
            values.setflags(write=False)
        buckets.append(
            DestinationBucket(
                NeuronRows32(targets), PaddedEdgeRows32(edge_ids), counts, occupied
            )
        )
    return tuple(buckets)


def silence_sources(connectome: Connectome, silenced: tuple[int, ...]) -> Connectome:
    if any(index < 0 or index >= connectome.neuron_ids.size for index in silenced):
        raise ValueError('Silenced source index is outside the neuron ordering')
    mask = np.isin(connectome.sources, silenced)
    counts, weights = connectome.counts.copy(), connectome.weights_mv.copy()
    counts[mask], weights[mask] = 0, 0
    counts.setflags(write=False)
    weights.setflags(write=False)
    return Connectome(
        connectome.neuron_ids,
        connectome.sources,
        connectome.destinations,
        counts,
        weights,
    )
