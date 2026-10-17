import mlx.core as mx
import numpy as np


def two_sum(left: mx.array, right: mx.array) -> tuple[mx.array, mx.array]:
    total = left + right
    recovered = total - left
    residual = (left - (total - recovered)) + (right - recovered)
    return total, residual


def compensated_tree(values: mx.array) -> tuple[mx.array, mx.array]:
    high, low = values, mx.zeros_like(values)
    while high.shape[-1] > 1:
        total, residual = two_sum(high[..., ::2], high[..., 1::2])
        correction = (low[..., ::2] + low[..., 1::2]) + residual
        high, low = two_sum(total, correction)
    return high[..., 0], low[..., 0]


SCALE = 0.275


SCALE_HIGH = float(np.float32(SCALE))


SCALE_LOW = float(np.float32(SCALE - SCALE_HIGH))


def split(value: mx.array) -> tuple[mx.array, mx.array]:
    expanded = 4097.0 * value
    high = expanded - (expanded - value)
    return high, value - high


def two_product(left: mx.array, right: mx.array) -> tuple[mx.array, mx.array]:
    product = left * right
    left_high, left_low = split(left)
    right_high, right_low = split(right)
    residual = left_low * right_low - (
        ((product - left_high * right_high) - left_low * right_high)
        - left_high * right_low
    )
    return product, residual


def scaled_count_sum(
    count_high: mx.array, count_low: mx.array, initial: mx.array
) -> tuple[mx.array, mx.array, mx.array]:
    terms = [initial]
    for count in (count_high, count_low):
        for scale in (SCALE_HIGH, SCALE_LOW):
            terms.extend(two_product(count, mx.array(scale, dtype=mx.float32)))
    terms.extend([mx.zeros_like(initial)] * 7)
    high, low = compensated_tree(mx.stack(terms, axis=-1))
    result = mx.where((count_high == 0) & (count_low == 0), initial, high + low)
    return result, count_high, count_low
