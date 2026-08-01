"""
utils.py
--------
Common helper functions used across the adaptive modem project:
- Gaussian Q-function
- MSE / PSNR calculation (Eq. 4.1, 4.2 in reference paper)
- Bit <-> byte helpers for image encoding
"""

import numpy as np
from scipy.special import erfc


def qfunc(x):
    """
    Gaussian Q-function: Q(x) = 0.5 * erfc(x / sqrt(2))
    Used in all theoretical BER expressions (Eq. 1.1 - 1.4).
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * erfc(x / np.sqrt(2))


def compute_mse(original, received):
    """
    Mean Square Error between original and received image (Eq. 4.1).
    MSE = sum((Xij - Xij')^2) / (M*N)
    """
    original = original.astype(np.float64)
    received = received.astype(np.float64)
    return np.mean((original - received) ** 2)


def compute_psnr(original, received, max_val=255.0):
    """
    Peak Signal-to-Noise Ratio (Eq. 4.2).
    PSNR = 10 * log10(255^2 / MSE)
    Returns a large finite value instead of inf when MSE == 0.
    """
    mse = compute_mse(original, received)
    if mse == 0:
        return 100.0
    return 10.0 * np.log10((max_val ** 2) / mse)


def bytes_to_bits(byte_array):
    """Convert a numpy uint8 array (e.g. flattened image) into a bit array (0/1)."""
    return np.unpackbits(byte_array.astype(np.uint8))


def bits_to_bytes(bit_array):
    """Convert a bit array (0/1) back into a numpy uint8 byte array."""
    n_bits = len(bit_array)
    pad = (-n_bits) % 8
    if pad:
        bit_array = np.concatenate([bit_array, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bit_array.astype(np.uint8))


def pad_bits_to_multiple(bits, k):
    """Zero-pad a bit array so its length is a multiple of k (bits per symbol)."""
    n = len(bits)
    pad = (-n) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return bits, pad
