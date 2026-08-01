"""
modulation.py
-------------
Modulator / demodulator implementations for the four schemes used in the
adaptive modem: BPSK, QPSK, 16-QAM and 64-QAM.

Each scheme provides:
    modulate(bits)   -> complex baseband symbols (unit average energy)
    demodulate(symbols) -> recovered bits (hard decision)

Bits-per-symbol matches Table 1 / Table 2.1 of the reference material:
    BPSK  : 1 bit/symbol
    QPSK  : 2 bits/symbol
    16QAM : 4 bits/symbol
    64QAM : 6 bits/symbol
"""

import numpy as np

BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6}


# ----------------------------- BPSK -----------------------------------
def bpsk_modulate(bits):
    return np.where(bits == 0, -1.0, 1.0).astype(complex)


def bpsk_demodulate(symbols):
    return (np.real(symbols) >= 0).astype(np.uint8)


# ----------------------------- QPSK -----------------------------------
def qpsk_modulate(bits):
    bits = bits.reshape(-1, 2)
    i = np.where(bits[:, 0] == 0, -1.0, 1.0)
    q = np.where(bits[:, 1] == 0, -1.0, 1.0)
    symbols = (i + 1j * q) / np.sqrt(2)
    return symbols


def qpsk_demodulate(symbols):
    i_bits = (np.real(symbols) >= 0).astype(np.uint8)
    q_bits = (np.imag(symbols) >= 0).astype(np.uint8)
    return np.column_stack([i_bits, q_bits]).reshape(-1)


# ----------------------------- 16-QAM ----------------------------------
def _gray_pam_levels(bits_per_dim, k):
    """Generate Gray-coded PAM levels for k bits per I or Q dimension."""
    m = 2 ** k
    levels = np.arange(-(m - 1), m, 2)  # e.g. k=2 -> [-3,-1,1,3]
    return levels


def _bits_to_gray_index(bits_group):
    """Convert a group of bits (Gray-coded) to PAM level index."""
    n = len(bits_group)
    gray = 0
    for b in bits_group:
        gray = (gray << 1) | int(b)
    # Gray -> binary
    binary = gray
    mask = binary >> 1
    while mask != 0:
        binary ^= mask
        mask >>= 1
    return binary


def _index_to_gray_bits(index, n):
    binary = index
    gray = binary ^ (binary >> 1)
    return [(gray >> (n - 1 - i)) & 1 for i in range(n)]


def qam_modulate(bits, k_total):
    """
    General square-QAM modulator using Gray coding per dimension.
    k_total: total bits per symbol (4 for 16QAM, 6 for 64QAM)
    """
    k_dim = k_total // 2  # bits per I or Q dimension
    levels = _gray_pam_levels(k_dim, k_dim)
    bits = bits.reshape(-1, k_total)
    i_bits = bits[:, :k_dim]
    q_bits = bits[:, k_dim:]

    i_idx = np.array([_bits_to_gray_index(row) for row in i_bits])
    q_idx = np.array([_bits_to_gray_index(row) for row in q_bits])

    i_val = levels[i_idx]
    q_val = levels[q_idx]

    symbols = (i_val + 1j * q_val).astype(complex)
    # Normalize to unit average energy
    avg_energy = np.mean(levels.astype(float) ** 2) * 2
    symbols /= np.sqrt(avg_energy)
    return symbols


def qam_demodulate(symbols, k_total):
    k_dim = k_total // 2
    levels = _gray_pam_levels(k_dim, k_dim)
    avg_energy = np.mean(levels.astype(float) ** 2) * 2
    scale = np.sqrt(avg_energy)

    i_val = np.real(symbols) * scale
    q_val = np.imag(symbols) * scale

    def nearest_index(vals):
        vals = vals.reshape(-1, 1)
        dist = np.abs(vals - levels.reshape(1, -1))
        return np.argmin(dist, axis=1)

    i_idx = nearest_index(i_val)
    q_idx = nearest_index(q_val)

    i_bits = np.array([_index_to_gray_bits(idx, k_dim) for idx in i_idx])
    q_bits = np.array([_index_to_gray_bits(idx, k_dim) for idx in q_idx])

    bits = np.concatenate([i_bits, q_bits], axis=1)
    return bits.reshape(-1).astype(np.uint8)


# --------------------------- Dispatch table -----------------------------
def modulate(scheme, bits):
    if scheme == "BPSK":
        return bpsk_modulate(bits)
    elif scheme == "QPSK":
        return qpsk_modulate(bits)
    elif scheme == "16QAM":
        return qam_modulate(bits, 4)
    elif scheme == "64QAM":
        return qam_modulate(bits, 6)
    else:
        raise ValueError(f"Unknown modulation scheme: {scheme}")


def demodulate(scheme, symbols):
    if scheme == "BPSK":
        return bpsk_demodulate(symbols)
    elif scheme == "QPSK":
        return qpsk_demodulate(symbols)
    elif scheme == "16QAM":
        return qam_demodulate(symbols, 4)
    elif scheme == "64QAM":
        return qam_demodulate(symbols, 6)
    else:
        raise ValueError(f"Unknown modulation scheme: {scheme}")
