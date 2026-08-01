"""
adaptive.py
-----------
Adaptive modulation switching logic, based directly on "Table 1: Summary
of Switching Levels" from the reference paper (also Table 2.1 in the
project proposal). For a given target BER, the SNR (gamma, dB) of the
current frame is mapped to a modulation scheme.
"""

import numpy as np

# Switching thresholds copied from Table 1 (SNR ranges in dB)
SWITCHING_LEVELS = {
    0.1: [  # Target BER = 0.1  (10%)
        (0, 2, "BPSK"),
        (3, 7, "QPSK"),
        (8, 14, "16QAM"),
        (15, 40, "64QAM"),
    ],
    0.01: [  # Target BER = 0.01 (1%)
        (0, 5, "BPSK"),
        (6, 11, "QPSK"),
        (12, 17, "16QAM"),
        (18, 40, "64QAM"),
    ],
    0.001: [  # Target BER = 0.001 (0.1%)
        (0, 7, "BPSK"),
        (8, 13, "QPSK"),
        (14, 19, "16QAM"),
        (20, 40, "64QAM"),
    ],
}

BPS = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6}


def select_modulation(snr_db, target_ber=0.001):
    """
    Select the modulation scheme for the given channel SNR (dB) and
    target BER, using the switching table (Table 1 / Table 2.1).
    SNR values below 0 use BPSK; above 40 use 64QAM (edge handling).
    """
    if target_ber not in SWITCHING_LEVELS:
        raise ValueError(f"target_ber must be one of {list(SWITCHING_LEVELS)}")

    levels = SWITCHING_LEVELS[target_ber]

    if snr_db < levels[0][0]:
        return levels[0][2]
    if snr_db > levels[-1][1]:
        return levels[-1][2]

    for low, high, scheme in levels:
        if low <= snr_db <= high:
            return scheme

    # Fallback (shouldn't happen given contiguous ranges)
    return levels[-1][2]


def throughput_bps(scheme):
    """Bits-per-symbol (throughput) for a given modulation scheme."""
    return BPS[scheme]


def adaptive_bps_curve(snr_db_range, target_ber=0.001):
    """BPS throughput of the adaptive modem across an SNR range (Fig. 4)."""
    schemes = [select_modulation(s, target_ber) for s in snr_db_range]
    return np.array([throughput_bps(s) for s in schemes]), schemes
