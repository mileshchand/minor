"""
=========================================================
adaptive.py

Adaptive Modulation Controller

Author : Ojaswi Chand

Selects modulation scheme according to channel SNR.

Thresholds:

SNR < 5 dB        -> BPSK
5 <= SNR < 10     -> QPSK
10 <= SNR < 15    -> 16-QAM
SNR >= 15         -> 64-QAM

=========================================================
"""

import numpy as np


#########################################################
# Default SNR Thresholds
#########################################################

DEFAULT_THRESHOLDS = {
    "BPSK": 5,
    "QPSK": 10,
    "16QAM": 15
}


#########################################################
# Modulation Selection
#########################################################

def choose_modulation(snr_db):
    """
    Select modulation based on SNR.

    Parameters
    ----------
    snr_db : float

    Returns
    -------
    str
        "BPSK"
        "QPSK"
        "16QAM"
        "64QAM"
    """

    if snr_db < DEFAULT_THRESHOLDS["BPSK"]:

        return "BPSK"

    elif snr_db < DEFAULT_THRESHOLDS["QPSK"]:

        return "QPSK"

    elif snr_db < DEFAULT_THRESHOLDS["16QAM"]:

        return "16QAM"

    else:

        return "64QAM"


#########################################################
# Bits Per Symbol
#########################################################

def bits_per_symbol(modulation):
    """
    Return bits carried by one symbol.
    """

    table = {
        "BPSK": 1,
        "QPSK": 2,
        "16QAM": 4,
        "64QAM": 6
    }

    if modulation not in table:
        raise ValueError(f"Unknown modulation: {modulation}")

    return table[modulation]


#########################################################
# Spectral Efficiency
#########################################################

def spectral_efficiency(modulation):
    """
    Spectral efficiency (bits/symbol).

    For M-ary modulation:
        η = log2(M)
    """

    return bits_per_symbol(modulation)


#########################################################
# Throughput
#########################################################

def calculate_throughput(bit_rate, ber):
    """
    Effective throughput.

    Throughput = Bit Rate × (1 − BER)
    """

    return bit_rate * (1.0 - ber)


#########################################################
# Modulation Summary
#########################################################

def modulation_summary(snr_db):
    """
    Return modulation information as a dictionary.
    """

    mod = choose_modulation(snr_db)

    return {
        "snr": snr_db,
        "modulation": mod,
        "bits_per_symbol": bits_per_symbol(mod),
        "spectral_efficiency": spectral_efficiency(mod)
    }


#########################################################
# Print Decision Table
#########################################################

def print_thresholds():
    """
    Display adaptive switching thresholds.
    """

    print("\nAdaptive Modulation Thresholds")
    print("--------------------------------")
    print("SNR < 5 dB        -> BPSK")
    print("5 <= SNR < 10 dB -> QPSK")
    print("10 <= SNR < 15 dB -> 16-QAM")
    print("SNR >= 15 dB      -> 64-QAM")
    print("--------------------------------")


#########################################################
# Test
#########################################################

if __name__ == "__main__":

    print_thresholds()

    print()

    for snr in range(0, 21):

        info = modulation_summary(snr)

        print(
            f"SNR = {snr:2d} dB | "
            f"{info['modulation']:6s} | "
            f"{info['bits_per_symbol']} bits/symbol"
        )