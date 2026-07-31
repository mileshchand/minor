"""
=========================================================
utils.py

Utility Functions


Contains helper functions used throughout the project.
=========================================================
"""

import numpy as np
import random
import time


#########################################################
# Random Seed
#########################################################

def set_seed(seed=42):
    """
    Set random seed for reproducible simulations.
    """

    np.random.seed(seed)
    random.seed(seed)


#########################################################
# Linear SNR
#########################################################

def db_to_linear(db):
    """
    Convert dB to linear scale.
    """

    return 10 ** (db / 10)


#########################################################
# dB Scale
#########################################################

def linear_to_db(value):
    """
    Convert linear value to dB.
    """

    return 10 * np.log10(value)


#########################################################
# Signal Power
#########################################################

def signal_power(signal):
    """
    Average signal power.
    """

    signal = np.asarray(signal)

    return np.mean(np.abs(signal) ** 2)


#########################################################
# Normalize Signal
#########################################################

def normalize_signal(signal):
    """
    Normalize average signal power to one.
    """

    power = signal_power(signal)

    if power == 0:
        return signal

    return signal / np.sqrt(power)


#########################################################
# Padding Bits
#########################################################

def pad_bits(bits, bits_per_symbol):
    """
    Pad bitstream with zeros.
    """

    bits = np.asarray(bits, dtype=np.uint8)

    remainder = len(bits) % bits_per_symbol

    if remainder == 0:
        return bits

    padding = bits_per_symbol - remainder

    zeros = np.zeros(
        padding,
        dtype=np.uint8
    )

    return np.concatenate((bits, zeros))


#########################################################
# Remove Padding
#########################################################

def remove_padding(bits, original_length):
    """
    Remove padded bits.
    """

    return bits[:original_length]


#########################################################
# Execution Timer
#########################################################

class Timer:

    def __init__(self):

        self.start_time = None

    def start(self):

        self.start_time = time.time()

    def stop(self):

        return time.time() - self.start_time


#########################################################
# Progress Bar
#########################################################

def progress(current, total):

    percent = (current / total) * 100

    print(
        f"\rProgress : {percent:6.2f}%",
        end=""
    )


#########################################################
# Binary String
#########################################################

def bits_to_string(bits):
    """
    Convert bit array to printable string.
    """

    return "".join(str(int(b)) for b in bits)


#########################################################
# Error Percentage
#########################################################

def error_percentage(tx_bits, rx_bits):
    """
    Percentage of incorrect bits.
    """

    minimum = min(len(tx_bits), len(rx_bits))

    tx = np.asarray(tx_bits[:minimum])

    rx = np.asarray(rx_bits[:minimum])

    return (
        np.sum(tx != rx) /
        minimum
    ) * 100


#########################################################
# Simulation Summary
#########################################################

def simulation_summary():

    print("\n====================================")

    print("Adaptive Image Transmission System")

    print("------------------------------------")

    print("Supported Modulation Schemes")

    print(" • BPSK")

    print(" • QPSK")

    print(" • 16-QAM")

    print(" • 64-QAM")

    print("------------------------------------")

    print("Channel")

    print(" • AWGN")

    print(" • Rayleigh")

    print("------------------------------------")

    print("Performance Metrics")

    print(" • BER")

    print(" • Throughput")

    print(" • Image Reconstruction")

    print("====================================\n")


#########################################################
# Module Test
#########################################################

if __name__ == "__main__":

    set_seed()

    simulation_summary()

    bits = np.random.randint(0, 2, 17)

    print("Original Length :", len(bits))

    padded = pad_bits(bits, 6)

    print("Padded Length :", len(padded))

    print("Signal Power :", signal_power(np.array([1, -1, 1, -1])))

    print("Linear (10 dB) :", db_to_linear(10))

    print("dB (10) :", linear_to_db(10))
