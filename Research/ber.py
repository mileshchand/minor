"""
=========================================================
ber.py

Bit Error Rate (BER) Module

Functions:
    • BER Calculation
    • Error Count
    • BER Table
    • Average BER

Author : Ojaswi Chand
=========================================================
"""

import numpy as np


#########################################################
# Calculate BER
#########################################################

def calculate_ber(tx_bits, rx_bits):
    """
    Calculate Bit Error Rate (BER).

    BER = Number of Bit Errors / Total Number of Bits
    """

    tx_bits = np.asarray(tx_bits, dtype=np.uint8)
    rx_bits = np.asarray(rx_bits, dtype=np.uint8)

    minimum = min(len(tx_bits), len(rx_bits))

    tx_bits = tx_bits[:minimum]
    rx_bits = rx_bits[:minimum]

    errors = np.sum(tx_bits != rx_bits)

    ber = errors / minimum

    return ber


#########################################################
# Count Bit Errors
#########################################################

def count_bit_errors(tx_bits, rx_bits):
    """
    Count total bit errors.
    """

    tx_bits = np.asarray(tx_bits, dtype=np.uint8)
    rx_bits = np.asarray(rx_bits, dtype=np.uint8)

    minimum = min(len(tx_bits), len(rx_bits))

    tx_bits = tx_bits[:minimum]
    rx_bits = rx_bits[:minimum]

    return np.sum(tx_bits != rx_bits)


#########################################################
# Accuracy
#########################################################

def accuracy(tx_bits, rx_bits):
    """
    Percentage of correctly received bits.
    """

    ber = calculate_ber(tx_bits, rx_bits)

    return (1 - ber) * 100


#########################################################
# BER Table
#########################################################

def print_ber_table(snr_values, ber_values):
    """
    Print BER results in table format.
    """

    print("\n===================================")
    print("          BER RESULTS")
    print("===================================")
    print("{:<10} {:<15}".format("SNR(dB)", "BER"))
    print("-----------------------------------")

    for snr, ber in zip(snr_values, ber_values):
        print("{:<10} {:.8f}".format(snr, ber))

    print("===================================\n")


#########################################################
# Average BER
#########################################################

def average_ber(ber_values):
    """
    Calculate average BER.
    """

    return np.mean(ber_values)


#########################################################
# BER Summary
#########################################################

def ber_summary(snr_values, ber_values):

    print_ber_table(snr_values, ber_values)

    avg = average_ber(ber_values)

    print(f"Average BER : {avg:.8f}")

    best = np.argmin(ber_values)

    print(f"Best BER    : {ber_values[best]:.8f}")

    print(f"Best SNR    : {snr_values[best]} dB")


#########################################################
# Test
#########################################################

if __name__ == "__main__":

    np.random.seed(0)

    tx = np.random.randint(0, 2, 1000)

    rx = tx.copy()

    # Introduce random bit errors
    error_indices = np.random.choice(1000, 20, replace=False)

    rx[error_indices] ^= 1

    ber = calculate_ber(tx, rx)

    print("BER :", ber)

    print("Errors :", count_bit_errors(tx, rx))

    print("Accuracy :", accuracy(tx, rx), "%")

    snr = [0, 2, 4, 6, 8]

    bers = [0.20, 0.11, 0.05, 0.01, ber]

    ber_summary(snr, bers)