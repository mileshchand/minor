"""
=========================================================
channel.py

Wireless Channel Models

Contains:
    • AWGN Channel
    • Rayleigh Fading Channel
=========================================================
"""

import numpy as np


#########################################################
# Utility Function
#########################################################

def calculate_noise_variance(signal, snr_db):
    """
    Calculate the noise variance for a given SNR.

    Parameters
    ----------
    signal : ndarray
        Transmitted symbols.

    snr_db : float
        Signal-to-noise ratio in dB.

    Returns
    -------
    float
        Noise variance.
    """

    signal_power = np.mean(np.abs(signal) ** 2)

    snr_linear = 10 ** (snr_db / 10)

    noise_variance = signal_power / snr_linear

    return noise_variance


#########################################################
# AWGN Channel
#########################################################

def awgn_channel(signal, snr_db):
    """
    Add Additive White Gaussian Noise (AWGN)
    """

    noise_variance = calculate_noise_variance(signal, snr_db)

    if np.iscomplexobj(signal):

        noise = (
            np.sqrt(noise_variance / 2)
            * (np.random.randn(*signal.shape)
               + 1j * np.random.randn(*signal.shape))
        )

    else:

        noise = (
            np.sqrt(noise_variance)
            * np.random.randn(*signal.shape)
        )

    received = signal + noise

    return received


#########################################################
# Rayleigh Channel
#########################################################

def rayleigh_channel(signal, snr_db):
    """
    Flat Rayleigh fading channel followed by AWGN.
    """

    h = (
        np.random.randn(*signal.shape)
        + 1j * np.random.randn(*signal.shape)
    ) / np.sqrt(2)

    faded_signal = h * signal

    noise_variance = calculate_noise_variance(faded_signal, snr_db)

    noise = (
        np.sqrt(noise_variance / 2)
        * (np.random.randn(*signal.shape)
           + 1j * np.random.randn(*signal.shape))
    )

    received = faded_signal + noise

    # Perfect channel estimation (equalization)
    received = received / h

    return received


#########################################################
# SNR Measurement
#########################################################

def estimate_snr(tx_signal, rx_signal):
    """
    Estimate SNR from transmitted and received signals.
    """

    signal_power = np.mean(np.abs(tx_signal) ** 2)

    noise = rx_signal - tx_signal

    noise_power = np.mean(np.abs(noise) ** 2)

    snr = signal_power / noise_power

    return 10 * np.log10(snr)


#########################################################
# Channel Test
#########################################################

if __name__ == "__main__":

    np.random.seed(0)

    tx = np.random.choice([-1, 1], 10000)

    snr = 10

    rx = awgn_channel(tx, snr)

    est = estimate_snr(tx, rx)

    print("Requested SNR :", snr, "dB")
    print("Estimated SNR :", round(est, 2), "dB")

    rx_rayleigh = rayleigh_channel(tx.astype(complex), snr)

    print("Rayleigh channel test completed.")