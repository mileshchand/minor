"""
channel.py
----------
AWGN channel model: r(t) = s(t) + n(t)   (Eq. 2.1 of the proposal)

add_awgn_noise() adds complex Gaussian noise to a stream of unit-energy
symbols so that the resulting Eb/N0 / SNR matches the requested value,
scaled by bits-per-symbol so that BER vs SNR curves line up with the
theoretical formulas (Eq. 1.1 - 1.4), which are expressed in terms of
average SNR per symbol (gamma).
"""

import numpy as np


def add_awgn_noise(symbols, snr_db):
    """
    Add AWGN to complex symbols for a given average SNR (dB) per symbol.

    symbols  : complex ndarray, unit average energy per symbol
    snr_db   : desired average SNR (gamma) in dB

    Returns noisy complex symbols.
    """
    snr_linear = 10 ** (snr_db / 10.0)
    # Average symbol energy is assumed to be 1 (modulators normalize this)
    noise_power = 1.0 / snr_linear
    noise_std = np.sqrt(noise_power / 2.0)  # split between I and Q

    noise = noise_std * (np.random.randn(*symbols.shape) +
                          1j * np.random.randn(*symbols.shape))
    return symbols + noise


def estimate_channel_snr(true_snr_db, estimation_noise_std=0.0):
    """
    Simulate SNR estimation/feedback from receiver to transmitter.
    In an ideal system estimation_noise_std=0 gives perfect feedback;
    a nonzero value can be used to study feedback-error robustness.
    """
    if estimation_noise_std == 0.0:
        return true_snr_db
    return true_snr_db + np.random.normal(0, estimation_noise_std)
