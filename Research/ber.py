"""
ber.py
------
Theoretical bit-error-rate expressions for BPSK, QPSK, 16-QAM and 64-QAM
in an AWGN channel, taken directly from Eq. (1.1)-(1.4) of the reference
IEEE paper, plus a Monte-Carlo (simulated) BER routine used to validate
those formulas against the actual modulate -> AWGN -> demodulate chain.
"""

import numpy as np
from utils import qfunc
from modulation import modulate, demodulate, BITS_PER_SYMBOL
from channel import add_awgn_noise


# ------------------------ Theoretical BER (Eq. 1.1 - 1.4) ------------------
def ber_bpsk_theory(gamma):
    """Eq. (1.1): P_BPSK(gamma) = Q(sqrt(2*gamma))"""
    gamma = np.asarray(gamma, dtype=float)
    return qfunc(np.sqrt(2 * gamma))


def ber_qpsk_theory(gamma):
    """Eq. (1.2): P_QPSK(gamma) = Q(sqrt(gamma))"""
    gamma = np.asarray(gamma, dtype=float)
    return qfunc(np.sqrt(gamma))


def ber_16qam_theory(gamma):
    """
    Eq. (1.3):
    P_16QAM(gamma) = 1/4*[Q(sqrt(g/5)) + Q(3*sqrt(g/5))] + 1/2*Q(sqrt(g/5))
    """
    gamma = np.asarray(gamma, dtype=float)
    g5 = np.sqrt(gamma / 5.0)
    term = 0.25 * (qfunc(g5) + qfunc(3 * g5)) + 0.5 * qfunc(g5)
    return term


def ber_64qam_theory(gamma):
    """
    Eq. (1.4), standard closed-form approximation for 64-QAM BER in AWGN
    (kept consistent with the structure given in the paper):
    """
    gamma = np.asarray(gamma, dtype=float)
    g21 = np.sqrt(gamma / 21.0)
    term = (7.0 / 24.0) * qfunc(g21) \
        + (1.0 / 4.0) * qfunc(3 * g21) \
        - (1.0 / 24.0) * qfunc(5 * g21) \
        + (1.0 / 24.0) * qfunc(9 * g21) \
        - (1.0 / 24.0) * qfunc(13 * g21)
    return np.clip(term, 0, 1)


THEORY_FUNCS = {
    "BPSK": ber_bpsk_theory,
    "QPSK": ber_qpsk_theory,
    "16QAM": ber_16qam_theory,
    "64QAM": ber_64qam_theory,
}


def theoretical_ber_curve(scheme, snr_db_range):
    """Return theoretical BER array over an SNR(dB) range for a scheme."""
    gamma_linear = 10 ** (np.asarray(snr_db_range) / 10.0)
    return THEORY_FUNCS[scheme](gamma_linear)


# ------------------------ Monte-Carlo simulated BER -------------------------
def simulate_ber(scheme, snr_db, n_bits=200_000, seed=None):
    """
    Monte-Carlo BER estimate: generate random bits, modulate, pass through
    AWGN channel at snr_db, demodulate, and count bit errors.
    Used to validate/replicate the theoretical curves (Sec. V of paper /
    Validation Strategy of proposal).
    """
    if seed is not None:
        np.random.seed(seed)

    k = BITS_PER_SYMBOL[scheme]
    n_bits = (n_bits // k) * k  # make divisible by bits/symbol
    tx_bits = np.random.randint(0, 2, n_bits).astype(np.uint8)

    symbols = modulate(scheme, tx_bits)
    rx_symbols = add_awgn_noise(symbols, snr_db)
    rx_bits = demodulate(scheme, rx_symbols)

    n_errors = np.sum(tx_bits != rx_bits[:n_bits])
    return n_errors / n_bits


def simulated_ber_curve(scheme, snr_db_range, n_bits=200_000, seed=None):
    """Simulated BER across a range of SNR values (averaged Monte-Carlo)."""
    return np.array([
        simulate_ber(scheme, snr, n_bits=n_bits, seed=seed)
        for snr in snr_db_range
    ])
