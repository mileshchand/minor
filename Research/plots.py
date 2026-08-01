"""
plots.py
--------
Recreates the figures from the reference paper:
    Fig. 2 - BER curves for BPSK/QPSK/16QAM/64QAM (theoretical) vs SNR
    Fig. 3 - Adaptive modem theoretical BER curve for 3 target BERs
    Fig. 4 - BPS throughput of adaptive modulation for the 3 targets
    Fig. 6 - Received images: adaptive vs non-adaptive modem
    Fig. 7 - Simulated BER curve for adaptive modem
    Fig. 8 - PSNR vs channel SNR: adaptive vs fixed modulation
All figures are saved into the results/ folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from ber import theoretical_ber_curve, simulated_ber_curve
from adaptive import select_modulation, adaptive_bps_curve, SWITCHING_LEVELS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SCHEMES = ["BPSK", "QPSK", "16QAM", "64QAM"]
COLORS = {"BPSK": "tab:red", "QPSK": "tab:green", "16QAM": "tab:blue", "64QAM": "black"}


def plot_ber_curves(snr_range=np.arange(0, 31, 1)):
    """Fig. 2: BER curve for different modulation schemes (theoretical)."""
    plt.figure(figsize=(7, 5))
    for scheme in SCHEMES:
        ber = theoretical_ber_curve(scheme, snr_range)
        plt.semilogy(snr_range, ber, "-o", color=COLORS[scheme],
                      label=f"{scheme} Theoretical", markersize=3)

    for target in [0.1, 0.01, 0.001]:
        plt.axhline(target, color="gray", linestyle="--", linewidth=0.8)

    plt.title("BER CURVE FOR BPSK, QPSK, 16-QAM, 64-QAM")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.ylim(1e-6, 1)
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig2_ber_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_adaptive_theoretical_ber(snr_range=np.arange(0, 31, 1)):
    """Fig. 3: Adaptive modem theoretical BER curve for 3 target BERs."""
    plt.figure(figsize=(7, 5))
    for target, style in zip([0.1, 0.01, 0.001], ["-^b", "-or", "-^k"]):
        ber_vals = []
        for snr in snr_range:
            scheme = select_modulation(snr, target)
            ber_vals.append(theoretical_ber_curve(scheme, [snr])[0])
        plt.semilogy(snr_range, ber_vals, style, label=f"target BER={target}", markersize=4)

    plt.title("Adaptive modem theoretical BER curve")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig3_adaptive_theoretical_ber.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_adaptive_throughput(snr_range=np.arange(0, 31, 1)):
    """Fig. 4: BPS throughput of adaptive modulation for all three BER targets."""
    plt.figure(figsize=(7, 5))
    for target, style in zip([0.1, 0.01, 0.001], ["-ob", "-or", "-ok"]):
        bps_vals, _ = adaptive_bps_curve(snr_range, target)
        plt.plot(snr_range, bps_vals, style, label=f"target BER {target}", markersize=4)

    plt.title("Simulation BPS throughput in AWGN")
    plt.xlabel("SNR in dB")
    plt.ylabel("BPS throughput bits/symbol")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig4_adaptive_throughput.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_simulated_adaptive_ber(snr_range=np.arange(0, 31, 2), target_ber=0.001, n_bits=100_000):
    """Fig. 7: Simulated (Monte-Carlo) BER curve for the adaptive modem."""
    ber_vals = []
    for snr in snr_range:
        scheme = select_modulation(snr, target_ber)
        ber_vals.append(simulated_ber_curve(scheme, [snr], n_bits=n_bits)[0])

    plt.figure(figsize=(7, 5))
    plt.semilogy(snr_range, np.array(ber_vals) + 1e-9, "-o", color="tab:blue")
    plt.title(f"Simulated BER curve in AWGN at Target BER={target_ber}")
    plt.xlabel("SNR in dB")
    plt.ylabel("BER")
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig7_simulated_adaptive_ber.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_psnr_vs_snr(snr_range, psnr_adaptive, psnr_fixed, fixed_label="fixed modulation(16QAM)"):
    """Fig. 8: PSNR vs. Channel SNR in Adaptive image transmission."""
    plt.figure(figsize=(7, 5))
    plt.plot(snr_range, psnr_adaptive, "-ob", label="Adaptive modulation")
    plt.plot(snr_range, psnr_fixed, "-sr", label=fixed_label)
    plt.title("PSNR vs channel SNR in adaptive image transmission")
    plt.xlabel("channel SNR in dB")
    plt.ylabel("PSNR of image in db")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig8_psnr_vs_snr.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_image_comparison(original, rx_adaptive, rx_fixed, fixed_label="non-adaptive modem(16QAM)"):
    """Fig. 6: Original / adaptive / non-adaptive received images."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original TX image")
    axes[1].imshow(rx_adaptive, cmap="gray")
    axes[1].set_title("Adaptive modem RX")
    axes[2].imshow(rx_fixed, cmap="gray")
    axes[2].set_title(fixed_label)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig6_image_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path
