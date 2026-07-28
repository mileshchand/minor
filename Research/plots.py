"""
=========================================================
plots.py

Visualization Module

Functions:
    • BER vs SNR Plot
    • Original vs Reconstructed Image
    • Constellation Diagram
    • Throughput Plot

Author : Ojaswi Chand
=========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


#########################################################
# Create Results Folder
#########################################################

RESULT_DIR = "results"

os.makedirs(RESULT_DIR, exist_ok=True)


#########################################################
# BER Curve
#########################################################

def plot_ber_curve(snr_values, ber_values):

    plt.figure(figsize=(8,5))

    plt.semilogy(
        snr_values,
        ber_values,
        'o-',
        linewidth=2,
        markersize=6
    )

    plt.grid(True, which='both')

    plt.xlabel("SNR (dB)")

    plt.ylabel("Bit Error Rate (BER)")

    plt.title("BER vs SNR")

    plt.tight_layout()

    plt.savefig(os.path.join(
        RESULT_DIR,
        "ber_curve.png"
    ))

    plt.show()


#########################################################
# Throughput Curve
#########################################################

def plot_throughput(snr_values,
                    throughput_values):

    plt.figure(figsize=(8,5))

    plt.plot(
        snr_values,
        throughput_values,
        's-',
        linewidth=2
    )

    plt.grid(True)

    plt.xlabel("SNR (dB)")

    plt.ylabel("Throughput")

    plt.title("Throughput vs SNR")

    plt.tight_layout()

    plt.savefig(os.path.join(
        RESULT_DIR,
        "throughput.png"
    ))

    plt.show()


#########################################################
# Original vs Reconstructed Image
#########################################################

def show_images(original_image_path,
                reconstructed_image):

    original = Image.open(
        original_image_path
    ).convert("L")

    original = np.array(original)

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)

    plt.imshow(
        original,
        cmap="gray"
    )

    plt.title("Original")

    plt.axis("off")

    plt.subplot(1,2,2)

    plt.imshow(
        reconstructed_image,
        cmap="gray"
    )

    plt.title("Reconstructed")

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(os.path.join(
        RESULT_DIR,
        "comparison.png"
    ))

    plt.show()


#########################################################
# Constellation Diagram
#########################################################

def plot_constellation(symbols,
                       title="Constellation"):

    plt.figure(figsize=(6,6))

    plt.scatter(
        symbols.real,
        symbols.imag,
        s=8
    )

    plt.grid(True)

    plt.xlabel("In-phase")

    plt.ylabel("Quadrature")

    plt.title(title)

    plt.tight_layout()

    filename = (
        title.lower()
             .replace(" ","_")
             + ".png"
    )

    plt.savefig(os.path.join(
        RESULT_DIR,
        filename
    ))

    plt.show()


#########################################################
# BER Comparison
#########################################################

def compare_modulations(results):
    """
    results = {
        "BPSK": [...],
        "QPSK": [...],
        "16QAM": [...],
        "64QAM": [...]
    }
    """

    plt.figure(figsize=(8,5))

    for mod, values in results.items():

        snr = np.arange(
            0,
            2*len(values),
            2
        )

        plt.semilogy(
            snr,
            values,
            marker='o',
            label=mod
        )

    plt.grid(True, which="both")

    plt.xlabel("SNR (dB)")

    plt.ylabel("BER")

    plt.title(
        "BER Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(
        RESULT_DIR,
        "ber_comparison.png"
    ))

    plt.show()


#########################################################
# Save Figure Helper
#########################################################

def save_current(name):

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            name
        )
    )


#########################################################
# Test
#########################################################

if __name__ == "__main__":

    snr = np.arange(0,21,2)

    ber = np.exp(-snr/3)

    plot_ber_curve(snr, ber)

    throughput = np.linspace(
        1,
        6,
        len(snr)
    )

    plot_throughput(
        snr,
        throughput
    )

    symbols = (
        np.random.randn(500)
        +
        1j*np.random.randn(500)
    )

    plot_constellation(
        symbols,
        "Test Constellation"
    )

    print("plots.py test completed.")