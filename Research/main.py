"""
main.py
-------
Entry point for the minor project simulation:
"Replication of MATLAB-Standard BER Results for Adaptive Modulation
Using Python SDR Tools"

Runs, in order:
  1. Theoretical BER curves for BPSK/QPSK/16QAM/64QAM         -> Fig. 2
  2. Adaptive modem theoretical BER curve (3 targets)          -> Fig. 3
  3. Adaptive modem BPS throughput (3 targets)                 -> Fig. 4
  4. Monte-Carlo simulated BER curve for adaptive modem        -> Fig. 7
  5. Adaptive image transmission experiment                    -> Fig. 6, 8

All output figures are written to results/.
Replace images/image.png with your own picture to use a real test image.
"""

import os
import numpy as np

from image_processing import load_grayscale_image, transmit_image_adaptive, transmit_image_fixed
from utils import compute_psnr
import plots

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "images", "image.png")
TARGET_BER = 0.001  # 0.1% target used for the image transmission experiment


def run_ber_analysis():
    print("[1/5] Generating theoretical BER curves (Fig. 2)...")
    p2 = plots.plot_ber_curves()
    print(f"      saved -> {p2}")

    print("[2/5] Generating adaptive modem theoretical BER curve (Fig. 3)...")
    p3 = plots.plot_adaptive_theoretical_ber()
    print(f"      saved -> {p3}")

    print("[3/5] Generating adaptive modem BPS throughput curve (Fig. 4)...")
    p4 = plots.plot_adaptive_throughput()
    print(f"      saved -> {p4}")

    print("[4/5] Running Monte-Carlo simulated BER curve for adaptive modem (Fig. 7)...")
    p7 = plots.plot_simulated_adaptive_ber(target_ber=TARGET_BER)
    print(f"      saved -> {p7}")


def run_image_transmission_experiment():
    print("[5/5] Running adaptive image transmission experiment (Fig. 6, Fig. 8)...")

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(
            f"Place a test image at {IMAGE_PATH} (any format, will be "
            f"converted to grayscale)."
        )

    img = load_grayscale_image(IMAGE_PATH, size=(128, 128))

    # SNR sweep used both for the single-image comparison at moderate SNR
    # and for the PSNR-vs-SNR curve
    snr_sweep = np.arange(0, 21, 2)

    psnr_adaptive_list = []
    psnr_fixed_list = []

    for snr in snr_sweep:
        snr_seq = np.array([snr])  # constant channel condition per run
        rx_adaptive = transmit_image_adaptive(img, snr_seq, target_ber=TARGET_BER, seed=1)
        rx_fixed = transmit_image_fixed(img, snr_seq, scheme="16QAM", seed=1)

        psnr_adaptive_list.append(compute_psnr(img, rx_adaptive))
        psnr_fixed_list.append(compute_psnr(img, rx_fixed))

    p8 = plots.plot_psnr_vs_snr(snr_sweep, psnr_adaptive_list, psnr_fixed_list)
    print(f"      saved -> {p8}")

    # Single side-by-side comparison at a representative low/medium SNR (Fig. 6)
    demo_snr = np.array([6])
    rx_adaptive_demo = transmit_image_adaptive(img, demo_snr, target_ber=TARGET_BER, seed=2)
    rx_fixed_demo = transmit_image_fixed(img, demo_snr, scheme="16QAM", seed=2)
    p6 = plots.plot_image_comparison(img, rx_adaptive_demo, rx_fixed_demo)
    print(f"      saved -> {p6}")


def main():
    os.makedirs(plots.RESULTS_DIR, exist_ok=True)
    run_ber_analysis()
    run_image_transmission_experiment()
    print("\nAll figures generated in the 'results/' folder.")


if __name__ == "__main__":
    main()
