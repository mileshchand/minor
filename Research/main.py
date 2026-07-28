"""
===========================================================
Adaptive Modulation System for Image Transmission
Minor Project

Author : Ojaswi Chand
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from image_processing import (
    image_to_bits,
    bits_to_image
)

from adaptive import choose_modulation

from modulation import (
    bpsk_mod,
    bpsk_demod,
    qpsk_mod,
    qpsk_demod,
    qam16_mod,
    qam16_demod,
    qam64_mod,
    qam64_demod
)

from channel import (
    awgn_channel,
    rayleigh_channel
)

from ber import calculate_ber

from plots import (
    plot_ber_curve,
    show_images
)

##########################################################
# SETTINGS
##########################################################

IMAGE_PATH = "images/image.png"

USE_RAYLEIGH = False

SNR_VALUES = np.arange(0, 21, 2)

##########################################################
# LOAD IMAGE
##########################################################

print("Loading image...")

tx_bits, image_shape = image_to_bits(IMAGE_PATH)

print("Image Loaded")
print("Total Bits :", len(tx_bits))

##########################################################
# BER STORAGE
##########################################################

ber_results = []

##########################################################
# START SIMULATION
##########################################################

print("\nStarting Simulation...\n")

for snr in SNR_VALUES:

    modulation = choose_modulation(snr)

    print("-----------------------------------")
    print(f"SNR : {snr} dB")
    print(f"Selected Modulation : {modulation}")

    ######################################################
    # MODULATION
    ######################################################

    if modulation == "BPSK":

        tx_symbols = bpsk_mod(tx_bits)

    elif modulation == "QPSK":

        tx_symbols = qpsk_mod(tx_bits)

    elif modulation == "16QAM":

        tx_symbols = qam16_mod(tx_bits)

    elif modulation == "64QAM":

        tx_symbols = qam64_mod(tx_bits)

    ######################################################
    # CHANNEL
    ######################################################

    if USE_RAYLEIGH:

        rx_symbols = rayleigh_channel(tx_symbols, snr)

    else:

        rx_symbols = awgn_channel(tx_symbols, snr)

    ######################################################
    # DEMODULATION
    ######################################################

    if modulation == "BPSK":

        rx_bits = bpsk_demod(rx_symbols)

    elif modulation == "QPSK":

        rx_bits = qpsk_demod(rx_symbols)

    elif modulation == "16QAM":

        rx_bits = qam16_demod(rx_symbols)

    elif modulation == "64QAM":

        rx_bits = qam64_demod(rx_symbols)

    ######################################################
    # BER
    ######################################################

    minimum = min(len(tx_bits), len(rx_bits))

    tx_compare = tx_bits[:minimum]

    rx_compare = rx_bits[:minimum]

    ber = calculate_ber(tx_compare, rx_compare)

    ber_results.append(ber)

    print(f"BER : {ber:.8f}")

##########################################################
# FINAL IMAGE
##########################################################

received_image = bits_to_image(rx_compare, image_shape)

##########################################################
# DISPLAY
##########################################################

show_images(
    IMAGE_PATH,
    received_image
)

##########################################################
# BER GRAPH
##########################################################

plot_ber_curve(
    SNR_VALUES,
    ber_results
)

##########################################################
# SAVE OUTPUT IMAGE
##########################################################

plt.imsave(
    "images/reconstructed.png",
    received_image,
    cmap="gray"
)

##########################################################
# SUMMARY
##########################################################

print("\n====================================")

print("Simulation Complete")

print("Output Image Saved : images/reconstructed.png")

print("BER Graph Generated")

print("====================================")