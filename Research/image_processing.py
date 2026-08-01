"""
image_processing.py
--------------------
Implements the "Simple Image Encoder / Decoder" + adaptive image
transmission algorithm described in the paper (Section IV):

    I.    Read image
    II.   Convert image to serial bitstream
    III.  Split data into equal-sized frames
    IV.   Start at minimum-level modulation
    V.    Check channel condition, select mode (modulation switch)
    VI.   Share info with receiver
    VII.  Display / evaluate image quality (adaptive vs non-adaptive)
    VIII. Verify target BER at receiver
"""

import numpy as np
from PIL import Image

from adaptive import select_modulation, BPS
from modulation import modulate, demodulate
from channel import add_awgn_noise
from utils import bytes_to_bits, bits_to_bytes, pad_bits_to_multiple


def load_grayscale_image(path, size=(128, 128)):
    """Load an image, convert to grayscale, resize for fast simulation."""
    img = Image.open(path).convert("L").resize(size)
    return np.array(img, dtype=np.uint8)


def image_to_bits(img_array):
    """Step I-II: read image -> serial bitstream."""
    flat = img_array.flatten()
    return bytes_to_bits(flat), img_array.shape


def bits_to_image(bits, shape):
    """Reconstruct image array from a recovered bitstream."""
    byte_arr = bits_to_bytes(bits)
    n_pixels = shape[0] * shape[1]
    byte_arr = byte_arr[:n_pixels]
    if len(byte_arr) < n_pixels:
        byte_arr = np.concatenate([byte_arr, np.zeros(n_pixels - len(byte_arr), dtype=np.uint8)])
    return byte_arr.reshape(shape)


def transmit_image_adaptive(img_array, snr_sequence, target_ber=0.001,
                             frame_size=4096, seed=None):
    """
    Steps III-VIII: split bitstream into frames, pick modulation per frame
    based on the (given/estimated) channel SNR for that frame, modulate,
    pass through AWGN, demodulate, and reassemble the image.

    snr_sequence: array of SNR(dB) values, one per frame (cycled if shorter
                  than number of frames), representing time-varying channel
                  conditions during the frame-based transmission.
    """
    if seed is not None:
        np.random.seed(seed)

    bits, shape = image_to_bits(img_array)
    n_frames = int(np.ceil(len(bits) / frame_size))
    rx_bits_all = []

    for f in range(n_frames):
        start = f * frame_size
        end = min(start + frame_size, len(bits))
        frame_bits = bits[start:end]

        snr_db = snr_sequence[f % len(snr_sequence)]
        scheme = select_modulation(snr_db, target_ber)
        k = BPS[scheme]

        padded_bits, pad = pad_bits_to_multiple(frame_bits, k)
        symbols = modulate(scheme, padded_bits)
        rx_symbols = add_awgn_noise(symbols, snr_db)
        rx_bits = demodulate(scheme, rx_symbols)

        if pad:
            rx_bits = rx_bits[:-pad]
        rx_bits_all.append(rx_bits)

    rx_bits_full = np.concatenate(rx_bits_all)
    rx_img = bits_to_image(rx_bits_full, shape)
    return rx_img


def transmit_image_fixed(img_array, snr_sequence, scheme="16QAM",
                          frame_size=4096, seed=None):
    """Non-adaptive baseline: transmit every frame using a fixed scheme,
    used for the adaptive-vs-non-adaptive PSNR comparison (Fig. 6, Fig. 8)."""
    if seed is not None:
        np.random.seed(seed)

    bits, shape = image_to_bits(img_array)
    k = BPS[scheme]
    n_frames = int(np.ceil(len(bits) / frame_size))
    rx_bits_all = []

    for f in range(n_frames):
        start = f * frame_size
        end = min(start + frame_size, len(bits))
        frame_bits = bits[start:end]
        snr_db = snr_sequence[f % len(snr_sequence)]

        padded_bits, pad = pad_bits_to_multiple(frame_bits, k)
        symbols = modulate(scheme, padded_bits)
        rx_symbols = add_awgn_noise(symbols, snr_db)
        rx_bits = demodulate(scheme, rx_symbols)

        if pad:
            rx_bits = rx_bits[:-pad]
        rx_bits_all.append(rx_bits)

    rx_bits_full = np.concatenate(rx_bits_all)
    rx_img = bits_to_image(rx_bits_full, shape)
    return rx_img
