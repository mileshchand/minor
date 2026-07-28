"""
=========================================================
modulation.py

Adaptive Modulation Module

Contains:
    • BPSK
    • QPSK
    • (16-QAM and 64-QAM added in Part 2)

=========================================================
"""

import numpy as np

#########################################################
# Utility Functions
#########################################################

def pad_bits(bits, bits_per_symbol):
    """
    Pad bit array with zeros so that its length is a multiple
    of bits_per_symbol.
    """
    bits = np.asarray(bits, dtype=np.uint8)

    remainder = len(bits) % bits_per_symbol

    if remainder != 0:
        padding = bits_per_symbol - remainder
        bits = np.concatenate(
            [bits, np.zeros(padding, dtype=np.uint8)]
        )

    return bits


#########################################################
# BPSK
#########################################################

def bpsk_mod(bits):
    """
    BPSK Mapping

    0 -> -1
    1 -> +1
    """

    bits = np.asarray(bits, dtype=np.uint8)

    symbols = 2 * bits - 1

    return symbols.astype(float)


def bpsk_demod(received):
    """
    BPSK Demodulation
    """

    bits = (received >= 0).astype(np.uint8)

    return bits


#########################################################
# QPSK
#########################################################

def qpsk_mod(bits):
    """
    Gray-coded QPSK

    00 -> -1 -1j
    01 -> -1 +1j
    11 -> +1 +1j
    10 -> +1 -1j
    """

    bits = pad_bits(bits, 2)

    bits = bits.reshape((-1, 2))

    symbols = []

    for b0, b1 in bits:

        if (b0, b1) == (0, 0):
            s = -1 - 1j

        elif (b0, b1) == (0, 1):
            s = -1 + 1j

        elif (b0, b1) == (1, 1):
            s = 1 + 1j

        else:
            s = 1 - 1j

        symbols.append(s)

    symbols = np.array(symbols)

    # Normalize power
    symbols = symbols / np.sqrt(2)

    return symbols


def qpsk_demod(received):
    """
    Gray-coded QPSK Demodulator
    """

    received = received * np.sqrt(2)

    bits = []

    for s in received:

        i = s.real
        q = s.imag

        if i < 0 and q < 0:

            bits.extend([0, 0])

        elif i < 0 and q >= 0:

            bits.extend([0, 1])

        elif i >= 0 and q >= 0:

            bits.extend([1, 1])

        else:

            bits.extend([1, 0])

    return np.array(bits, dtype=np.uint8)


#########################################################
# Average Symbol Energy
#########################################################

def average_symbol_energy(symbols):
    """
    Compute average symbol energy.
    """

    return np.mean(np.abs(symbols) ** 2)


#########################################################
# Symbol Power
#########################################################

def normalize(symbols):
    """
    Normalize average symbol power to 1.
    """

    energy = average_symbol_energy(symbols)

    return symbols / np.sqrt(energy)


#########################################################
# Test Module
#########################################################

if __name__ == "__main__":

    np.random.seed(0)

    bits = np.random.randint(0, 2, 20)

    print("Original Bits")
    print(bits)

    print("\n----- BPSK -----")

    s = bpsk_mod(bits)

    r = bpsk_demod(s)

    print("Recovered Correctly:",
          np.array_equal(bits, r[:len(bits)]))

    print("\n----- QPSK -----")

    s = qpsk_mod(bits)

    r = qpsk_demod(s)

    print("Recovered Correctly:",
          np.array_equal(bits, r[:len(bits)]))

    #########################################################
# 16-QAM (Gray Coded)
#########################################################

# Gray-coded amplitude mapping
# Bits -> Level
# 00 -> -3
# 01 -> -1
# 11 -> +1
# 10 -> +3

_gray_to_level_16 = {
    (0, 0): -3,
    (0, 1): -1,
    (1, 1):  1,
    (1, 0):  3
}

_level_to_gray_16 = {
    -3: (0, 0),
    -1: (0, 1),
     1: (1, 1),
     3: (1, 0)
}


def qam16_mod(bits):
    """
    Gray-coded 16-QAM Modulator

    Every symbol contains 4 bits:
        b0 b1 -> I
        b2 b3 -> Q
    """

    bits = pad_bits(bits, 4)

    bits = bits.reshape((-1, 4))

    symbols = []

    for b in bits:

        i_bits = (b[0], b[1])
        q_bits = (b[2], b[3])

        i = _gray_to_level_16[i_bits]
        q = _gray_to_level_16[q_bits]

        symbols.append(complex(i, q))

    symbols = np.array(symbols)

    # Normalize average symbol energy
    symbols = symbols / np.sqrt(10)

    return symbols


def qam16_demod(received):
    """
    Gray-coded 16-QAM Demodulator
    """

    received = received * np.sqrt(10)

    bits = []

    for s in received:

        i = s.real
        q = s.imag

        #############################
        # Decision on I component
        #############################

        if i < -2:
            i_level = -3

        elif i < 0:
            i_level = -1

        elif i < 2:
            i_level = 1

        else:
            i_level = 3

        #############################
        # Decision on Q component
        #############################

        if q < -2:
            q_level = -3

        elif q < 0:
            q_level = -1

        elif q < 2:
            q_level = 1

        else:
            q_level = 3

        bits.extend(_level_to_gray_16[i_level])
        bits.extend(_level_to_gray_16[q_level])

    return np.array(bits, dtype=np.uint8)


#########################################################
# Test 16-QAM
#########################################################

if __name__ == "__main__":

    np.random.seed(42)

    bits = np.random.randint(0, 2, 40)

    tx = qam16_mod(bits)

    rx = qam16_demod(tx)

    print("\n----- 16-QAM -----")
    print("Recovered Correctly:",
          np.array_equal(bits, rx[:len(bits)]))

    #########################################################
# 64-QAM (Gray Coded)
#########################################################

# Gray coding (3 bits -> amplitude)
#
# Bits      Level
# 000  ->   -7
# 001  ->   -5
# 011  ->   -3
# 010  ->   -1
# 110  ->    1
# 111  ->    3
# 101  ->    5
# 100  ->    7

_gray_to_level_64 = {
    (0,0,0): -7,
    (0,0,1): -5,
    (0,1,1): -3,
    (0,1,0): -1,
    (1,1,0):  1,
    (1,1,1):  3,
    (1,0,1):  5,
    (1,0,0):  7
}

_level_to_gray_64 = {
    -7:(0,0,0),
    -5:(0,0,1),
    -3:(0,1,1),
    -1:(0,1,0),
     1:(1,1,0),
     3:(1,1,1),
     5:(1,0,1),
     7:(1,0,0)
}


def qam64_mod(bits):
    """
    Gray-coded 64-QAM Modulator

    Every symbol contains 6 bits

    First 3 bits -> I component
    Last 3 bits  -> Q component
    """

    bits = pad_bits(bits, 6)

    bits = bits.reshape((-1, 6))

    symbols = []

    for b in bits:

        i_bits = tuple(b[:3])
        q_bits = tuple(b[3:])

        i = _gray_to_level_64[i_bits]
        q = _gray_to_level_64[q_bits]

        symbols.append(complex(i, q))

    symbols = np.array(symbols)

    # Average symbol energy = 42
    symbols = symbols / np.sqrt(42)

    return symbols


def qam64_demod(received):
    """
    Gray-coded 64-QAM Demodulator
    """

    received = received * np.sqrt(42)

    bits = []

    levels = np.array([-7,-5,-3,-1,1,3,5,7])

    for s in received:

        i = s.real
        q = s.imag

        # Nearest neighbour decision
        i_level = levels[np.argmin(np.abs(levels - i))]
        q_level = levels[np.argmin(np.abs(levels - q))]

        bits.extend(_level_to_gray_64[int(i_level)])
        bits.extend(_level_to_gray_64[int(q_level)])

    return np.array(bits, dtype=np.uint8)


#########################################################
# 64-QAM TEST
#########################################################

if __name__ == "__main__":

    np.random.seed(0)

    bits = np.random.randint(0,2,120)

    tx = qam64_mod(bits)

    rx = qam64_demod(tx)

    print("\n-----64-QAM-----")
    print("Recovered Correctly:",
          np.array_equal(bits, rx[:len(bits)]))