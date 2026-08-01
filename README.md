# Adaptive Modem BER Replication (Python)

Companion Python simulation for the minor project:
"Replication of MATLAB-Standard BER Results for Adaptive Modulation Using Python SDR Tools"

## Folder structure
```
images/                 - put your test image here as image.png (a placeholder is included)
results/                - all generated plots are saved here
adaptive.py             - SNR->modulation switching table (Table 1 / 2.1) and throughput
ber.py                  - theoretical BER formulas (Eq. 1.1-1.4) + Monte-Carlo BER simulation
channel.py              - AWGN channel model
image_processing.py     - image <-> bitstream, frame-based adaptive image transmission
main.py                 - runs everything end-to-end
modulation.py           - BPSK/QPSK/16QAM/64QAM modulator & demodulator
plots.py                - recreates Fig. 2,3,4,6,7,8 from the reference paper
utils.py                - Q-function, MSE/PSNR, bit helpers
```

## How to run
```bash
pip install numpy scipy matplotlib pillow
python main.py
```

This will:
1. Plot theoretical BER curves for BPSK/QPSK/16-QAM/64-QAM vs SNR (Fig. 2)
2. Plot the adaptive modem's theoretical BER curve for 3 target BERs (Fig. 3)
3. Plot BPS throughput of adaptive modulation (Fig. 4)
4. Run a Monte-Carlo simulation to validate the theoretical BER curve (Fig. 7)
5. Run an adaptive vs. fixed-modulation image transmission experiment and
   plot PSNR vs. channel SNR + a visual image comparison (Fig. 6, Fig. 8)

Replace `images/image.png` with your own picture (any format) for the
image-transmission experiment — it's converted to 128x128 grayscale
automatically.

## Notes for your report
- BER formulas are implemented exactly as given in Eq. (1.1)-(1.4) of the
  reference paper (Q-function based expressions for BPSK, QPSK, 16-QAM).
  The 64-QAM formula uses the standard closed-form AWGN approximation
  consistent with the structure in Eq. (1.4).
- The switching table in adaptive.py is copied verbatim from Table 1 /
  Table 2.1 (three target BERs: 0.1, 0.01, 0.001).
- Use ber.simulate_ber() / simulated_ber_curve() with a larger n_bits
  (e.g. 10^5-10^6, per your proposal's Monte Carlo validation strategy)
  for smoother curves — the default in main.py is kept lower for speed.
