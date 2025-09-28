# Sound Barometer (LAeq → Color) 
# Minimal real-time "barometer" that maps A-weighted loudness to a color bar.
# Blue = quiet, Red = loud. 
# Includes a simple peak-hold flash.

# Dependencies:
# pip install numpy sounddevice pygame

# Notes / Calibration:
#  This computes a RELATIVE LAeq from your mic signal using an A-weighting curve
#  applied in the frequency domain. Without a reference calibration (e.g., a
#  known dB(A) tone), values are qualitative. Adjust CAL_DB_OFFSET to
#  roughly align with a known level.

# Controls:
#   ESC or Q = quit
#   +/- = adjust CAL_DB_OFFSET (rough calibration offset in dB)
#   1/2 = change analysis window (seconds)

import sys
import time
import numpy as np
import sounddevice as sd
import pygame

# Config
FS = 48000                # sample rate (Hz); Use 44100 if device prefers it.
WINDOW_SEC = 1.0          # analysis window length in seconds
CAL_DB_OFFSET = 0.0       # manual dB offset to "calibrate" the readout (relative)
PEAK_HOLD_SEC = 0.75      # duration of little white flash when a higher peak occurs
DEVICE = None             # set to an int or string to choose a specific input device

# A-weighting (frequency-domain curve)
def a_weight_db(f):
    """Return A-weighting in dB for array of frequencies f (Hz)."""
    f = np.asarray(f, dtype=np.float64)
    # Avoid division by zero at DC
    f = np.maximum(f, 1e-6)
    ra = (12194**2 * f**4) / (
        (f**2 + 20.6**2)
        * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
        * (f**2 + 12194**2)
    )
    return 20.0 * np.log10(ra) + 2.0

def color_from_db(db, db_lo=35, db_hi=85):
    """Map dB(A) to an RGB color from blue to green to yellow to red.
    db_lo maps to blue; db_hi maps to red.
    """
    x = np.clip((db - db_lo) / max(1e-9, (db_hi - db_lo)), 0.0, 1.0)
 
  # Gradient: blue(0) → cyan(0.33) → green(0.5) → yellow(0.66) → red(1.0)
  # Piecewise RGB interpolation.
    if x < 0.33:
        # blue (0,0,255) → cyan (0,255,255)
        t = x / 0.33
        r = 0
        g = int(255 * t)
        b = 255
    
    elif x < 0.5:
        # cyan (0,255,255) → green (0,255,0)
        t = (x - 0.33) / (0.5 - 0.33)
        r = 0
        g = 255
        b = int(255 * (1 - t))
    
    elif x < 0.66:
        # green (0,255,0) → yellow (255,255,0)
        t = (x - 0.5) / (0.66 - 0.5)
        r = int(255 * t)
        g = 255
        b = 0
    
    else:
        # yellow (255,255,0) → red (255,0,0)
        t = (x - 0.66) / (1.0 - 0.66)
        r = 255
        g = int(255 * (1 - t))
        b = 0
    return (r, g, b)

def compute_laeq_db(samples, fs):
    """Compute a relative LAeq (dB) from time-domain samples.
    Steps: FFT power spectrum → apply A-weighting (in linear) → sum power → 10*log10.
    """
    # Hann window to reduce spectral leakage
    win = np.hanning(len(samples))
    xw = samples * win
    
    # FFT (rfft for real input)
    X = np.fft.rfft(xw)
    
    # Power spectrum (proportional)
    P = (np.abs(X) ** 2) / np.sum(win**2)
    freqs = np.fft.rfftfreq(len(samples), d=1.0/fs)

    # A-weight in dB to linear power factor
    A_db = a_weight_db(freqs)
    A_lin = 10.0 ** (A_db / 10.0)

    # Weighted power (relative)
    Pw = P * A_lin

    # Avoid log of zero
    Pw_sum = np.maximum(Pw.sum(), 1e-20)
    laeq_db = 10.0 * np.log10(Pw_sum) + CAL_DB_OFFSET
    return laeq_db

def main():
    global CAL_DB_OFFSET, WINDOW_SEC

    # Pygame setup
    pygame.init()
    w, h = 640, 240
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption("Sound Thermometer — LAeq → Color")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    # Audio stream
    frames_per_block = int(FS * WINDOW_SEC)

    stream = sd.InputStream(
        samplerate=FS,
        blocksize=frames_per_block,
        channels=1,
        dtype="float32",
        device=DEVICE,
    )
    stream.start()

    peak_seen = -1e9
    peak_flash_until = 0.0

    try:
        while True:
            # Read audio
            data, _ = stream.read(frames_per_block)
            mono = data[:, 0].astype(np.float64)

            # Metrics
            laeq = compute_laeq_db(mono, FS)

            # "Peak" as simple sample peak in dBFS (relative); not true dBZpk.
            # Convert to "dB relative" by 20*log10(max(abs(x))+eps). Offset to roughly align.
            peak_rel = 20.0 * np.log10(np.max(np.abs(mono)) + 1e-12) + 100.0 + CAL_DB_OFFSET

            if peak_rel > peak_seen + 0.1:  # update if noticeably higher
                peak_seen = peak_rel
                peak_flash_until = time.time() + PEAK_HOLD_SEC

            # Draw
            screen.fill((0, 0, 0))
            color = color_from_db(laeq)
            pygame.draw.rect(screen, color, pygame.Rect(0, 0, w, h))

            # Flash indicator for peak
            if time.time() < peak_flash_until:
                pygame.draw.circle(screen, (255, 255, 255), (w - 30, 30), 10)

            # Text overlays
            txt = font.render(f"LAeq (rel): {laeq:5.1f} dB(A)   Peak (rel): {peak_seen:5.1f} dB", True, (0, 0, 0))
            screen.blit(txt, (12, 12))
            txt2 = font.render(f"Window={WINDOW_SEC:.2f}s   Offset={CAL_DB_OFFSET:+.1f} dB   FS={FS}", True, (0, 0, 0))
            screen.blit(txt2, (12, 42))

            pygame.display.flip()
            clock.tick(30)

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        raise KeyboardInterrupt
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        CAL_DB_OFFSET += 1.0
                    elif event.key == pygame.K_MINUS:
                        CAL_DB_OFFSET -= 1.0
                    elif event.key == pygame.K_1:
                        WINDOW_SEC = max(0.1, WINDOW_SEC * 0.5)
                        frames_per_block = int(FS * WINDOW_SEC)
                    elif event.key == pygame.K_2:
                        WINDOW_SEC = min(2.0, WINDOW_SEC * 1.5)
                        frames_per_block = int(FS * WINDOW_SEC)

    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        pygame.quit()

if __name__ == "__main__":
    main()
