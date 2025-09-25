# A tiny real-time loudness “barometer.” 
Captures mic input, computes A-weighted LAeq in short windows, and paints the window blue to red from quiet to loud. A white dot flashes on new peaks.

## Features

- A-weighting applied in the frequency domain
- Adjustable analysis window
- Peak flash indicator (simple relative peak)
- Lightweight GUI via pygame

## Controls
`Q` / `ESC` = quit
`+` / `-` = adjust CAL_DB_OFFSET (rough dB alignment)
`1` / `2` = shrink/grow analysis window (faster vs steadier)


## Notes & Calibration
- LAeq here is relative without a reference. If you have a known level (e.g., a calibrated SLM reading or a 1 kHz sine at a known dB SPL), nudge `CAL_DB_OFFSET` to line up.
- Sample rate default is 48 kHz; change to 44.1 kHz if your device prefers it.

## Troubleshooting
- No audio / device error: Select a specific input device in the code (`DEVICE = <index or name>`).
- Choppy display: Try a longer window (press `2`) or lower sample rate.
