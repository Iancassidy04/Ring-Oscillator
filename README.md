# Ring Oscillator Design and Verification

CMOS ring oscillators designed in Python and simulated in Cadence Virtuoso (gpdk045, 45 nm) for a 1 GHz, 50% duty cycle output across stage counts n = 3 to 51.

EE5410 VLSI Circuit Design, Fall 2026.

## Files

| File | Purpose |
| --- | --- |
| `design_optimization.py` | Calculates W_n, W_p (5 nm grid) and C_EXT for each n from hand delay equations |
| `compre_optimization.py` | Checks selected designs across process corners, VDD and C_EXT variation |
| `plot_CSV.py` | Measures frequency, duty cycle, edges and start-up from a Cadence transient CSV |
| `RingOSC_data.csv` | Exported transient data for the simulated rings |
| `RING_OSC_RESULTS/` | Generated figures (PNG + PDF) and result tables |
| `Cadence Imagers/` | Schematic screenshots |
| `Simulation Plots/` | Simulation figures |

## Usage

Requires Python 3.10+ with `numpy` and `matplotlib`.

```
python design_optimization.py     # design table for every odd n
python compre_optimization.py     # corner and sensitivity comparison
python plot_CSV.py                # analyze RingOSC_data.csv
```

Device parameters come from the gpdk045 PDK Model Report (TT corner); the oxide thickness is assumed.

## Author

**Ian Cassidy**, Electrical Engineering Masters Student, University of Vermont
