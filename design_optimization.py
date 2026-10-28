# Ian Cassidy
# VLSI, Fall 2026
# Ring Oscillator Design using GPDK045 in Cadence Virtuoso

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os

# Design targets
f = 1e9
D = 0.5
N = 51

# Tuning options for external capacitance
TUNE_CEXT = True
C_FINE = 1e-19
C_COARSE = 0.1e-15

# Width selection
D_GOAL = 1e-4           # Duty cycle error goal
W_SEARCH_MAX = 2e-6     # Largest W_n tested

# Parameters for wide/short W/L (10u/0.04u) devices
IDSATN = 5.96e-3
IDSATP = 3.06e-3
L_SPEC = 0.04e-6    # L for Idsat spec
W_SPEC = 10e-6      # W for Idsat spec
VTP = -0.43             
VTN = 0.38
VDD = 1.1

# Physical constraints
L_n = 45e-9
L_p = 45e-9
WMIN = 120e-9
W_GRID = 5e-9   # gpdk_045 W resolution limit

EPS_OX = 3.9 * 8.854e-12
TOXE = 1.8e-9           # Estimate of electrical oxide thicknes

V_ovn = VDD - VTN
V_ovp = VDD - np.abs(VTP)

# Square law fit to Idsat so hand calc matches simulator at VDD
kn = 2 * IDSATN * L_SPEC / (W_SPEC * V_ovn**2)
kp = 2 * IDSATP * L_SPEC / (W_SPEC * V_ovp**2)

Coxn = EPS_OX / TOXE
Coxp = EPS_OX / TOXE

# Junction cap per unit width: Cj*Ldiff + 2*Cjsw  (perimeter ~ 2W)
LDIFF = 0.14e-6
CJ_N = 1.177e-3
CJ_P = 1.308e-3

CJSW_N = 6.562e-11
CJSW_P = 7.604e-11

C_jn = CJ_N * LDIFF + 2 * CJSW_N
C_jp = CJ_P * LDIFF + 2 * CJSW_P


# Model equations
def CL_CALC(W_n, W_p, C_add):
    # Full capacitance includes gate of next stage, drains of current stage, and added load
    return Coxn * W_n * L_n + Coxp * W_p * L_p + C_jn * W_n + C_jp * W_p + C_add


def DELAYS(W_n, W_p, C_add):
    CL = CL_CALC(W_n, W_p, C_add)
    t_pHL = 2 * CL * VDD / (kn * (W_n / L_n) * V_ovn**2)
    t_pLH = 2 * CL * VDD / (kp * (W_p / L_p) * V_ovp**2)
    return t_pHL, t_pLH

def F_AND_D(W_n, W_p, C_add, n):
    t_pHL, t_pLH = DELAYS(W_n, W_p, C_add)
    F = 1 / (n * (t_pHL + t_pLH))
    DUTY = ((n + 1) * t_pHL + (n - 1) * t_pLH) / (2 * n * (t_pHL + t_pLH))
    return F, DUTY


def POWER(W_n, W_p, C_add, n):
    return n * CL_CALC(W_n, W_p, C_add) * VDD**2 * f

def SNAP(x, grid):
    return np.round(x / grid) * grid

def R_FROM_D(n):
    return (n - 1 - 2 * n * D) / (2 * n * D - n - 1)


def DESIGN(n):
    # Calculate exact W_p / W_n for target D
    RATIO = R_FROM_D(n) * (kn * V_ovn**2 * L_p) / (kp * V_ovp**2 * L_n)

    # Try W_n on the 5 nm grid, round W_p to the grid
    best = None

    for W_n in np.arange(WMIN, W_SEARCH_MAX + W_GRID / 2, W_GRID):
        W_p = max(SNAP(RATIO * W_n, W_GRID), WMIN)
        _, DUTY = F_AND_D(W_n, W_p, 0, n)
        err = np.abs(DUTY - D) / D

        if best is None or err < best[0] - 1e-15:
            best = (err, W_n, W_p)

        if err <= D_GOAL:
            break

    _, W_n, W_p = best  # Keep the smallest pair that meets D_GOAL or the best one found

    # Calculate CEXT so the total load gives f exactly
    t_pHL, t_pLH = DELAYS(W_n, W_p, 0)
    CL_DEV = CL_CALC(W_n, W_p, 0)

    CL_REQ = CL_DEV / (n * f * (t_pHL + t_pLH))
    C_EXACT = CL_REQ - CL_DEV

    if C_EXACT < 0:
        return None

    C_add = SNAP(C_EXACT, C_FINE if TUNE_CEXT else C_COARSE)
    return W_n, W_p, C_add, C_EXACT


# Don't run for design comparison
if __name__ == "__main__":

    STAGES = np.arange(3, N + 1, 2)
    KEYS = ["Wn", "Wp", "C", "P", "EF", "ED", "EF_COARSE"]
    R = {k: [] for k in KEYS}

    print(f"Ring Oscillator Design (VDD = {VDD} V, L = {L_n*1e9:.0f} nm, W grid = {W_GRID*1e9:.0f} nm, "
          f"CEXT {'tuned' if TUNE_CEXT else 'coarse'})")
    
    print("-" * 92)

    print(f"{'n':>4} {'Wn (nm)':>8} {'Wp (nm)':>8} {'CEXT (fF)':>10} {'f (GHz)':>10} {'D':>9} "
          f"{'f err %':>9} {'D err %':>9} {'P (uW)':>8}")

    for n in STAGES:
        OUT = DESIGN(n)
        if OUT is None:
            print(f"{n:>4}  ring too slow to reach {f/1e9:.1f} GHz")
            for k in KEYS:
                R[k].append(np.nan)
            continue

        W_n, W_p, C_add, C_EXACT = OUT

        F0, D0 = F_AND_D(W_n, W_p, C_add, n)
        F1, _ = F_AND_D(W_n, W_p, SNAP(C_EXACT, C_COARSE), n)

        VALS = [W_n, W_p, C_add, POWER(W_n, W_p, C_add, n),
                100 * (F0 - f) / f, 100 * (D0 - D) / D, 100 * (F1 - f) / f]
        
        for k, v in zip(KEYS, VALS):
            R[k].append(v)

        print(f"{n:>4} {W_n*1e9:>8.0f} {W_p*1e9:>8.0f} {C_add*1e15:>10.4f} {F0/1e9:>10.6f} {D0:>9.6f} "
              f"{VALS[4]:>9.4f} {VALS[5]:>9.4f} {VALS[3]*1e6:>8.2f}")

    R = {k: np.array(v, dtype=float) for k, v in R.items()}

    # Plots
    script_dir = Path(__file__).parent
    folder_path = script_dir / "RING_OSC_RESULTS"
    os.makedirs(folder_path, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    ax[0, 0].plot(STAGES, R["Wn"] * 1e9, marker="o", ms=3, label="$W_n$")
    ax[0, 0].plot(STAGES, R["Wp"] * 1e9, marker="s", ms=3, label="$W_p$")
    ax[0, 0].set_ylabel("Width (nm)")
    ax[0, 0].set_title("Widths (5 nm grid)")
    ax[0, 0].legend()

    ax[0, 1].plot(STAGES, R["C"] * 1e15, marker="o", ms=3, color="C2")
    ax[0, 1].set_ylabel("CEXT per Stage (fF)")
    ax[0, 1].set_title("Added Load")

    ax[1, 0].plot(STAGES, np.abs(R["EF_COARSE"]), marker="s", ms=3, color="C7",
                  label=f"CEXT rounded to {C_COARSE*1e15:.1f} fF")
    ax[1, 0].plot(STAGES, np.abs(R["EF"]), marker="o", ms=3, color="C0",
                  label=f"CEXT {'tuned' if TUNE_CEXT else 'coarse'}")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_ylabel("|Frequency Error| (%)")
    ax[1, 0].set_title("Frequency Error")
    ax[1, 0].legend()

    ax[1, 1].plot(STAGES, np.abs(R["ED"]), marker="o", ms=3, color="C1")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylabel("|Duty Cycle Error| (%)")
    ax[1, 1].set_title("Duty Cycle Error (set by W_p / W_n)")

    for a in ax.flat:
        a.grid(True, which="both", alpha=0.3)
    for a in ax[1]:
        a.set_xlabel("Number of Stages (n)")

    fig.tight_layout()
    fig.savefig(folder_path / "ring_osc_design.png", dpi=150)

    plt.show()