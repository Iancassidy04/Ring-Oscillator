# Ian Cassidy
# VLSI, Fall 2026
# designs from design_optimization.py for select n are checked across process corners, supply variation, and load cap variation

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import csv
import os

import design_optimization as DO

N_LIST = [3, 7, 11, 21, 35, 51]     # Stage counts to compare (must be odd)
VDD_VAR = 0.05          # Supply variation (+/- 5%)
C_VAR = 0.10            # CEXT variation (+/- 10%)

# Report figure settings
FIG_DPI = 300           # PNG resolution for the report
SHOW_TITLES = True      # Set False if the report uses captions instead of plot titles
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9, "savefig.bbox": "tight"})

# Process corners, 1.1V Standard-Vt, W/L = 10u/0.04u
# Source: gpdk045 PDK Model Report Tables 2.5 (NMOS) and 2.7 (PMOS)
# FS = fast NMOS / slow PMOS, SF = slow NMOS / fast PMOS
CORNERS = {
    "TT": dict(VTN=0.38, IDN=5.96e-3, VTP=-0.43, IDP=3.06e-3),
    "FF": dict(VTN=0.32, IDN=6.68e-3, VTP=-0.38, IDP=3.42e-3),
    "SS": dict(VTN=0.43, IDN=5.23e-3, VTP=-0.48, IDP=2.69e-3),
    "FS": dict(VTN=0.35, IDN=6.31e-3, VTP=-0.46, IDP=2.875e-3),  # PMOS Idsat listed as 0 in report, interpolated SS/TT
    "SF": dict(VTN=0.41, IDN=5.59e-3, VTP=-0.41, IDP=3.24e-3),
}

def F_AND_D_CORNER(W_n, W_p, C_add, n, corner="TT", VDD=DO.VDD):
    c = CORNERS[corner]

    # Refit k' at this corner (Idsat spec taken at nominal VDD)
    kn = 2 * c["IDN"] * DO.L_SPEC / (DO.W_SPEC * (DO.VDD - c["VTN"])**2)
    kp = 2 * c["IDP"] * DO.L_SPEC / (DO.W_SPEC * (DO.VDD - np.abs(c["VTP"]))**2)
    V_ovn = VDD - c["VTN"]
    V_ovp = VDD - np.abs(c["VTP"])

    CL = DO.CL_CALC(W_n, W_p, C_add)
    t_pHL = 2 * CL * VDD / (kn * (W_n / DO.L_n) * V_ovn**2)
    t_pLH = 2 * CL * VDD / (kp * (W_p / DO.L_p) * V_ovp**2)
    F = 1 / (n * (t_pHL + t_pLH))
    DUTY = ((n + 1) * t_pHL + (n - 1) * t_pLH) / (2 * n * (t_pHL + t_pLH))
    return F, DUTY

# Design and evaluate each n
RESULTS = {}
for n in N_LIST:
    if n % 2 == 0:
        print(f"n = {n}: skipped, an even number of inverters latches instead of oscillating")
        continue

    OUT = DO.DESIGN(n)
    if OUT is None:
        print(f"n = {n}: ring too slow to reach {DO.f/1e9:.1f} GHz")
        continue
    W_n, W_p, C_add, _ = OUT

    R = {"Wn": W_n, "Wp": W_p, "C": C_add,
         "P": DO.POWER(W_n, W_p, C_add, n),
         "AREA": n * (W_n * DO.L_n + W_p * DO.L_p),     # Total gate area (m^2)
         "C_TOT": n * C_add}                            # Total added cap (F)

    # Process corners
    R["F_C"], R["D_C"] = {}, {}
    for c in CORNERS:
        R["F_C"][c], R["D_C"][c] = F_AND_D_CORNER(W_n, W_p, C_add, n, c)

    F_TT = R["F_C"]["TT"]
    # Sensitivity at TT, percent change in f
    R["SENS"] = {
        f"VDD +{VDD_VAR*100:.0f}%": 100 * (F_AND_D_CORNER(W_n, W_p, C_add, n, "TT", DO.VDD * (1 + VDD_VAR))[0] / F_TT - 1),
        f"VDD -{VDD_VAR*100:.0f}%": 100 * (F_AND_D_CORNER(W_n, W_p, C_add, n, "TT", DO.VDD * (1 - VDD_VAR))[0] / F_TT - 1),
        f"CEXT +{C_VAR*100:.0f}%": 100 * (F_AND_D_CORNER(W_n, W_p, C_add * (1 + C_VAR), n)[0] / F_TT - 1),
        f"CEXT -{C_VAR*100:.0f}%": 100 * (F_AND_D_CORNER(W_n, W_p, C_add * (1 - C_VAR), n)[0] / F_TT - 1),
    }

    F_ALL = np.array(list(R["F_C"].values()))
    D_ALL = np.array(list(R["D_C"].values()))
    R["F_SPREAD"] = 100 * (F_ALL.max() - F_ALL.min()) / DO.f     # Corner f spread (%)
    R["D_WORST"] = 100 * np.max(np.abs(D_ALL - DO.D))              # Worst corner |D - 0.5| (% pts)
    RESULTS[n] = R

NS = list(RESULTS)
SK = list(RESULTS[NS[0]]["SENS"])

# Comparison table (printed and saved as CSV for the report)
rows = [
    ("Wn (nm)",                     lambda R: f"{R['Wn']*1e9:.0f}"),
    ("Wp (nm)",                     lambda R: f"{R['Wp']*1e9:.0f}"),
    ("CEXT per stage (fF)",         lambda R: f"{R['C']*1e15:.4f}"),
    ("Total CEXT (fF)",             lambda R: f"{R['C_TOT']*1e15:.2f}"),
    ("Total gate area (um^2)",      lambda R: f"{R['AREA']*1e12:.3f}"),
    ("Power @ 1 GHz (uW)",          lambda R: f"{R['P']*1e6:.1f}"),
]
rows += [(f"f @ {c} (GHz)", lambda R, c=c: f"{R['F_C'][c]/1e9:.4f}") for c in CORNERS]
rows += [(f"D @ {c}",       lambda R, c=c: f"{R['D_C'][c]:.4f}") for c in CORNERS]
rows += [("Corner f spread (%)",          lambda R: f"{R['F_SPREAD']:.1f}"),
         ("Worst corner |D-0.5| (% pts)", lambda R: f"{R['D_WORST']:.2f}")]
rows += [(f"df/f for {k} (%)", lambda R, k=k: f"{R['SENS'][k]:+.2f}") for k in SK]

print("Design Comparison")
print("-" * (32 + 12 * len(NS)))
print(f"{'':32}" + "".join(f"{'n = ' + str(n):>12}" for n in NS))
for name, FMT in rows:
    print(f"{name:32}" + "".join(f"{FMT(RESULTS[n]):>12}" for n in NS))

script_dir = Path(__file__).parent
folder_path = script_dir / "RING_OSC_RESULTS"
os.makedirs(folder_path, exist_ok=True)

with open(folder_path / "compare_table.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["Parameter"] + [f"n = {n}" for n in NS])
    for name, FMT in rows:
        w.writerow([name] + [FMT(RESULTS[n]) for n in NS])


# Report figures (one per file)
COLORS = plt.cm.viridis(np.linspace(0, 0.85, len(NS)))
x = np.arange(len(CORNERS))
bw = 0.8 / len(NS)
OFFSET = bw * (len(NS) - 1) / 2

def SAVE(fig, name, title):
    if SHOW_TITLES:
        fig.axes[0].set_title(title)
    fig.savefig(folder_path / f"{name}.png", dpi=FIG_DPI)
    fig.savefig(folder_path / f"{name}.pdf")

# Figure 1: frequency at each corner
fig, ax = plt.subplots(figsize=(7, 4))
for i, n in enumerate(NS):
    ax.bar(x + i * bw, [RESULTS[n]["F_C"][c] / 1e9 for c in CORNERS], bw, color=COLORS[i], label=f"n = {n}")
ax.axhline(DO.f / 1e9, color="k", ls="--", lw=0.8, label="Target")
ax.set_xticks(x + OFFSET, list(CORNERS))
ax.set_xlabel("Process Corner")
ax.set_ylabel("Frequency (GHz)")
ax.set_ylim(0.8, 1.2)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(ncol=4, loc="upper right")
SAVE(fig, "fig_corner_frequency", "Oscillation Frequency Across Process Corners")

# Figure 2: duty cycle error at each corner
fig, ax = plt.subplots(figsize=(7, 4))
for i, n in enumerate(NS):
    ax.bar(x + i * bw, [100 * (RESULTS[n]["D_C"][c] - DO.D) for c in CORNERS], bw, color=COLORS[i], label=f"n = {n}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x + OFFSET, list(CORNERS))
ax.set_xlabel("Process Corner")
ax.set_ylabel("D - 0.5 (% points)")
ax.grid(True, axis="y", alpha=0.3)
ax.legend(ncol=3)
SAVE(fig, "fig_corner_duty_cycle", "Duty Cycle Error Across Process Corners")

# Figure 3: sensitivity to supply and load cap
xs = np.arange(len(SK))
fig, ax = plt.subplots(figsize=(7, 4))
for i, n in enumerate(NS):
    ax.bar(xs + i * bw, [RESULTS[n]["SENS"][k] for k in SK], bw, color=COLORS[i], label=f"n = {n}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xs + OFFSET, SK)
ax.set_ylabel("Frequency Change (%)")
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] * 1.45)       # Headroom for legend
ax.grid(True, axis="y", alpha=0.3)
ax.legend(ncol=3, loc="upper center")
SAVE(fig, "fig_sensitivity", "Frequency Sensitivity at TT")

# Figure 4: cost, power and total added cap
xn = np.arange(len(NS))
fig, ax = plt.subplots(figsize=(7, 4))
ax2 = ax.twinx()
b1 = ax.bar(xn - 0.2, [RESULTS[n]["P"] * 1e6 for n in NS], 0.4, color="C3", label="Power")
b2 = ax2.bar(xn + 0.2, [RESULTS[n]["C_TOT"] * 1e15 for n in NS], 0.4, color="C7", label="Total CEXT")
ax.set_xticks(xn, [f"n = {n}" for n in NS])
ax.set_ylabel("Power @ 1 GHz (uW)", color="C3")
ax2.set_ylabel("Total CEXT (fF)", color="C7")
ax.set_ylim(0, ax.get_ylim()[1] * 1.2)                      # Headroom for legend
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.2)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(handles=[b1, b2], loc="upper center", ncol=2)
SAVE(fig, "fig_cost", "Power and Added Capacitance")

# Figure 5: robustness vs n, worst corner duty cycle error and CEXT sensitivity
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(NS, [RESULTS[n]["D_WORST"] for n in NS], marker="o", color="C1",
        label="Worst corner |D - 0.5| (% points)")
ax.plot(NS, [np.abs(RESULTS[n]["SENS"][f"CEXT +{C_VAR*100:.0f}%"]) for n in NS], marker="s", color="C2",
        label=f"|df/f| for CEXT +{C_VAR*100:.0f}% (%)")
ax.set_xlabel("Number of Stages (n)")
ax.set_ylabel("Error (%)")
ax.set_xticks(NS)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.3)
ax.legend()
SAVE(fig, "fig_robustness_vs_n", "Duty Cycle and Load Sensitivity vs Stage Count")

print(f"\nSaved figures (PNG + PDF) and compare_table.csv to {folder_path}")

plt.show()