# Ian Cassidy
# VLSI, Fall 2026
# Ring Oscillator Simulation Analysis from Cadence transient CSV

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import csv
import os

CSV_FILE = "RingOSC_data.csv"       # Exported transient data (same folder as this script)

NET_NAMES = {
    "/net97": "N = 3",
    "/net3":  "N = 5",
    "/net11": "N = 7",
    "/net21": "N = 19",
    "/net43": "N = 51",
}

# Targets (from design_optimization.py) and measurement settings
F_TARGET = 1e9
D_TARGET = 0.5
VDD = 1.1
N_MEASURE = 10
SETTLE_TOL = 0.01
N_ZOOM = 3

# Report figure settings
FIG_DPI = 300
SHOW_TITLES = True
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 9, "savefig.bbox": "tight"})

# Read CSV
def READ_CSV(path):
    with open(path) as fcsv:
        header = next(csv.reader(fcsv))
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    NETS = {}
    for k in range(0, len(header), 2):
        net = header[k].rsplit(" ", 1)[0]
        t, v = data[:, k], data[:, k + 1]

        keep = ~np.isnan(t) & ~np.isnan(v)
        NETS[net] = (t[keep], v[keep])
    return NETS

# Threshold crossings, linearly interpolated between samples
def CROSSINGS(t, v, th):
    above = v >= th
    up = np.where(~above[:-1] & above[1:])[0]
    dn = np.where(above[:-1] & ~above[1:])[0]

    interp = lambda i: t[i] + (th - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i])
    return interp(up), interp(dn)

# 10% to 90% rise and fall times on each edge
def EDGE_TIMES(t, v, v_lo, v_hi):
    lo, hi = v_lo + 0.1 * (v_hi - v_lo), v_lo + 0.9 * (v_hi - v_lo)

    up_lo, dn_lo = CROSSINGS(t, v, lo)
    up_hi, dn_hi = CROSSINGS(t, v, hi)

    rise = [h - l for l in up_lo for h in up_hi[up_hi > l][:1]]
    fall = [l - h for h in dn_hi for l in dn_lo[dn_lo > h][:1]]

    return np.array(rise), np.array(fall)

def MEASURE(t, v):
    th = VDD / 2
    t_up, t_dn = CROSSINGS(t, v, th)

    if len(t_up) < N_MEASURE + 2:
        return None

    PERIODS = np.diff(t_up)
    T_FINAL = np.mean(PERIODS[-N_MEASURE:])

    # Start-up: first rising edge after which every period stays within SETTLE_TOL
    ok = np.abs(PERIODS / T_FINAL - 1) <= SETTLE_TOL
    bad = np.where(~ok)[0]

    i0 = bad[-1] + 1 if len(bad) else 0
    T_START = t_up[i0]

    # Duty cycle per period
    win = t_up[-(N_MEASURE + 1):]

    HIGH = []
    for a, b in zip(win[:-1], win[1:]):
        falls = t_dn[(t_dn > a) & (t_dn < b)]
        if len(falls):
            HIGH.append(falls[0] - a)

    HIGH = np.array(HIGH)
    PER = np.diff(win)[:len(HIGH)]

    # Swing and edges over the measurement window
    m = t >= win[0]
    V_MAX, V_MIN = v[m].max(), v[m].min()
    RISE, FALL = EDGE_TIMES(t[m], v[m], V_MIN, V_MAX)

    F = 1 / T_FINAL
    D = np.mean(HIGH / PER)
    return {"F": F, "T": T_FINAL, "D": D, "T_STD": np.std(PERIODS[-N_MEASURE:]),
            "T_HIGH": np.mean(HIGH), "T_START": T_START, "V_MAX": V_MAX, "V_MIN": V_MIN,
            "T_RISE": np.mean(RISE) if len(RISE) else np.nan,
            "T_FALL": np.mean(FALL) if len(FALL) else np.nan,
            "EF": 100 * (F - F_TARGET) / F_TARGET, "ED": 100 * (D - D_TARGET) / D_TARGET,
            "t_up": t_up, "t_dn": t_dn, "TH": th}


# Analyze every net
script_dir = Path(__file__).parent
NETS = READ_CSV(script_dir / CSV_FILE)
RESULTS = {}

for net, (t, v) in NETS.items():
    R = MEASURE(t, v)

    if R is None:
        print(f"{net}: fewer than {N_MEASURE + 2} rising edges, not analyzed")
        continue

    RESULTS[net] = R

ORDER = sorted(RESULTS, key=lambda k: RESULTS[k]["F"])  # Slowest to fastest
LABEL = lambda net: NET_NAMES.get(net, net)

print(f"Simulation results (last {N_MEASURE} periods, threshold {VDD/2:.2f} V)")
print("-" * 108)
print(f"{'Net':>10} {'f (GHz)':>9} {'f err %':>8} {'T (ps)':>8} {'D (%)':>7} {'D err %':>8} "
      f"{'Rise (ps)':>10} {'Fall (ps)':>10} {'Swing (V)':>14} {'Start-up (ns)':>14} {'T std (fs)':>11}")

for net in ORDER:
    R = RESULTS[net]

    print(f"{LABEL(net):>10} {R['F']/1e9:>9.4f} {R['EF']:>+8.2f} {R['T']*1e12:>8.1f} {100*R['D']:>7.2f} "
          f"{R['ED']:>+8.2f} {R['T_RISE']*1e12:>10.1f} {R['T_FALL']*1e12:>10.1f} "
          f"{R['V_MIN']:>6.3f}-{R['V_MAX']:<6.3f}  {R['T_START']*1e9:>13.2f} {R['T_STD']*1e15:>11.1f}")

folder_path = script_dir / "RING_OSC_RESULTS"
os.makedirs(folder_path, exist_ok=True)

with open(folder_path / "sim_results.csv", "w", newline="") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["Net", "f (GHz)", "f error (%)", "Period (ps)", "Duty cycle (%)", "D error (%)",
                "Rise 10-90 (ps)", "Fall 90-10 (ps)", "Vmin (V)", "Vmax (V)", "Start-up (ns)", "Period std (fs)"])
    
    for net in ORDER:
        R = RESULTS[net]
        w.writerow([LABEL(net), f"{R['F']/1e9:.4f}", f"{R['EF']:.2f}", f"{R['T']*1e12:.1f}",
                    f"{100*R['D']:.2f}", f"{R['ED']:.2f}", f"{R['T_RISE']*1e12:.1f}", f"{R['T_FALL']*1e12:.1f}",
                    f"{R['V_MIN']:.3f}", f"{R['V_MAX']:.3f}", f"{R['T_START']*1e9:.2f}", f"{R['T_STD']*1e15:.1f}"])


# Plots
COLORS = dict(zip(ORDER, plt.cm.viridis(np.linspace(0, 0.85, len(ORDER)))))

def SAVE(fig, name, title):
    if SHOW_TITLES:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(folder_path / f"{name}.png", dpi=FIG_DPI)
    fig.savefig(folder_path / f"{name}.pdf")

# Full transient for every net with start-up marked
fig, ax = plt.subplots(len(ORDER), 1, figsize=(8, 1.6 * len(ORDER) + 0.6), sharex=True)
ax = np.atleast_1d(ax)

for a, net in zip(ax, ORDER):
    t, v = NETS[net]
    R = RESULTS[net]

    a.plot(t * 1e9, v, color=COLORS[net], lw=0.8)
    a.axvline(R["T_START"] * 1e9, color="k", ls="--", lw=0.8)
    a.text(R["T_START"] * 1e9, 0.55, f" settled {R['T_START']*1e9:.1f} ns", va="center", fontsize=8)
    
    a.set_ylabel(f"{LABEL(net)}\n(V)", fontsize=9)
    a.set_ylim(-0.15, VDD + 0.15)
    a.grid(True, alpha=0.3)

ax[-1].set_xlabel("Time (ns)")
fig.tight_layout()
SAVE(fig, "sim_transient_all", "Transient Output")

# Zoomed steady state with annotated period, high time, f and D
fig, ax = plt.subplots(len(ORDER), 1, figsize=(8, 1.9 * len(ORDER) + 0.6))
ax = np.atleast_1d(ax)

for a, net in zip(ax, ORDER):
    t, v = NETS[net]
    R = RESULTS[net]

    t_up, t_dn, th = R["t_up"], R["t_dn"], R["TH"]
    t0, t1 = t_up[-(N_ZOOM + 1)], t_up[-1]
    pad = 0.15 * R["T"]

    m = (t >= t0 - pad) & (t <= t1 + pad)
    a.plot(t[m] * 1e9, v[m], color=COLORS[net], lw=1.2)

    a.axhline(th, color="gray", ls=":", lw=0.8)
    a.text((t1 + pad) * 1e9, th, " VDD/2", va="center", fontsize=8, color="gray")

    # One annotated period from rise to fall and rise to rise
    r0 = t_up[-2]
    f0 = t_dn[t_dn > r0][0]

    r1 = t_up[-1]
    y_T, y_H = VDD + 0.08, th

    a.annotate("", xy=(r1 * 1e9, y_T), xytext=(r0 * 1e9, y_T),
               arrowprops=dict(arrowstyle="<->", color="k", lw=0.8))
    
    a.text((r0 + r1) / 2 * 1e9, y_T + 0.03, f"T = {R['T']*1e12:.1f} ps", ha="center", va="bottom", fontsize=8)
    a.annotate("", xy=(f0 * 1e9, y_H), xytext=(r0 * 1e9, y_H),
               arrowprops=dict(arrowstyle="<->", color="C3", lw=0.8))
    
    a.text((r0 + f0) / 2 * 1e9, y_H + 0.05, f"t_high = {R['T_HIGH']*1e12:.1f} ps",
           ha="center", va="bottom", fontsize=8, color="C3")
    
    for x in (r0, f0, r1):
        a.plot(x * 1e9, th, "o", ms=3, color="k")

    a.text(0.01, 0.95, f"{LABEL(net)}:  f = {R['F']/1e9:.4f} GHz ({R['EF']:+.2f}%),  "
                      f"D = {100*R['D']:.2f}% ({R['ED']:+.2f}%)",
           transform=a.transAxes, va="top", fontsize=9,
           bbox=dict(boxstyle="round", fc="white", ec="0.8"))
    
    a.set_ylim(-0.15, VDD + 0.35)
    a.set_ylabel("V")
    a.grid(True, alpha=0.3)

ax[-1].set_xlabel("Time (ns)")
fig.tight_layout()
SAVE(fig, "sim_steady_state_annotated", f"Frequency and Duty Cycle at Steady State")

# Frequency and duty cycle vs target
x = np.arange(len(ORDER))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].bar(x, [RESULTS[n]["F"] / 1e9 for n in ORDER], color=[COLORS[n] for n in ORDER])
ax[0].axhline(F_TARGET / 1e9, color="k", ls="--", lw=0.8, label="Target")

for i, n in enumerate(ORDER):
    ax[0].text(i, RESULTS[n]["F"] / 1e9, f"{RESULTS[n]['EF']:+.1f}%", ha="center", va="bottom", fontsize=9)

ax[0].set_ylabel("Frequency (GHz)")
ax[0].set_ylim(0, 1.25 * max(RESULTS[n]["F"] for n in ORDER) / 1e9)
ax[0].legend(loc="lower right")

ax[1].bar(x, [100 * RESULTS[n]["D"] for n in ORDER], color=[COLORS[n] for n in ORDER])
ax[1].axhline(100 * D_TARGET, color="k", ls="--", lw=0.8, label="Target")

for i, n in enumerate(ORDER):
    ax[1].text(i, 100 * RESULTS[n]["D"], f"{RESULTS[n]['ED']:+.2f}%", ha="center", va="bottom", fontsize=9)

D_ALL = [100 * RESULTS[n]["D"] for n in ORDER]
span = max(2.0, max(abs(d - 50) for d in D_ALL) * 1.6)

ax[1].set_ylim(50 - span, 50 + span)
ax[1].set_ylabel("Duty Cycle (%)")
ax[1].legend(loc="lower right")

for a in ax:
    a.set_xticks(x, [LABEL(n) for n in ORDER])
    a.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
SAVE(fig, "sim_f_d_vs_target", "Simulated Frequency and Duty Cycle vs Target")

print(f"\nSaved figures (PNG + PDF) and sim_results.csv to {folder_path}")

plt.show()