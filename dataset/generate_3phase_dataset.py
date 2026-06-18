"""
Synthetic dataset for a THREE-PHASE grid-tied inverter fault detection/classification.

Fault classes:
    0 - Healthy
    1 - Open-Circuit Switch Fault (one IGBT stuck open -> phase current distortion)
    2 - DC Injection Fault (DC bias in one or more phase currents)
    3 - Grid Voltage Sag/Swell (grid disturbance on one or all phases)
    4 - Islanding (loss of grid, frequency drifts)
    5 - PLL Desynchronization (phase angle drift)

Feature vector per sample (three-phase extended):
    Vdc          - DC link voltage (V)
    e_a, e_b, e_c        - three-phase grid voltages (V)
    i_a, i_b, i_c        - three-phase inverter output currents (A)
    i_a_ref, i_b_ref, i_c_ref   - three-phase reference currents (A)
    err_a, err_b, err_c  - per-phase tracking errors
    i_a_prev, i_b_prev, i_c_prev - previous sample currents
    P            - three-phase active power estimate
    Q            - three-phase reactive power estimate
    freq_est     - PLL estimated frequency (Hz)
    THD_est      - estimated THD (%)

Total: 21 features + label columns
Rows: 120,000 (2,400 windows x 50 samples, 400 windows per class)
"""

import numpy as np
import pandas as pd

# ---------- Simulation parameters ----------
fs        = 5000       # logging rate (Hz)
f_grid    = 50         # nominal grid frequency (Hz)
Vgrid_amp = 311.0      # phase voltage amplitude (~220 Vrms)
Vdc_nom   = 700.0      # DC link voltage (typical 3-phase grid-tied)
I_amp_nom = 15.0       # nominal current amplitude per phase (~3kW/phase)

window_size          = 50    # samples per window (10 ms)
n_windows_per_class  = 400   # windows per fault class

rng = np.random.default_rng(42)

fault_labels = {
    0: "Healthy",
    1: "Open_Circuit_Switch_Fault",
    2: "DC_Injection_Fault",
    3: "Grid_Voltage_Sag_Swell",
    4: "Islanding",
    5: "PLL_Desync",
}

rows = []

for label, name in fault_labels.items():
    for w in range(n_windows_per_class):

        # Randomize operating point slightly
        Vdc    = Vdc_nom + rng.normal(0, 5)
        I_amp  = I_amp_nom * rng.uniform(0.6, 1.0)
        phase0 = rng.uniform(0, 2 * np.pi)   # initial phase angle

        # Three-phase offsets: a=0, b=-120, c=+120 degrees
        phase_offsets = np.array([0, -2*np.pi/3, 2*np.pi/3])

        # Default (healthy) parameters
        f_actual      = f_grid
        Vgrid_scale   = np.ones(3)   # per-phase scale
        dc_offset     = np.zeros(3)  # per-phase DC offset in current
        fault_phase   = None
        islanded      = False
        pll_drift     = 0.0

        # Configure fault-specific conditions
        if label == 1:   # Open-circuit: one phase loses positive or negative half-cycle
            fault_phase = rng.integers(0, 3)   # which phase (0=a, 1=b, 2=c)

        elif label == 2:   # DC injection: 1-2 phases get a DC bias
            n_affected = rng.integers(1, 3)
            affected   = rng.choice(3, size=n_affected, replace=False)
            for ph in affected:
                dc_offset[ph] = rng.uniform(1.0, 4.0) * rng.choice([-1, 1])

        elif label == 3:   # Sag/swell: can be single-phase or all-phase
            if rng.random() > 0.5:   # single-phase sag/swell
                ph = rng.integers(0, 3)
                Vgrid_scale[ph] = rng.choice([rng.uniform(0.5, 0.85),
                                               rng.uniform(1.1, 1.3)])
            else:   # three-phase sag/swell
                scale = rng.choice([rng.uniform(0.5, 0.85),
                                     rng.uniform(1.1, 1.3)])
                Vgrid_scale[:] = scale

        elif label == 4:   # Islanding: frequency drifts, voltage decays
            islanded  = True
            f_actual  = f_grid + rng.uniform(-2.0, 2.0)

        elif label == 5:   # PLL desync: phase angle error
            pll_drift = rng.uniform(0.3, 1.2) * rng.choice([-1, 1])

        # Time array for this window
        t = (np.arange(window_size) + w * window_size) / fs

        # Build per-phase signals
        e   = np.zeros((3, window_size))   # grid voltages
        i_L = np.zeros((3, window_size))   # inverter currents
        i_ref = np.zeros((3, window_size)) # reference currents

        for ph in range(3):
            ang = 2 * np.pi * f_grid * t + phase0 + phase_offsets[ph]

            # Grid voltage
            if islanded:
                decay  = np.exp(-3 * (t - t[0]))
                e[ph]  = Vgrid_amp * np.sin(2*np.pi*f_actual*t + phase0 + phase_offsets[ph]) * decay
            else:
                e[ph]  = Vgrid_amp * Vgrid_scale[ph] * np.sin(ang)

            # Reference current (in phase with grid voltage for unity PF)
            i_ref[ph] = I_amp * np.sin(ang)

            # Actual inverter current
            i_L[ph] = i_ref[ph].copy()

            if label == 1 and ph == fault_phase:
                # Clip positive half-cycle to ~0 (open-circuit fault)
                i_L[ph] = np.where(np.sin(ang) > 0, 0.05 * i_ref[ph], i_ref[ph])

            if label == 2:
                i_L[ph] = i_L[ph] + dc_offset[ph]

            if label == 3:
                # Current tracking degrades proportionally to voltage deviation
                i_L[ph] = i_ref[ph] * (1 - 0.25 * (1 - Vgrid_scale[ph]))

            if label == 4:
                i_L[ph] = I_amp * np.sin(2*np.pi*f_actual*t + phase0 + phase_offsets[ph]) \
                           * (1 + 0.08*np.sin(2*np.pi*3*t))

            if label == 5:
                i_L[ph] = I_amp * np.sin(ang + pll_drift)

            # Add measurement noise
            e[ph]     += rng.normal(0, 2.0,  window_size)
            i_L[ph]   += rng.normal(0, 0.2,  window_size)
            i_ref[ph] += rng.normal(0, 0.05, window_size)

        # Previous sample (shift by 1)
        i_prev = np.roll(i_L, 1, axis=1)
        i_prev[:, 0] = i_prev[:, 1]

        # Tracking errors
        err = i_ref - i_L

        # Three-phase instantaneous power
        P = np.sum(e * i_L, axis=0)   # sum of phase powers
        Q = np.sum(e * np.roll(i_L, window_size//4, axis=1), axis=0)  # approx reactive

        # THD estimate (fault-dependent)
        if label in (1, 4):
            thd_base = rng.uniform(10, 20)
        elif label == 5:
            thd_base = rng.uniform(5, 10)
        elif label == 2:
            thd_base = rng.uniform(4, 8)
        else:
            thd_base = rng.uniform(1.5, 4.0)
        thd_est = thd_base + rng.normal(0, 0.4, window_size)

        # PLL frequency estimate
        if label == 4:
            freq_est = f_actual + rng.normal(0, 0.1, window_size)
        elif label == 5:
            freq_est = f_grid   + rng.normal(0, 0.05, window_size)
        else:
            freq_est = f_grid   + rng.normal(0, 0.01, window_size)

        for k in range(window_size):
            rows.append({
                # Operating condition
                "window_id"  : label * n_windows_per_class + w,
                "sample_idx" : k,
                "Vdc"        : Vdc,
                # Three-phase grid voltages
                "e_a"        : e[0, k],
                "e_b"        : e[1, k],
                "e_c"        : e[2, k],
                # Three-phase inverter currents
                "i_a"        : i_L[0, k],
                "i_b"        : i_L[1, k],
                "i_c"        : i_L[2, k],
                # Reference currents
                "i_a_ref"    : i_ref[0, k],
                "i_b_ref"    : i_ref[1, k],
                "i_c_ref"    : i_ref[2, k],
                # Tracking errors
                "err_a"      : err[0, k],
                "err_b"      : err[1, k],
                "err_c"      : err[2, k],
                # Previous sample currents
                "i_a_prev"   : i_prev[0, k],
                "i_b_prev"   : i_prev[1, k],
                "i_c_prev"   : i_prev[2, k],
                # Power and frequency
                "P"          : P[k],
                "Q"          : Q[k],
                "freq_est"   : freq_est[k],
                "THD_est"    : thd_est[k],
                # Labels
                "fault_label": label,
                "fault_name" : name,
            })

df = pd.DataFrame(rows)

out_path = "/home/claude/dataset/inverter_3phase_fault_dataset.csv"
df.to_csv(out_path, index=False)

print(f"Saved: {out_path}")
print(f"Total rows    : {len(df):,}")
print(f"Total windows : {df['window_id'].nunique():,}")
print(f"Features      : {len(df.columns)-2} (excl. label columns)")
print(f"\nClass distribution:")
print(df.groupby(['fault_label','fault_name']).size().reset_index(name='rows').to_string(index=False))
print(f"\nSample (first 3 rows):")
print(df.head(3).to_string(index=False))
