import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model

# --- Plot styling ---
textsize = 14
plt.rcParams.update({
    #'font.family': 'Calibri',
    'font.size': textsize,
    'axes.titlesize': textsize,
    'axes.labelsize': textsize,
    'legend.fontsize': textsize,
    'xtick.labelsize': textsize,
    'ytick.labelsize': textsize,
    "axes.linewidth": 2.5
})

# --- 1. Load Model and Scaler ---
try:
    loaded_model = load_model('MDeveloper.keras', compile=False)
    loaded_scaler = joblib.load('scaler.joblib')
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    print("Please ensure 'MDeveloper.keras' and 'scaler.joblib' are in the current directory.")
    raise SystemExit

# ---------------------------------------------------------------------
# 2. Define process-variable ranges and fixed operating point
# ---------------------------------------------------------------------
# Feature order (must match training):
# [T_mem_in, T_con_in, S, v_chan, vac, L_type, sp_type, spa_type, membrane]

ranges = {
    "T_mem_in": np.arange(50.0, 81.0, 0.5),
    "T_con_in": np.arange(20.0, 41.0, 0.5),
    "S":        np.arange(0.0, 300.0, 1.0),
    "v_chan":   np.arange(0.01, 0.08, 0.001),
    "vac":      np.arange(2e4, 1e5, 1000.0)
}

# Fixed base condition (around which each variable is swept)
fixed_values = {
    "T_mem_in": 80.0,
    "T_con_in": 20.0,
    "S":        35.0,
    "v_chan":   0.04,
    "vac":      2e4,
    "L_type":   1,
    "sp_type":  0,
    "spa_type": 0,
    "membrane": 0
}

xlabels = {
    "T_mem_in": r"$T_{\mathrm{mem,in}}$ (°C)",
    "T_con_in": r"$T_{\mathrm{con,in}}$ (°C)",
    "S":        r"$S$ (g/kg)",        # adjust to your units
    "v_chan":   r"$v_{\mathrm{chan}}$ (m/s)",
    "vac":      r"$p_{\mathrm{vac}}$ (Pa)"
}

# ---------------------------------------------------------------------
# 3. Dictionaries to store all predictions (nothing gets overwritten)
# ---------------------------------------------------------------------
all_flux_pred  = {}   # e.g. all_flux_pred["T_mem_in"] -> array
all_tcond_pred = {}   # e.g. all_tcond_pred["T_mem_in"] -> array
all_ranges     = {}   # store the x-values used

# ---------------------------------------------------------------------
# 4. Sweep each variable and make one figure per sweep
# ---------------------------------------------------------------------
for var_name, var_range in ranges.items():
    print(f"\nSweeping variable: {var_name}")
    n = len(var_range)

    # Start from fixed conditions
    T_mem_in = np.full(n, fixed_values["T_mem_in"])
    T_con_in = np.full(n, fixed_values["T_con_in"])
    S        = np.full(n, fixed_values["S"])
    v_chan   = np.full(n, fixed_values["v_chan"])
    vac      = np.full(n, fixed_values["vac"])
    L_type   = np.full(n, fixed_values["L_type"])
    sp_type  = np.full(n, fixed_values["sp_type"])
    spa_type = np.full(n, fixed_values["spa_type"])
    membrane = np.full(n, fixed_values["membrane"])

    # Replace the swept variable with its range
    if var_name == "T_mem_in":
        T_mem_in = var_range
    elif var_name == "T_con_in":
        T_con_in = var_range
    elif var_name == "S":
        S = var_range
    elif var_name == "v_chan":
        v_chan = var_range
    elif var_name == "vac":
        vac = var_range
    else:
        raise ValueError(f"Unknown variable name: {var_name}")

    # Build input matrix in the correct column order
    X_new_raw = np.column_stack((
        T_mem_in,
        T_con_in,
        S,
        v_chan,
        vac,
        L_type,
        sp_type,
        spa_type,
        membrane
    ))

    print(f"  X_new_raw shape for {var_name}: {X_new_raw.shape}")

    # Scale and predict
    X_new_scaled = loaded_scaler.transform(X_new_raw)
    Y_pred_new   = loaded_model.predict(X_new_scaled, verbose=0)

    Tcond_pred = Y_pred_new[:, 0]
    Flux_pred  = Y_pred_new[:, 1]

    # ---- Store in dictionaries so they are not overwritten ----
    all_flux_pred[var_name]  = Flux_pred
    all_tcond_pred[var_name] = Tcond_pred
    all_ranges[var_name]     = var_range

    # ---- Plot for this sweep ----
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = var_range
    xlabel = xlabels.get(var_name, var_name)

    # Flux on primary axis
    color_flux = 'tab:blue'
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Predicted Flux (units)', color=color_flux)  # adjust units
    line1, = ax1.plot(x, Flux_pred, color=color_flux, linewidth=3,
                      label='Predicted Flux')
    ax1.tick_params(axis='y', labelcolor=color_flux)
    ax1.grid(True, which='major', axis='x', linestyle='--', alpha=0.5)

    # Tcond on secondary axis
    ax2 = ax1.twinx()
    color_tcond = 'tab:red'
    ax2.set_ylabel('Predicted $T_{cond}$ (°C)', color=color_tcond)
    line2, = ax2.plot(x, Tcond_pred, color=color_tcond, linewidth=3,
                      linestyle='--', label='Predicted $T_{cond}$')
    ax2.tick_params(axis='y', labelcolor=color_tcond)

    # Title (mention sweep + fixed values)
    plt.title(
        f'Model Predictions vs. {var_name}\n'
        f'(Fixed: $T_{{mem,in}}$={fixed_values["T_mem_in"]}°C, '
        f'$T_{{con,in}}$={fixed_values["T_con_in"]}°C, '
        f'$S$={fixed_values["S"]}, $v_{{chan}}$={fixed_values["v_chan"]}, '
        f'$p_{{vac}}$={fixed_values["vac"]}, etc.)'
    )

    # Combine legends
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best')

    plt.tight_layout()
    plt.show()
