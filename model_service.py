import numpy as np
import joblib
from keras.models import load_model
from typing import List, Dict, Any

loaded_model = load_model("MDeveloper.keras")
loaded_scaler = joblib.load("scaler.joblib")

FEATURE_ORDER = [
    "T_mem_in",
    "T_con_in",
    "S",
    "v_chan",
    "vac",
    "L_type",
    "sp_type",
    "spa_type",
    "membrane",
]

def predict_vagmd(inputs: dict) -> dict:
    x = np.array([[
        inputs["T_mem_in"],
        inputs["T_con_in"],
        inputs["S"],
        inputs["v_chan"],
        inputs["vac"],
        inputs["L_type"],
        inputs["sp_type"],
        inputs["spa_type"],
        inputs["membrane"],
    ]], dtype=float)

    x_scaled = loaded_scaler.transform(x)
    y_pred = loaded_model.predict(x_scaled, verbose=0)

    return {
        "Tcond_pred": float(y_pred[0, 0]),
        "Flux_pred": float(y_pred[0, 1]),
    }


def validate_case_inputs(inputs: dict) -> List[str]:
    warnings = []

    if not (50 <= inputs["T_mem_in"] <= 80):
        warnings.append("T_mem_in outside training range (50–80 °C)")
    if not (20 <= inputs["T_con_in"] <= 40):
        warnings.append("T_con_in outside training range (20–40 °C)")
    if not (0 <= inputs["S"] <= 299):
        warnings.append("Salinity outside training range (0–299 g/kg)")
    if not (0.01 <= inputs["v_chan"] <= 0.079):
        warnings.append("Channel velocity outside training range (0.01–0.079 m/s)")
    if not (20000 <= inputs["vac"] <= 99000):
        warnings.append("Vacuum pressure outside training range (20k–99k Pa)")

    return warnings

def compare_vagmd_cases(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compare multiple V-AGMD cases using MDeveloper.
    Each case = {"name": str, "inputs": dict}
    """

    if not cases:
        raise ValueError("At least one case must be provided.")

    results = []

    # Run predictions
    for case in cases:
        name = case["name"]
        inputs = case["inputs"]

        pred = predict_vagmd(inputs)
        warnings = validate_case_inputs(inputs)

        results.append({
            "name": name,
            "inputs": inputs,
            "outputs": pred,
            "warnings": warnings
        })

    # Baseline = first case
    baseline_flux = results[0]["outputs"]["Flux_pred"]
    baseline_tcond = results[0]["outputs"]["Tcond_pred"]

    # Compute deltas
    for r in results:
        flux = r["outputs"]["Flux_pred"]
        tcond = r["outputs"]["Tcond_pred"]

        r["comparison_vs_baseline"] = {
            "delta_flux": flux - baseline_flux,
            "delta_tcond": tcond - baseline_tcond,
            "percent_flux_change": (
                100 * (flux - baseline_flux) / baseline_flux
                if baseline_flux != 0 else None
            ),
            "percent_tcond_change": (
                100 * (tcond - baseline_tcond) / baseline_tcond
                if baseline_tcond != 0 else None
            )
        }

    # Rank by flux
    ranked_by_flux = sorted(
        [
            {
                "name": r["name"],
                "Flux_pred": r["outputs"]["Flux_pred"]
            }
            for r in results
        ],
        key=lambda x: x["Flux_pred"],
        reverse=True
    )

    return {
        "baseline_case": results[0]["name"],
        "num_cases": len(results),
        "results": results,
        "ranked_by_flux": ranked_by_flux
    }