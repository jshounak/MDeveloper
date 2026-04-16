from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "MDeveloper.keras"
SCALER_PATH = BASE_DIR / "scaler.joblib"

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

_loaded_model = None
_loaded_scaler = None


def get_model():
    global _loaded_model
    if _loaded_model is None:
        print("Loading Keras model...")
        from keras.models import load_model
        _loaded_model = load_model(MODEL_PATH, compile=False)
        print("Keras model loaded.")
    return _loaded_model


def get_scaler():
    global _loaded_scaler
    if _loaded_scaler is None:
        print("Loading scaler...")
        _loaded_scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded.")
    return _loaded_scaler


def validate_required_inputs(inputs: dict) -> None:
    missing = [key for key in FEATURE_ORDER if key not in inputs]
    if missing:
        raise ValueError(f"Missing required inputs: {', '.join(missing)}")


def predict_vagmd(inputs: dict) -> dict:
    validate_required_inputs(inputs)

    x = np.array(
        [[float(inputs[key]) for key in FEATURE_ORDER]],
        dtype=float,
    )

    scaler = get_scaler()
    model = get_model()

    x_scaled = scaler.transform(x)
    y_pred = model.predict(x_scaled, verbose=0)

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
    if not cases:
        raise ValueError("At least one case must be provided.")

    results = []

    for case in cases:
        name = case["name"]
        inputs = case["inputs"]

        pred = predict_vagmd(inputs)
        warnings = validate_case_inputs(inputs)

        results.append(
            {
                "name": name,
                "inputs": inputs,
                "outputs": pred,
                "warnings": warnings,
            }
        )

    baseline_flux = results[0]["outputs"]["Flux_pred"]
    baseline_tcond = results[0]["outputs"]["Tcond_pred"]

    for result in results:
        flux = result["outputs"]["Flux_pred"]
        tcond = result["outputs"]["Tcond_pred"]

        result["comparison_vs_baseline"] = {
            "delta_flux": flux - baseline_flux,
            "delta_tcond": tcond - baseline_tcond,
            "percent_flux_change": (
                100 * (flux - baseline_flux) / baseline_flux
                if baseline_flux != 0
                else None
            ),
            "percent_tcond_change": (
                100 * (tcond - baseline_tcond) / baseline_tcond
                if baseline_tcond != 0
                else None
            ),
        }

    ranked_by_flux = sorted(
        [
            {
                "name": result["name"],
                "Flux_pred": result["outputs"]["Flux_pred"],
            }
            for result in results
        ],
        key=lambda x: x["Flux_pred"],
        reverse=True,
    )

    return {
        "baseline_case": results[0]["name"],
        "num_cases": len(results),
        "results": results,
        "ranked_by_flux": ranked_by_flux,
    }