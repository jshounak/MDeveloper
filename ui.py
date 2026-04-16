import os

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
import altair as alt

load_dotenv(override=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MDeveloper", layout="wide")
st.title("MDeveloper: V-AGMD Performance Tool")

st.sidebar.header("Operating Conditions")

T_mem_in = st.sidebar.number_input("Feed inlet temp (°C)", 50.0, 90.0, 80.0)
T_con_in = st.sidebar.number_input("Condenser inlet temp (°C)", 10.0, 40.0, 25.0)
S = st.sidebar.number_input("Salinity (g/kg)", 0.0, 300.0, 35.0)
v_chan = st.sidebar.number_input("Channel velocity (m/s)", 0.01, 0.1, 0.04)
vac = st.sidebar.number_input("Vacuum pressure (Pa)", 20000, 100000, 20000)
L_type = st.sidebar.selectbox("Module type", [0, 1])
sp_type = st.sidebar.selectbox("Spacer type", [0, 1])
spa_type = st.sidebar.selectbox("Spacer arrangement", [0, 1])
membrane = st.sidebar.selectbox("Membrane", [0, 1])

payload = {
    "T_mem_in": T_mem_in,
    "T_con_in": T_con_in,
    "S": S,
    "v_chan": v_chan,
    "vac": vac,
    "L_type": L_type,
    "sp_type": sp_type,
    "spa_type": spa_type,
    "membrane": membrane,
}

st.header("Single Prediction")

if st.button("Predict Performance"):
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        st.subheader("Results")
        st.metric("Distillate Flux (LMH)", round(result["Flux_pred"], 2))
        st.metric("Condenser Outlet Temp (°C)", round(result["Tcond_pred"], 2))
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
    except KeyError as exc:
        st.error(f"Unexpected response format: missing {exc}")

st.header("Compare Salinity Sweep")

salinity_text = st.text_input(
    "Enter salinity values for comparison (g/kg, comma-separated)",
    value="35, 70, 100"
)

try:
    salinities = [
        float(x.strip()) for x in salinity_text.split(",") if x.strip() != ""
    ]
    salinities = sorted(list(set(salinities)))
except ValueError:
    st.error("Please enter valid numeric salinity values separated by commas.")
    salinities = []


if st.button("Run Comparison"):
    try:
        cases = [
            {
                "name": f"S = {salinity}",
                "inputs": {
                    **payload,
                    "S": salinity,
                },
            }
            for salinity in salinities
        ]

        response = requests.post(
            f"{API_URL}/compare",
            json={"cases": cases},
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        df = pd.DataFrame(
            [
                {
                    "Case": result["name"],
                    "Salinity": result["inputs"]["S"],
                    "Flux": result["outputs"]["Flux_pred"],
                    "Condenser Outlet Temperature": result["outputs"]["Tcond_pred"],
                }
                for result in data["results"]
            ]
        ).sort_values("Salinity")

        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Condenser Outlet Temperature vs Salinity")
            temp_fig = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("Salinity:Q", title="Salinity (g/kg)"),
                y=alt.Y("Condenser Outlet Temperature:Q", title="Condenser Outlet Temperature (C)"),
                tooltip=["Salinity", "Condenser Outlet Temperature"]
            ).properties(height=350)
            st.altair_chart(temp_fig, use_container_width=True)

        with col2:
            st.subheader("Flux vs Salinity")
            flux_fig = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("Salinity:Q", title="Salinity (g/kg)"),
                y=alt.Y("Flux:Q", title="Flux (LMH)"),
                tooltip=["Salinity", "Flux"]
            ).properties(height=350)
            st.altair_chart(flux_fig, use_container_width=True)

    except requests.RequestException as exc:
        st.error(f"Comparison request failed: {exc}")
    except KeyError as exc:
        st.error(f"Unexpected response format: missing {exc}")

st.header("Ask MDeveloper")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask about V-AGMD performance...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={"message": user_input},
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        reply = data.get("response", "No response returned from backend.")
    except requests.RequestException as exc:
        reply = f"Chat request failed: {exc}"

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        st.write(reply)