import streamlit as st
import requests
import pandas as pd
from dotenv import load_dotenv
import os
from openai import OpenAI


load_dotenv(override=True)

API_URL = "http://localhost:8000"

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

if st.button("Predict Performance"):
    payload = {
        "T_mem_in": T_mem_in,
        "T_con_in": T_con_in,
        "S": S,
        "v_chan": v_chan,
        "vac": vac,
        "L_type": L_type,
        "sp_type": sp_type,
        "spa_type": spa_type,
        "membrane": membrane
    }

    response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()

        st.subheader("Results")

        st.metric("Distillate Flux (LMH)", round(result["Flux_pred"], 2))
        st.metric("Condenser Outlet Temp (°C)", round(result["Tcond_pred"], 2))

    else:
        st.error("Prediction failed")


    st.header("Compare Salinity Sweep")

salinities = st.multiselect(
    "Select salinity values (g/kg)",
    [35, 70, 100, 140],
    default=[35, 70, 100]
)

if st.button("Run Comparison"):

    cases = []
    for s in salinities:
        cases.append({
            "name": f"S = {s}",
            "inputs": {
                "T_mem_in": T_mem_in,
                "T_con_in": T_con_in,
                "S": s,
                "v_chan": v_chan,
                "vac": vac,
                "L_type": L_type,
                "sp_type": sp_type,
                "spa_type": spa_type,
                "membrane": membrane
            }
        })

    response = requests.post(f"{API_URL}/compare", json={"cases": cases})

    if response.status_code == 200:
        data = response.json()

        df = pd.DataFrame([
            {
                "Case": r["name"],
                "Flux": r["outputs"]["Flux_pred"],
                "T_cond": r["outputs"]["Tcond_pred"]
            }
            for r in data["results"]
        ])

        st.dataframe(df)

        st.line_chart(df.set_index("Case"))

st.header("Ask MDeveloper")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask about V-AGMD performance...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = requests.post(
        f"{API_URL}/chat",
        json={"message": user_input}
    )

    data = response.json()
    st.write(data)  # debug
    reply = data.get("response", "No 'response' key found in backend output.")

    st.session_state.messages.append({"role": "assistant", "content": reply})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])