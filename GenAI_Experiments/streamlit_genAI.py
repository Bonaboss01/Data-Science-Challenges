import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.header("🧠 GenAI Forecast Copilot")

question = st.text_input("Ask a question about forecasts")

if st.button("Ask"):
    res = requests.post(
        f"{API_BASE}/genai/ask",
        json={"question": question}
    )

    if res.status_code == 200:
        st.write(res.json()["answer"])
    else:
        st.error("GenAI service error")
