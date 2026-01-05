from __future__ import annotations
import os
import streamlit as st
from openai import OpenAI

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "local-dev-key")
MODEL = os.environ.get("MODEL_NAME", "gpt2")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

st.set_page_config(page_title="GPT-2 124M UI (vLLM)", layout="wide")
st.title("GPT-2 124M - CPU UI (OpenAI-compatible server)")

prompt = st.text_area("Prompt", value="Write a concise AWS incident summary:", height=150)
col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("max_tokens", 16, 512, 256)
with col2:
    temperature = st.slider("temperature", 0.0, 1.5, 0.7)

if st.button("Generate"):
    with st.spinner("Generating..."):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    st.subheader("Response")
    st.write(resp.choices[0].message.content)
