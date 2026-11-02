import json
import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
project=Path(__file__).resolve().parents[2]
load_dotenv(project/".env")
p=Path(os.getenv("PULSEROUTE_ARTIFACTS",str(project/"artifacts")))/"registry.json"
st.title("Model lifecycle")
if not p.exists(): st.info("Registry appears after the first training run.")
else:
    data=json.loads(p.read_text()); st.dataframe(data["versions"],use_container_width=True); st.subheader("Promotion history"); st.json(data["history"])
