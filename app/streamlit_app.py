"""Streamlit prediction and lifecycle overview."""
import os, json
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pulseroute.model_loading import load_model

st.set_page_config(page_title="PulseRoute AI",page_icon="🚇",layout="wide")
st.title("🚇 PulseRoute AI")
st.caption("Transit delay risk demo using recorded MBTA subway stop performance.")
root=Path(__file__).resolve().parents[1]
load_dotenv(root/".env")
delay_threshold=float(os.getenv("PULSEROUTE_DELAY_THRESHOLD_MIN","5"))
artifact_dir=Path(os.getenv("PULSEROUTE_ARTIFACTS",str(root/"artifacts")))
model_path=Path(os.getenv("PULSEROUTE_MODEL",str(artifact_dir/"champion.joblib")))
if not model_path.exists():
    st.warning("No champion model is available yet. Follow the training steps in README.md.")
else:
    try:
        model=load_model(model_path)
    except Exception as exc:
        st.error(f"The saved model is incompatible with this app environment: {exc}")
        st.info("Use the pinned requirements.txt and restart or redeploy the app.")
        st.stop()
    route=st.text_input("Route ID", "Red")
    col1,col2,col3=st.columns(3)
    direction=col1.selectbox("Direction",["0","1"]); seq=col2.number_input("Stop sequence",1,300,10); hour=col3.slider("Scheduled hour",0,23,8)
    day=st.selectbox("Day of week",list(range(7)),format_func=lambda x:["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])
    month=st.slider("Month",1,12,6)
    if st.button("Estimate delay risk",type="primary"):
        row=pd.DataFrame([{"route_id":route,"direction_id":direction,"stop_sequence":seq,"hour":hour,"day_of_week":day,"month":month}])
        p=float(model.predict_proba(row)[0,1]); st.metric(f"Probability of arriving >{delay_threshold:g} minutes late",f"{p:.1%}")
report_path=artifact_dir/"evaluation.json"
if report_path.exists():
    report=json.loads(report_path.read_text()); st.subheader("Latest holdout evaluation")
    st.caption(f"Evaluation dataset source: {report.get('dataset_source_kind', 'unspecified')}. Synthetic fallback scores are not real-world performance.")
    st.write(f"Selected candidate: **{report['winner']}** · temporal holdout: {report['holdout_rows']} records beginning {report['holdout_start']}")
    st.json(report["metrics"][report["winner"]])
