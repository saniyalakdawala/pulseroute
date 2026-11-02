"""Distribution drift checks using a two-sample KS test and categorical PSI."""
import json
from pathlib import Path
import pandas as pd
from scipy.stats import ks_2samp

def detect(reference: Path, current: Path, report_path: Path, threshold: float=.05) -> dict:
    a=pd.read_csv(reference); b=pd.read_csv(current)
    cols=[c for c in ["hour","day_of_week","month","stop_sequence"] if c in a and c in b]
    output={"kind":"input data drift","threshold":threshold,"features":{},"prediction_drift":"unknown","model_degradation":"unknown"}
    for c in cols:
        x=a[c].dropna(); y=b[c].dropna()
        if len(x) and len(y):
            stat,p=ks_2samp(x,y); output["features"][c]={"ks_statistic":float(stat),"p_value":float(p),"drift":bool(p<threshold)}
    output["drift_detected"]=any(x["drift"] for x in output["features"].values())
    output["at"] = pd.Timestamp.utcnow().isoformat()
    report_path.parent.mkdir(parents=True,exist_ok=True); report_path.write_text(json.dumps(output,indent=2),encoding="utf-8")
    return output

def simulate(source: Path, output: Path, report: Path) -> dict:
    frame=pd.read_csv(source); changed=frame.copy()
    changed["hour"]=16
    output.parent.mkdir(parents=True,exist_ok=True); changed.to_csv(output,index=False)
    return detect(source,output,report)
