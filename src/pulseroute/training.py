"""Temporal training, real MLflow runs, and versioned local artifacts."""
from __future__ import annotations
import json, os, hashlib
import tempfile
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["route_id", "direction_id", "stop_sequence", "hour", "day_of_week", "month"]
NUM = ["stop_sequence", "hour", "day_of_week", "month"]
CAT = ["route_id", "direction_id"]

def make_pipeline(model):
    prep = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM), ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT)])
    return Pipeline([("preprocess", prep), ("model", model)])

def evaluate(y, pred, prob):
    return {"precision": float(precision_score(y,pred,zero_division=0)), "recall": float(recall_score(y,pred,zero_division=0)), "f1": float(f1_score(y,pred,zero_division=0)), "roc_auc": float(roc_auc_score(y,prob)) if len(set(y))==2 else None, "accuracy": float(accuracy_score(y,pred)), "confusion_matrix": confusion_matrix(y,pred).tolist()}

def train(data_path: Path, artifacts: Path, tracking_uri: str | None = None) -> dict:
    d = pd.read_csv(data_path, parse_dates=["service_date"])
    metadata_path=data_path.parent/"dataset_metadata.json"
    dataset_metadata=json.loads(metadata_path.read_text()) if metadata_path.exists() else {"source_kind":"UNSPECIFIED"}
    temp_setting=os.getenv("PULSEROUTE_TMP")
    if temp_setting:
        Path(temp_setting).mkdir(parents=True,exist_ok=True)
        tempfile.tempdir=str(Path(temp_setting).resolve())
    d = d.dropna(subset=["delayed"]).sort_values("service_date")
    cutoff = max(1, int(len(d)*0.8)); tr, te = d.iloc[:cutoff], d.iloc[cutoff:]
    if len(te)<10 or tr.delayed.nunique()<2 or te.delayed.nunique()<2: raise ValueError("Need at least 10 holdout rows and both classes in temporal train and holdout; try a larger acquisition window, for example acquire --days 90")
    candidates = {"majority": DummyClassifier(strategy="prior"), "logistic_regression": LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42), "random_forest": RandomForestClassifier(n_estimators=80,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=1), "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100,random_state=42)}
    results = {}
    mlflow = None
    try:
        import mlflow
        default_uri=f"sqlite:///{Path('mlflow.db').resolve().as_posix()}"
        mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", default_uri))
        mlflow.set_experiment("PulseRoute Transit Delay")
    except ImportError: mlflow = None
    for name, model in candidates.items():
        pipe = make_pipeline(model); pipe.fit(tr[FEATURES],tr.delayed.astype(int))
        prob = pipe.predict_proba(te[FEATURES])[:,1]; pred=(prob>=0.5).astype(int)
        metrics=evaluate(te.delayed.astype(int),pred,prob); results[name]=metrics
        if mlflow:
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params({"model":name,"seed":42,"temporal_train_rows":len(tr),"holdout_rows":len(te),"feature_count":len(FEATURES)})
                mlflow.log_metrics({k:v for k,v in metrics.items() if isinstance(v,(int,float)) and v is not None})
                mlflow.log_dict({"features":FEATURES,"dataset":str(data_path),"dataset_source_kind":dataset_metadata.get("source_kind","UNSPECIFIED"),"holdout_start":str(te.service_date.min()),"confusion_matrix":metrics["confusion_matrix"]},"evaluation.json")
                mlflow.sklearn.log_model(pipe, name="model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE, pip_requirements=[])
                results[name]["mlflow_run_id"]=run.info.run_id
        results[name]["pipeline"]=pipe
    winner=max(results,key=lambda n: results[n]["f1"])
    artifacts.mkdir(parents=True,exist_ok=True)
    version_path=artifacts/f"model_v{(len(json.loads((artifacts/'registry.json').read_text())['versions'])+1) if (artifacts/'registry.json').exists() else 1}.joblib"
    model_path=version_path; joblib.dump(results[winner]["pipeline"],model_path)
    import shutil
    shutil.copy2(model_path,artifacts/"candidate.joblib")
    sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    report={"winner":winner,"dataset_source_kind":dataset_metadata.get("source_kind","UNSPECIFIED"),"metrics":{n:{k:v for k,v in m.items() if k!="pipeline"} for n,m in results.items()},"features":FEATURES,"target":"actual stop arrival more than 5 minutes after scheduled arrival","train_rows":len(tr),"holdout_rows":len(te),"holdout_start":str(te.service_date.min()),"sha256":sha,"status":"Candidate"}
    report["environment"]={"python":platform.python_version(),"scikit_learn":__import__("sklearn").__version__,"numpy":np.__version__,"pandas":pd.__version__,"scipy":__import__("scipy").__version__,"joblib":joblib.__version__}
    (artifacts/version_path.with_suffix(".metadata.json").name).write_text(json.dumps(report["environment"],indent=2),encoding="utf-8")
    (artifacts/"evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    registry=artifacts/"registry.json"
    if not registry.exists(): registry.write_text(json.dumps({"versions":[],"history":[]},indent=2),encoding="utf-8")
    state=json.loads(registry.read_text()); version=len(state["versions"])+1
    state["versions"].append({"version":version,"model":version_path.name,"sha256":sha,"status":"Candidate","metrics":report["metrics"][winner],"created_at":pd.Timestamp.utcnow().isoformat()})
    registry.write_text(json.dumps(state,indent=2),encoding="utf-8")
    return report

def promote(artifacts: Path, approve: bool, rollback: bool=False) -> dict:
    registry=artifacts/"registry.json"; state=json.loads(registry.read_text())
    versions=state["versions"]
    if rollback:
        champs=[v for v in versions if v["status"]=="Archived"]
        if not champs: raise ValueError("No archived champion available for rollback")
        selected=champs[-1]
    else:
        cand=next((v for v in reversed(versions) if v["status"]=="Candidate"),None)
        if cand is None: raise ValueError("No candidate to promote")
        champion=next((v for v in reversed(versions) if v["status"]=="Champion"),None)
        comparison={"candidate_f1":cand["metrics"].get("f1"),"champion_f1":champion["metrics"].get("f1") if champion else None}
        comparison["candidate_meets_or_exceeds_champion"] = champion is None or comparison["candidate_f1"] >= comparison["champion_f1"]
        if not approve: return {"status":"approval_required","candidate_version":cand["version"],"comparison":comparison}
        if not comparison["candidate_meets_or_exceeds_champion"]:
            cand["status"]="Rejected"; state["history"].append({"action":"reject","version":cand["version"],"at":pd.Timestamp.utcnow().isoformat(),"reason":"candidate F1 below champion","comparison":comparison}); registry.write_text(json.dumps(state,indent=2),encoding="utf-8")
            return {"status":"Rejected","version":cand["version"],"comparison":comparison}
        selected=cand
    for v in versions:
        if v["status"]=="Champion": v["status"]="Archived"
    selected["status"]="Champion"
    import shutil
    shutil.copy2(artifacts/selected["model"], artifacts/"champion.joblib")
    metadata_source=artifacts/selected["model"].replace(".joblib",".metadata.json")
    if (artifacts/metadata_source).exists(): shutil.copy2(artifacts/metadata_source,artifacts/"champion.metadata.json")
    state["history"].append({"action":"rollback" if rollback else "promote","version":selected["version"],"at":pd.Timestamp.utcnow().isoformat(),"reason":"operator-approved candidate met champion F1 gate" if approve else "operator rollback"})
    registry.write_text(json.dumps(state,indent=2),encoding="utf-8")
    return {"status":"Champion","version":selected["version"],"sha256":selected["sha256"]}
