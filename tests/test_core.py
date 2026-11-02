import datetime, zoneinfo
import numpy as np
import pandas as pd
from pulseroute.data import prepare
from pulseroute.training import evaluate
from pulseroute.training import train, promote
from pulseroute.drift import simulate
from pulseroute import api

def test_prepare_constructs_actual_schedule_delay_label():
    midnight=datetime.datetime(2024,1,1,tzinfo=zoneinfo.ZoneInfo("America/New_York")).timestamp()
    raw=pd.DataFrame({"service_date":[20240101]*60,"start_time":[0]*60,"route_id":["Red"]*60,"direction_id":[False]*60,"stop_sequence":list(range(1,61)),"scheduled_arrival_time":[36000]*60,"stop_timestamp":[int(midnight+36000+(i%2)*600) for i in range(60)]})
    out=prepare(raw); assert len(out)==60; assert out.delayed.nunique()==2
    raw.service_date=["2024-01-01"]*60
    iso_out=prepare(raw); assert iso_out.service_date.notna().all(); assert iso_out.direction_id.isin(["0","1"]).all()

def test_drift_detects_controlled_copy(tmp_path):
    rng=np.random.default_rng(42); n=100
    a=pd.DataFrame({"hour":rng.integers(0,5,n),"day_of_week":rng.integers(0,7,n),"month":rng.integers(1,13,n),"stop_sequence":rng.integers(1,30,n)})
    a.to_csv(tmp_path/"ref.csv",index=False)
    result=simulate(tmp_path/"ref.csv",tmp_path/"copy.csv",tmp_path/"report.json")
    assert result["drift_detected"] and result["model_degradation"]=="unknown"

def test_metrics_and_api_health_without_model(tmp_path,monkeypatch):
    assert evaluate([0,1],[0,1],[.1,.9])["roc_auc"]==1.0
    monkeypatch.setattr(api,"model_path",lambda:tmp_path/"missing.joblib")
    assert api.health()["status"]=="model_unavailable"
    from prometheus_client import generate_latest
    assert b"pulseroute_http_requests_total" in generate_latest()

def test_candidate_approval_promotion_and_rollback(tmp_path,monkeypatch):
    # Deterministic behavior fixture only; not evidence of real-world model quality.
    dates=pd.date_range("2024-01-01",periods=160)
    frame=pd.DataFrame({"service_date":dates,"route_id":["Red","Blue"]*80,"direction_id":["0","1"]*80,"stop_sequence":np.arange(160)%30+1,"hour":np.arange(160)%24,"day_of_week":dates.dayofweek,"month":dates.month,"delayed":np.arange(160)%2,"delay_minutes":np.arange(160)%12})
    source=tmp_path/"fixture.csv"; frame.to_csv(source,index=False)
    import tempfile
    monkeypatch.setattr(tempfile,"tempdir",str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI",f"sqlite:///{(tmp_path/'mlflow.db').as_posix()}")
    artifacts=tmp_path/"artifacts"; train(source,artifacts)
    assert promote(artifacts,False)["status"]=="approval_required"
    assert promote(artifacts,True)["status"]=="Champion"
    train(source,artifacts); promoted=promote(artifacts,True)
    assert promoted["status"]=="Champion"
    assert promote(artifacts,True,rollback=True)["status"]=="Champion"
