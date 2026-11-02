"""FastAPI inference service with Prometheus metrics."""
from __future__ import annotations
import os, time
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

app=FastAPI(title="PulseRoute AI",version="0.1.0",description="MBTA subway stop delay risk demo")
REQ=Counter("pulseroute_http_requests_total","HTTP requests","method","path","status")
ERROR=Counter("pulseroute_http_errors_total","HTTP error responses","method","path","status")
PRED=Counter("pulseroute_predictions_total","Predictions served",["outcome"])
FAIL=Counter("pulseroute_prediction_failures_total","Prediction failures")
LAT=Histogram("pulseroute_request_duration_seconds","Request latency",["path"])
READY=Gauge("pulseroute_service_ready","Service readiness")
VERSION=Gauge("pulseroute_model_version","Loaded model version",["version"])
app.mount("/metrics",make_asgi_app())

def model_path(): return Path(os.getenv("PULSEROUTE_MODEL","artifacts/champion.joblib"))
def get_model():
    p=model_path()
    if not p.exists(): raise RuntimeError(f"Model not found at {p}; train then promote a model")
    return joblib.load(p)

class Journey(BaseModel):
    route_id: str = Field(min_length=1,max_length=40)
    direction_id: str = "0"
    stop_sequence: int = Field(ge=1,le=300)
    hour: int = Field(ge=0,le=23)
    day_of_week: int = Field(ge=0,le=6)
    month: int = Field(ge=1,le=12)

class MetricsMiddleware:
    def __init__(self, app): self.app=app
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http": return await self.app(scope,receive,send)
        started=time.perf_counter(); status=[500]
        async def wrapped(message):
            if message["type"]=="http.response.start": status[0]=message["status"]
            await send(message)
        await self.app(scope,receive,wrapped)
        route=scope.get("route"); path=getattr(route,"path","unmatched"); method=scope.get("method","GET")
        REQ.labels(method,path,str(status[0])).inc()
        if status[0]>=400: ERROR.labels(method,path,str(status[0])).inc()
        LAT.labels(path).observe(time.perf_counter()-started)

app.add_middleware(MetricsMiddleware)

@app.get("/health")
def health():
    ready=model_path().exists(); READY.set(1 if ready else 0)
    return {"status":"ready" if ready else "model_unavailable","model_path":str(model_path())}

@app.get("/model")
def model_info():
    p=model_path()
    if not p.exists(): raise HTTPException(503,"No promoted model is available")
    import json
    reg=p.parent/"registry.json"; version="unknown"
    if reg.exists():
        state=json.loads(reg.read_text()); champion=[x for x in state["versions"] if x["status"]=="Champion"]
        if champion: version=str(champion[-1]["version"])
    VERSION.labels(version).set(1)
    return {"version":version,"model_path":str(p)}

@app.post("/predict")
def predict(journey: Journey):
    try:
        pipe=get_model(); row=pd.DataFrame([journey.model_dump()]); probability=float(pipe.predict_proba(row)[0,1]); label=int(probability>=0.5)
        PRED.labels("delayed" if label else "on_time").inc()
        return {"delayed_probability":probability,"predicted_delayed":bool(label),"threshold_minutes":float(os.getenv("PULSEROUTE_DELAY_THRESHOLD_MIN","5")),"model_version":model_info()["version"]}
    except HTTPException: raise
    except Exception as exc:
        FAIL.inc(); raise HTTPException(503,f"Prediction unavailable: {exc}") from exc
