"""Acquire and validate MBTA LAMP subway stop performance observations."""
from __future__ import annotations
import json
import os
import urllib.request
from pathlib import Path
import pandas as pd

BASE = "https://performancedata.mbta.com/lamp/subway-on-time-performance-v1"
INDEX = f"{BASE}/index.csv"

def acquire(output: Path, limit: int = 30) -> pd.DataFrame:
    """Download recent service-date parquet partitions listed by the MBTA."""
    output.mkdir(parents=True, exist_ok=True)
    index_path = output.parent / "raw" / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(INDEX, index_path)
    index = pd.read_csv(index_path)
    if index.empty:
        raise ValueError("MBTA published no service-date partitions in index.csv")
    # Index can contain a path column or a direct date column; retain published entries.
    col = next((c for c in index.columns if "path" in c.lower() or "file" in c.lower()), index.columns[-1])
    paths = index[col].dropna().astype(str).tolist()[-limit:]
    parts = []
    for item in paths:
        url = item if item.startswith("http") else f"{BASE}/{item.lstrip('/')}"
        try:
            parts.append(pd.read_parquet(url))
        except Exception:
            local = output.parent / "raw" / Path(url).name
            urllib.request.urlretrieve(url, local)
            parts.append(pd.read_parquet(local))
    if not parts:
        raise ValueError("No MBTA partitions could be downloaded")
    frame = pd.concat(parts, ignore_index=True)
    required = {"service_date", "start_time", "route_id", "direction_id", "stop_sequence", "scheduled_arrival_time", "stop_timestamp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MBTA schema missing required columns: {sorted(missing)}")
    delay_threshold=float(os.getenv("PULSEROUTE_DELAY_THRESHOLD_MIN","5"))
    frame = prepare(frame, delay_threshold)
    frame.to_csv(output / "transit.csv", index=False)
    metadata = {"source_kind":"MBTA_LAMP_REAL", "source": INDEX, "license": "MassDOT Developers License Agreement: https://www.mass.gov/doc/developers-license-agreement-11132009/download (review and accept current terms before use or redistribution)", "delay_threshold_minutes":delay_threshold, "rows": len(frame), "date_min": str(frame.service_date.min()), "date_max": str(frame.service_date.max()), "target_positive_rate": float(frame.delayed.mean())}
    (output / "dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return frame

def prepare(raw: pd.DataFrame, delay_threshold_min: float = 5.0) -> pd.DataFrame:
    """Construct schedule-only features and a late label from observed arrival."""
    d = raw.copy()
    date_values=d["service_date"]
    date_numbers=pd.to_numeric(date_values,errors="coerce")
    compact_date=date_numbers.between(19000101,29991231)
    parsed_date=pd.Series(pd.NaT,index=d.index,dtype="datetime64[ns]")
    if compact_date.any():
        parsed_date.loc[compact_date]=pd.to_datetime(date_numbers.loc[compact_date].astype("int64").astype(str),format="%Y%m%d",errors="coerce").to_numpy()
    if (~compact_date).any():
        parsed_date.loc[~compact_date]=pd.to_datetime(date_values.loc[~compact_date],errors="coerce").to_numpy()
    d["service_date"] = parsed_date
    d["scheduled_arrival_time"] = pd.to_numeric(d.scheduled_arrival_time, errors="coerce")
    stop_values=d["stop_timestamp"]
    if pd.api.types.is_datetime64_any_dtype(stop_values):
        parsed_stop=pd.to_datetime(stop_values,utc=True,errors="coerce")
        stop_epoch=parsed_stop.astype("int64").where(parsed_stop.notna()).astype(float)/1_000_000_000
    else:
        stop_epoch=pd.to_numeric(stop_values,errors="coerce")
        needs_datetime=stop_epoch.isna() & stop_values.notna()
        if needs_datetime.any():
            parsed_stop=pd.to_datetime(stop_values.loc[needs_datetime],utc=True,errors="coerce")
            stop_epoch.loc[needs_datetime]=parsed_stop.astype("int64").where(parsed_stop.notna()).astype(float)/1_000_000_000
    # Accept millisecond epochs as well as the documented second-based timestamps.
    stop_epoch.loc[stop_epoch.abs()>100_000_000_000] /= 1000
    d["stop_timestamp"] = stop_epoch
    d = d.dropna(subset=["service_date", "scheduled_arrival_time", "stop_timestamp"])
    scheduled_local=d.service_date + pd.to_timedelta(d.scheduled_arrival_time,unit="s")
    scheduled_aware=scheduled_local.dt.tz_localize("America/New_York",ambiguous="NaT",nonexistent="shift_forward")
    valid_schedule=scheduled_aware.notna()
    d=d.loc[valid_schedule].copy()
    scheduled_epoch=scheduled_aware.loc[valid_schedule].astype("int64") // 10**9
    d["delay_minutes"] = (d.stop_timestamp - scheduled_epoch) / 60.0
    d["hour"] = (d.scheduled_arrival_time // 3600).astype(int) % 24
    d["day_of_week"] = d.service_date.dt.dayofweek
    d["month"] = d.service_date.dt.month
    d["stop_sequence"] = pd.to_numeric(d.stop_sequence, errors="coerce")
    direction=d.direction_id
    if pd.api.types.is_bool_dtype(direction):
        d["direction_id"]=direction.astype(int).astype(str)
    else:
        d["direction_id"]=direction.astype(str).replace({"False":"0","True":"1"})
    d["delayed"] = (d.delay_minutes > delay_threshold_min).astype(int)
    keep = ["service_date", "route_id", "direction_id", "stop_sequence", "hour", "day_of_week", "month", "delayed", "delay_minutes"]
    result = d[keep].drop_duplicates().sort_values("service_date").reset_index(drop=True)
    if len(result) < 50:
        raise ValueError(f"Only {len(result)} valid arrival rows remained after schema and timestamp checks; increase the acquisition window or inspect the MBTA export")
    return result

def validate(path: Path) -> dict:
    d = pd.read_csv(path, parse_dates=["service_date"])
    required = {"service_date", "route_id", "hour", "day_of_week", "delayed", "delay_minutes"}
    if not required.issubset(d.columns): raise ValueError(f"Missing columns: {required-set(d.columns)}")
    return {"rows": len(d), "duplicates": int(d.duplicated().sum()), "missing_cells": int(d.isna().sum().sum()), "target_distribution": d.delayed.value_counts(normalize=True).sort_index().to_dict(), "date_min": str(d.service_date.min()), "date_max": str(d.service_date.max())}
