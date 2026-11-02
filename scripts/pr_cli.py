from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd
from pulseroute.data import acquire, validate
from pulseroute.training import train, promote
from pulseroute.drift import detect, simulate

p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
for name in ["acquire","demo-data","validate","train","promote","rollback","drift","simulate-drift"]: sub.add_parser(name)
sub.choices["acquire"].add_argument("--days",type=int,default=30,help="Number of latest published service dates to download")
args=p.parse_args(); data=Path(os.getenv("PULSEROUTE_DATA","data/processed/transit.csv")); artifacts=Path(os.getenv("PULSEROUTE_ARTIFACTS","artifacts"))
if args.cmd=="acquire": result={"rows":len(acquire(data.parent,args.days))}
elif args.cmd=="demo-data":
    import numpy as np, json
    rng=np.random.default_rng(2026); n=5000; dates=pd.date_range("2024-01-01",periods=n,freq="h")
    frame=pd.DataFrame({"service_date":dates,"route_id":rng.choice(["Red","Orange","Blue","Green-B","Green-C"],n),"direction_id":rng.choice(["0","1"],n),"stop_sequence":rng.integers(1,40,n),"hour":dates.hour,"day_of_week":dates.dayofweek,"month":dates.month})
    z=-2.5+0.12*frame.hour+0.035*frame.stop_sequence+0.25*(frame.day_of_week>=5)+rng.normal(0,0.8,n)
    frame["delayed"]=(rng.random(n)<(1/(1+np.exp(-z)))).astype(int); frame["delay_minutes"]=frame.delayed*(5+rng.gamma(2,3,n))+rng.normal(0,1,n)
    data.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(data,index=False)
    (data.parent/"dataset_metadata.json").write_text(json.dumps({"source_kind":"SYNTHETIC_DEVELOPMENT_FALLBACK","seed":2026,"rows":len(frame),"note":"Generated observations for local development only; not transit records."},indent=2))
    result={"rows":len(frame),"source_kind":"SYNTHETIC_DEVELOPMENT_FALLBACK"}
elif args.cmd=="validate": result=validate(data)
elif args.cmd=="train": result=train(data,artifacts)
elif args.cmd=="promote": result=promote(artifacts,True)
elif args.cmd=="rollback": result=promote(artifacts,True,True)
elif args.cmd=="drift": result=detect(Path(os.getenv("PULSEROUTE_REFERENCE",str(data))),Path(os.getenv("PULSEROUTE_CURRENT",str(data))),Path("monitoring/reports/drift.json"))
else: result=simulate(data,Path("data/current_drift_demo.csv"),Path("monitoring/reports/drift_demo.json"))
print(json.dumps(result,indent=2,default=str))
