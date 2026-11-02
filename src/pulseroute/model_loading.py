"""Model loading with an explicit scikit-learn version compatibility check."""
from __future__ import annotations
import json
import warnings
import platform
from pathlib import Path
import joblib
import sklearn
import numpy
import pandas
import scipy
from sklearn.exceptions import InconsistentVersionWarning

def load_model(path: Path):
    """Load only a trusted joblib artifact and reject a mismatched sklearn build."""
    path=Path(path)
    metadata_path=path.with_suffix(".metadata.json")
    if metadata_path.exists():
        metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        current={"python":".".join(platform.python_version().split(".")[:2]),"scikit_learn":sklearn.__version__,"numpy":numpy.__version__,"pandas":pandas.__version__,"scipy":scipy.__version__,"joblib":joblib.__version__}
        trained=metadata
        expected={key:(".".join(value.split(".")[:2]) if key=="python" else value) for key,value in trained.items() if key in current}
        mismatches={key:(expected[key],current[key]) for key in expected if expected[key] != current[key]}
        if mismatches:
            detail=", ".join(f"{name} trained={old}, runtime={new}" for name,(old,new) in mismatches.items())
            raise RuntimeError(f"Model/runtime dependency mismatch ({detail}). Use the pinned requirements and matching Python version, then restart or redeploy.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error",InconsistentVersionWarning)
            return joblib.load(path)
    except InconsistentVersionWarning as exc:
        raise RuntimeError(f"This model was trained with scikit-learn {exc.original_sklearn_version}, but the app is running {sklearn.__version__}. Install the pinned project requirements and restart or redeploy.") from exc
