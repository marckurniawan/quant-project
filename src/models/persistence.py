from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier


def save_model(model: RandomForestClassifier, path: str) -> None:
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str) -> RandomForestClassifier:
    path = Path(path)

    return joblib.load(path)