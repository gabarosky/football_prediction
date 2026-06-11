from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]

def load_elo():
    return joblib.load(
        ROOT / "saved_models" / "elo_model_v1.pkl"
    )

def load_dc():
    return joblib.load(
        ROOT / "saved_models" / "dixoncoles_model_v1.pkl"
    )