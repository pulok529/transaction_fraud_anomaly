"""Transaction fraud and anomaly detection package."""

from fraud_anomaly.data import generate, save_csv
from fraud_anomaly.features import FEATURE_COLUMNS, build_features, features_for_single
from fraud_anomaly.model import load_bundle, score, score_batch

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "generate",
    "save_csv",
    "FEATURE_COLUMNS",
    "build_features",
    "features_for_single",
    "load_bundle",
    "score",
    "score_batch",
]
