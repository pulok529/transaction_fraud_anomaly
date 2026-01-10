import argparse, os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURES=["amount","hour","country_risk","merchant_risk"]

def main(data_path="data/tx.csv"):
    if not os.path.exists(data_path):
        from make_synthetic import main as gen
        gen(out_path=data_path)
    df=pd.read_csv(data_path)
    X=df[FEATURES]
    y=df["label"].astype(int)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model=IsolationForest(n_estimators=300, random_state=42, contamination=0.02)
    model.fit(X_train)

    # anomaly score: higher means more anomalous
    score = -model.score_samples(X_valid)
    auc=roc_auc_score(y_valid, score)
    ap=average_precision_score(y_valid, score)
    print(f"AUC={auc:.3f}  PR-AUC={ap:.3f}")

    joblib.dump(model, "model.joblib")
    print("Saved model.joblib")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tx.csv")
    args=ap.parse_args()
    main(args.data)
