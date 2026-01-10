import argparse, joblib
import pandas as pd

FEATURES=["amount","hour","country_risk","merchant_risk"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--amount", type=float, required=True)
    ap.add_argument("--hour", type=int, required=True)
    ap.add_argument("--country_risk", type=float, required=True)
    ap.add_argument("--merchant_risk", type=float, required=True)
    args=ap.parse_args()

    model=joblib.load("model.joblib")
    X=pd.DataFrame([{k:getattr(args,k) for k in FEATURES}])
    score = -model.score_samples(X)[0]
    print(f"Anomaly score (higher=more suspicious): {score:.4f}")

if __name__=="__main__":
    main()
