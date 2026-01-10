import numpy as np, pandas as pd, os

def main(out_path="data/tx.csv", n=30000, seed=9):
    rng=np.random.default_rng(seed)
    amount = rng.lognormal(mean=4.0, sigma=0.6, size=n).clip(1, 20000)
    hour = rng.integers(0,24,size=n)
    country_risk = rng.beta(2,8,size=n)   # mostly low
    merchant_risk = rng.beta(2,8,size=n)

    # inject anomalies
    is_fraud = np.zeros(n, dtype=int)
    fraud_idx = rng.choice(n, size=int(0.02*n), replace=False)
    is_fraud[fraud_idx]=1
    amount[fraud_idx] = rng.lognormal(mean=6.0, sigma=0.5, size=len(fraud_idx)).clip(50, 60000)
    hour[fraud_idx] = rng.choice([0,1,2,3,4,23], size=len(fraud_idx))
    country_risk[fraud_idx] = rng.beta(6,2,size=len(fraud_idx))
    merchant_risk[fraud_idx] = rng.beta(6,2,size=len(fraud_idx))

    df=pd.DataFrame({
        "amount": amount,
        "hour": hour,
        "country_risk": country_risk,
        "merchant_risk": merchant_risk,
        "label": is_fraud
    })
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}. Fraud rate={df['label'].mean():.3f}")

if __name__=="__main__":
    main()
