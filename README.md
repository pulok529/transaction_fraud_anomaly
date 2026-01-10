# Transaction Fraud / Anomaly Detection - Starter Project

Unsupervised + weakly-supervised anomaly detection for transaction data.
- synthetic transaction generator
- Isolation Forest baseline
- simple thresholding + evaluation

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python score.py --amount 1200 --country_risk 0.7 --merchant_risk 0.8 --hour 2
```
