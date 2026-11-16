"""fraud_gbm.py -- Card-not-present fraud detection (gradient boosting).
Illustrative sample model; the transactions file is not included in this repository.
Owner: Fraud Analytics.
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

FEATURES = ["amount", "hour", "merchant_risk", "txn_count_24h", "distance_from_home",
            "is_foreign", "device_age_days"]
TARGET = "is_fraud"
THRESHOLD = 0.80          # block the transaction if score >= THRESHOLD

df = pd.read_csv("transactions.csv")

model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=7)
model.fit(df[FEATURES], df[TARGET])

df["fraud_score"] = model.predict_proba(df[FEATURES])[:, 1]
df["block"] = (df["fraud_score"] >= THRESHOLD).astype(int)
df.to_csv("scored_transactions.csv", index=False)
